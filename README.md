# Model Compression & Edge Deployment — Results

**MobileViTv3 · ImageNet-100 · PyTorch → ONNX (Explicit QDQ) → TensorRT INT8 → NVIDIA Jetson AGX Orin**

*By [Tommy Chen (陳龍懷)](https://linkedin.com/in/tommychen19920812)*

A complete, measured compression pipeline: structured pruning of a hybrid CNN + Transformer
backbone, accuracy recovery by knowledge distillation, Hessian-aware post-training
quantization with block-wise reconstruction, and INT8 TensorRT deployment benchmarked on
both a desktop GPU and an embedded Jetson platform. Every number below comes from a
`trtexec` log or a full-validation-set run, not from a simulated-quantization estimate.

---

> **Also in this repo:** [Depth Anything 3 — compression of a depth estimation
> model](./DA3_Compression_Results.md) — KITTI fine-tuning, geometry-aware
> knowledge distillation with paired bootstrap confidence intervals, and P3B structured
> pruning toward DA3-Small's compute budget *(in progress)*.

---

## TL;DR

| | |
|---|---|
| **Parameters** | 1.91 M → **0.61 M**  (**−68.1%**) |
| **MACs** | 56.4 M → **17.6 M**  (**−68.8%**) |
| **Accuracy after pruning + KD** | **62.74%** Top-1 — **+5.96 pp** over the XXS-KD baseline |
| **Accuracy after INT8 (Jetson Orin)** | **62.40%** Top-1 / 86.18% Top-5 — **−0.34 pp vs. FP32** |
| **Jetson Orin latency** | **0.864 ms** GPU compute, **1,118 qps** (batch 1) |
| **INT8 coverage** | **96.6%** of compute-bound layers (57/59) vs. 66.7% for the NVIDIA ModelOpt baseline |

---

## Pipeline

```text
MobileViTv3-XS  (teacher)
1.91 M params / 56.4 M MACs / 68.60% Top-1
        │
        ▼  Structured pruning
   P3B  — Transformer block-benefit pruning   (ICCVW 2025)
   GFP  — Group Fisher Pruning, CNN channel pruning  (ICML 2021)
        │
        ▼  Accuracy recovery
   Vanilla KD from the FP32 XS teacher  (alpha = 0.7, T = 2.0)
        │
        ▼
0.61 M params / 17.6 M MACs  (−68.1% / −68.8%)
62.74% Top-1 / 86.14% Top-5          ← FP32 reference for everything below
        │
        ▼  Post-training quantization
   Grid-search PTQ  →  block-wise reconstruction
   (AdaRound / APHQ-ViT / LS-ViT), 5% of the training set as calibration data
        │
        ▼  Export & build
   BN folding → explicit QDQ ONNX → trtexec --fp16 --best
        │
        ▼
TensorRT INT8 engine
RTX 4090      : 62.30% Top-1, 0.322 ms, 96.8% INT8 coverage
Jetson AGX Orin: 62.40% Top-1, 0.864 ms, 96.6% INT8 coverage
```

---

## Experimental Setup

| Item | Value |
|---|---|
| Dataset | ImageNet-100 (Tian et al., CMC, ECCV 2020 — fixed 100-synset subset), 5,000 validation images |
| Input resolution | 64 × 64 |
| Model family | MobileViTv3 (XS teacher, XXS reference) |
| Pruning | P3B block pruning (Transformer stages) + Group Fisher Pruning / GFP (CNN channels) |
| Distillation | Vanilla KD, alpha = 0.7, T = 2.0, FP32 XS teacher |
| Quantization | INT8 W/A; per-channel weights, per-tensor activations; 5% of train set for calibration |
| Export | BN folding → explicit QDQ ONNX |
| Engine build | `trtexec --fp16 --best`, TensorRT 10.3 (Orin) / 10.x (desktop) |
| Benchmark | batch 1, static shape `1x3x64x64`, GPU compute time (mean) from `trtexec` |

---

## 1 · Pruning + Distillation

Goal: prune MobileViTv3-XS down to the **XXS MAC budget** while beating XXS on accuracy.

| Model | Top-1 (%) | Top-5 (%) | Params | MACs | Note |
|---|---:|---:|---:|---:|---|
| XS | 68.60 | 89.10 | 1.91 M | 56.4 M | Teacher |
| XXS | 56.50 | 82.30 | 0.79 M | 17.4 M | Reference |
| XXS + KD | 56.78 | 81.76 | 0.79 M | 17.4 M | **Target baseline** |
| XS + P3B | 68.78 | 89.68 | 1.42 M | 50.5 M | P3B only |
| **XS + P3B + GFP + KD** | **62.74** | **86.14** | **0.61 M** | **17.6 M** | **Proposed** |

**Results**

- **−68.1% parameters / −68.8% MACs** vs. the XS teacher.
- **+5.96 pp Top-1 over XXS-KD** (62.74% vs. 56.78%) and **+6.24 pp over plain XXS**, at the
  same MAC budget — and with **23% fewer parameters** than XXS-KD (0.61 M vs. 0.79 M).
- P3B alone removes **26% of parameters and 10% of MACs while slightly *exceeding* the
  teacher** (68.78% vs. 68.60%), i.e. the block-benefit criterion is finding genuinely
  redundant Transformer capacity, not trading accuracy for size.

---

## 2 · Quantization Reconstruction

All three methods start from the same grid-search PTQ initialisation (62.04%); they differ
only in the block-wise reconstruction objective. The FP32 reference is **62.74%**.

| Method | PTQ | Reconstructed | Deploy (folded) | **TensorRT (final)** |
|---|---:|---:|---:|---:|
| AdaRound | 62.04% | 62.14% | 62.00% | 61.74% |
| APHQ-ViT | 62.04% | 61.92% | 62.04% | 62.04% |
| **LS-ViT** | 62.04% | **62.50%** | **62.42%** | **62.30%** |

**Results (RTX 4090 engine)**

- **LS-ViT: 62.30% Top-1 — +0.56 pp over AdaRound, +0.26 pp over APHQ-ViT.**
- LS-ViT is the only method whose gain **survives the whole export path** (reconstruction →
  BN folding → QDQ ONNX → TensorRT), which is what actually matters for deployment.

---

## 3 · TensorRT Engine Comparison

FP32 baseline: **62.74% Top-1 / 86.14% Top-5**. GPU compute time is the `trtexec` mean at
batch 1. **INT8 coverage** is the fraction of *compute-bound* engine layers (`gemm`,
`CaskConvolution`, `CaskPooling`, `PointWiseV2`) that TensorRT actually scheduled onto an
INT8 kernel, determined from the `TacticName` in `--exportLayerInfo` output — not from the
output tensor format, which is unreliable for conv/gemm layers. Reformat, shape and
elementwise layers are excluded because they carry no INT8 speedup to begin with.

### RTX 4090

| Method | Top-1 | Top-5 | GPU compute | INT8 coverage |
|---|---:|---:|---:|---:|
| ModelOpt (explicit QDQ) | 62.66% | 85.92% | 0.352 ms | 68.1% (47/69) |
| TRT implicit PTQ | 62.02% | 85.16% | 0.300 ms | 48.8% (42/86) |
| AdaRound | 61.74% | 85.68% | 0.322 ms | 96.8% (61/63) |
| APHQ-ViT | 62.04% | 85.92% | 0.332 ms | 96.8% (60/62) |
| **LS-ViT** | **62.30%** | **85.96%** | 0.322 ms | **96.8% (60/62)** |

### Jetson AGX Orin (TensorRT 10.3)

| Method | Top-1 | Top-5 | GPU compute | Throughput | INT8 coverage |
|---|---:|---:|---:|---:|---:|
| ModelOpt (explicit QDQ) | 62.60% | 86.14% | 0.974 ms | 1,000 qps | 66.7% (44/66) |
| TRT implicit PTQ | 61.92% | 85.50% | 0.718 ms | 1,385 qps | 53.8% (42/78) |
| AdaRound | 62.18% | 85.84% | 0.860 ms | 1,128 qps | 96.7% (58/60) |
| APHQ-ViT | 62.20% | 85.72% | 0.862 ms | 1,125 qps | 96.6% (57/59) |
| **LS-ViT** | **62.40%** | **86.18%** | 0.864 ms | 1,118 qps | **96.6% (57/59)** |

**Results**

- **Accuracy is essentially preserved through INT8**: 62.40% vs. 62.74% FP32 = **−0.34 pp
  Top-1**, and Top-5 is **above** FP32 (86.18% vs. 86.14%).
- **11.3% lower latency than the NVIDIA ModelOpt explicit-QDQ baseline** (0.864 ms vs.
  0.974 ms, +11.8% throughput) while running **30 pp more of the compute-bound graph in
  INT8** (57/59 = 96.6% vs. 44/66 = 66.7%) — ModelOpt leaves 22 of its 66 compute layers
  in FP32/FP16, which is exactly where the latency gap comes from.
- TensorRT's own implicit PTQ is the fastest engine (0.718 ms) but reaches INT8 on only
  **53.8%** of compute layers (42/78) and gives up **0.48 pp Top-1** relative to LS-ViT —
  the classic coverage/accuracy trade-off this pipeline is built to avoid.
- The ranking is **consistent across both platforms**, so the reconstruction gains are a
  property of the method, not of one compiler/hardware combination.

---

## 4 · Methods Implemented

Re-implemented from the original papers as modular, composable components.

### Pruning

| Method | Paper | Core idea |
|---|---|---|
| **P3B** | Pruning by Block Benefit (ICCVW 2025) | Task-driven global budget allocation across Transformer blocks + local Taylor-guided selection with soft masking |
| **GFP** | Group Fisher Pruning for Practical Network Compression (ICML 2021) | Second-order Fisher channel importance, one-shot global sorting |
| **ILG** | Slimming Shortcut (ICPR 2020) | Enables pruning of residual paths |
| **GConv** | Efficient Structured Pruning and Architecture Searching for Group Convolution (ICCV 2019) | NAS-driven group-conv structure optimisation |
| **NSP** | Network Slimming (ICCV 2017) | L1 sparsity on BN scaling factors |

### Quantization

| Method | Paper | Core idea |
|---|---|---|
| **AdaRound** | Up or Down? Adaptive Rounding for PTQ (ICML 2020) | Learned per-weight rounding via reconstruction loss + rounding regularisation |
| **APHQ-ViT** | Average Perturbation Hessian Based Reconstruction for ViTs (CVPR 2025) | Hessian-weighted reconstruction + GELU→ReLU MLP reconstruction |
| **LS-ViT** | Least-Squares Hessian Based Block Reconstruction (CVPR 2026) | Empirical-Fisher low-rank Hessian (SVD top-k, >99% memory reduction) with diagonal + low-rank covariance loss |

### In-house quantization framework

| Layer | Weight quantization | Design note |
|---|---|---|
| `QuantConv2d` | Per-channel INT8 | Quantizes its own input only, never its output |
| `QuantLinear` | Per-channel INT8 (grouped) | Independent per-tensor activation quantizer; `to_qkv` uses 3 groups (Q/K/V) |
| `QuantAdd` | — | Independent scales for shortcut vs. branch, so error stays separable |
| `QuantMatMul` | — | Quantizes both A and B inputs; FP32 output tensor |

Supporting machinery: STE-based FakeQuantization, BN folding before quantization,
grid-search (power-law candidates + coordinate descent) and 99.99%-percentile calibration,
and layer-wise precision control.

---

## 5 · Deployment Notes

- **Why explicit QDQ instead of TensorRT's native calibration** — TRT 10.x implicit INT8
  calibration fails on fused Conv+SiLU patterns; inserting explicit ONNX QDQ nodes bypasses
  the compiler fusion issue and gives deterministic, inspectable precision placement.
- **Quantization policy** — quantize compute-bound ops only (Conv / Gemm / MatMul); run ONNX
  shape inference first so the Transformer graph is analysed correctly.
- **Mixed precision** — build with `--fp16 --best`: QDQ regions map to INT8 kernels and the
  remaining layers fall back to FP16 automatically.
- **Verification** — every engine is validated end-to-end on the full 5,000-image validation
  set through the TensorRT runtime, not just on the PyTorch simulated-quantization model.
  Per-layer precision is confirmed by exporting `--exportLayerInfo` and classifying every
  compute-bound layer by its selected tactic, so "INT8 coverage" reflects kernels that
  actually run in INT8 rather than tensors that merely carry an INT8 format.

---

## Caveats

- ImageNet-100 at 64 × 64, not full ImageNet-1k at 224 × 224 — chosen to keep the full
  prune → distill → quantize → deploy loop iterable on available hardware.
- Latency figures are `trtexec` GPU compute time at batch 1 with static shapes; they exclude
  pre/post-processing.
- The latency comparison is against the NVIDIA ModelOpt explicit-QDQ INT8 baseline, i.e.
  INT8-vs-INT8. An FP32/FP16-engine reference on Orin has not been benchmarked yet.


---

## About

I work on making vision models small enough and fast enough to run on embedded hardware —
structured pruning (CNN channels and Transformer blocks), knowledge distillation, PTQ/QAT
with Hessian-aware reconstruction, and TensorRT INT8 deployment on NVIDIA Jetson.

**Contact:** [LinkedIn](https://linkedin.com/in/tommychen19920812) · a0912986300@gmail.com
