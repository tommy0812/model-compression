# MobileViTv3 Compression — Measured Results

**ImageNet-100 · structured pruning → knowledge distillation → INT8 PTQ → TensorRT on RTX 4090 and Jetson AGX Orin**

*By [Tommy Chen (陳龍懷)](https://linkedin.com/in/tommychen19920812)*

Every number below comes from a `trtexec` log or a full 5,000-image validation run through the TensorRT runtime.

```text
MobileViTv3-XS (FP32)
        │
        ▼  Structured pruning
   GFP  — Fisher channel pruning, CNN stages
   P3B  — block-benefit pruning, Transformer stages
        │
        ▼  Accuracy recovery
   Vanilla KD from the FP32 XS teacher
        │
        ▼  Post-training quantization, INT8
   TensorRT implicit PTQ · NVIDIA ModelOpt explicit QDQ
   QDrop · APHQ-ViT · LS-ViT (block-wise reconstruction)
   + QuantAdd — Q/DQ on the residual adds as well
        │
        ▼  Export & build
   BN folding → explicit QDQ ONNX → trtexec
        │
        ▼
TensorRT FP32 / FP16 / INT8 engines
RTX 4090/4050 · NVIDIA Jetson AGX Orin
```

---

## TL;DR

| | |
|---|---|
| **Pruning** | 1.91 M → **0.88 M** params (**−54%**), 1,012 M → **350 M** MACs (**−65%**) |
| **Accuracy at the XXS budget** | **82.20%** Top-1 — **+1.22 pp** over the XXS-KD baseline |
| **P3B alone** | **+0.84 pp above the FP32 teacher** while removing 21% of its parameters |
| **PTQ reconstruction** | **+4.6–4.8 pp Top-1 over NVIDIA ModelOpt** at identical latency and INT8 coverage — **within 0.16 pp of FP32** |
| **INT8 deployment** | With QuantAdd, **1.7–2.3× FP32 speed** and **6–15% faster than FP16** on RTX 4090 / RTX 4050 / Jetson AGX Orin, within 0.34 pp of FP32 |
| **INT8 coverage** | **95% of layers (77/81)** with QuantAdd, 91% without — against 53% (42/80) for TensorRT's implicit PTQ |

---

## Dataset and Experimental Setup

| Item | Value |
|---|---|
| Dataset | ImageNet-100 — 130,000 train / 5,000 validation images, 100 classes |
| Input resolution | 256 × 256 |
| Models | MobileViTv3-XS — the model being compressed, and the FP32 teacher it is distilled from · MobileViTv3-XXS — reference small model |
| Training | 100 epochs, batch 128, mixed precision, seed 42 — the same recipe for every row |
| Augmentation | Random resized crop (256), horizontal flip, colour jitter, ±15° rotation · evaluation: resize 293, centre crop 256 |
| Normalisation | ImageNet channel statistics, shared by training, evaluation and calibration |
| Pruning | P3B block pruning for the Transformer stages + GFP Fisher pruning for the CNN channels |
| Distillation | Vanilla KD from the FP32 XS teacher, α = 0.5, T = 4.0 |
| Quantization | INT8 weights and activations — per-channel weights, per-tensor activations; QuantAdd also quantizes the residual-add inputs |
| Calibration set | 2,500 images (25 per class) drawn once from the train split with a fixed seed, evaluation transform, no augmentation |
| Export | BN folding → ONNX with explicit QDQ nodes |
| Engine build | `trtexec`, TensorRT 10.3 (Jetson AGX Orin) / 10.10 (RTX 4090, RTX 4050); each engine built 3 times, fastest kept |
| Benchmark | batch 1, static shape `1x3x256x256`, CUDA graph, mean GPU compute time — median of 3 rounds (10 on Jetson AGX Orin, where run-to-run variance is higher) |
| Accuracy | Top-1 on the full 5,000-image validation split. PyTorch runs with TF32 disabled, so every FP32 number matches ONNX Runtime and TensorRT FP32; engines are measured through the TensorRT runtime on each platform |

---

## 1 · Pruning + Distillation

Goal: prune MobileViTv3-XS down to the XXS compute budget and beat XXS at that budget.

| Model | Top-1 (%) | Top-5 (%) | Params | MACs |
|---|---:|---:|---:|---:|
| XS — teacher | 82.96 | 96.42 | 1.91 M | 1,012 M |
| XXS | 78.72 | 94.60 | 0.79 M | 351 M |
| XXS + KD — target baseline | 80.98 | 95.70 | 0.79 M | 351 M |
| XS + GFP + KD | 81.06 | 95.56 | 0.62 M | 350 M |
| XS + P3B + KD | **83.80** | 96.74 | 1.51 M | 841 M |
| **XS + GFP + P3B + KD** | **82.20** | **96.02** | 0.88 M | 350 M |

- **At the XXS budget: +1.22 pp Top-1 over the XXS-KD baseline** (82.20 vs. 80.98) at the same MACs —
  i.e. **−54% params and −65% MACs** against the XS teacher for −0.76 pp.
- **Pruning both halves matters.** GFP on the CNN stages alone only ties the baseline (81.06); adding
  P3B on the Transformer stages is worth **+1.14 pp** at the same budget.
- **P3B alone beats the teacher by +0.84 pp** while removing 21% of its parameters — the Transformer
  stages carry real redundancy, which is what makes the combined result possible.

---

## 2 · Quantization Reconstruction

Within each Q/DQ placement, all three methods start from the same PTQ initialisation and the same
calibration set; only the objective that re-optimises weight rounding differs. Top-1 (%), FP32
reference **82.20**.

| Method | PTQ | Reconstructed | Deployed (ONNX) | TensorRT (RTX 4090) |
|---|---:|---:|---:|---:|
| **QDrop** | 79.26 | 81.84 | 81.96 | **82.06** |
| APHQ-ViT | 79.26 | 81.86 | 82.02 | 81.96 |
| LS-ViT | 79.26 | 81.84 | 81.78 | 81.74 |
| QDrop + QuantAdd | 79.18 | 81.94 | 81.92 | 82.00 |
| APHQ-ViT + QuantAdd | 79.18 | 82.06 | 81.94 | 81.82 |
| LS-ViT + QuantAdd | 79.18 | 81.88 | 81.92 | 81.84 |

Reconstruction recovers **+2.6 pp** over plain PTQ (**+2.7–2.9 pp** with QuantAdd), and the gain
survives export and the TensorRT build: the best engine lands **0.14 pp below FP32**.

---

## 3 · Deployment

Top-1 (%) · mean GPU compute time, batch 1, `1x3x256x256`. Most accurate INT8 engine per platform in
bold, fastest engine in italics.

| Method | RTX 4090 | RTX 4050 | Jetson AGX Orin |
|---|---|---|---|
| FP32 | 82.20 · 0.411 ms | 82.20 · 0.712 ms | 82.20 · 1.485 ms |
| FP16 | 82.20 · 0.268 ms | 82.16 · 0.412 ms | 82.20 · 0.695 ms |
| TensorRT implicit PTQ | 81.18 · 0.262 ms | 81.22 · 0.435 ms | 81.12 · 0.663 ms |
| NVIDIA ModelOpt (explicit QDQ) | 77.42 · 0.250 ms | 77.38 · 0.372 ms | 77.22 · 0.705 ms |
| QDrop | **82.06 · 0.250 ms** | **82.06 · 0.359 ms** | 81.96 · 0.704 ms |
| APHQ-ViT | 81.96 · 0.249 ms | 81.92 · 0.369 ms | 81.92 · 0.702 ms |
| LS-ViT | 81.74 · 0.249 ms | 81.88 · 0.359 ms | **82.04 · 0.704 ms** |
| QDrop + QuantAdd | 82.00 · *0.239 ms* | 81.86 · 0.349 ms | 81.92 · *0.651 ms* |
| APHQ-ViT + QuantAdd | 81.82 · *0.239 ms* | 81.82 · 0.360 ms | 81.92 · 0.660 ms |
| LS-ViT + QuantAdd | 81.84 · *0.239 ms* | 81.84 · *0.348 ms* | 81.74 · 0.655 ms |

INT8 coverage: the explicit-QDQ builds run **91% of layers in INT8 (74/81)**, **95% (77/81) with
QuantAdd**; TensorRT's implicit PTQ reaches only **53% (42/80)**, or 45% (40/88) on RTX 4050.

**Results**

- **Reconstruction is accuracy for free.** ModelOpt and the reconstruction methods ship the same
  explicit-QDQ graph, the same 91% INT8 coverage and the same latency — the best reconstruction is
  **+4.6 to +4.8 pp Top-1** on every platform. Only the weight rounding differs.
- **Coverage alone is not enough.** ModelOpt puts 91% of layers in INT8 and is still the *least*
  accurate INT8 engine, 3.8–3.9 pp below implicit PTQ at 53% coverage.
- **The three reconstruction methods land within 0.32 pp of each other, and the best one changes with
  the platform** — QDrop on the RTX GPUs, LS-ViT on Orin. The robust result is the margin over
  ModelOpt, not the ranking among them.
- **Quantizing the residual adds makes INT8 the fastest engine on every platform.** QuantAdd lifts
  coverage from 91% to 95% and cuts latency 2–8% against the same method without it — most on
  Orin — for −0.30 to +0.10 pp Top-1. Given the same residual-add placement on RTX 4090, ModelOpt
  reaches 77.32% at 0.246 ms; QDrop + QuantAdd is **+4.68 pp and 3% faster**.
- **QDrop + QuantAdd lands within 0.34 pp of FP32 at 1.7–2.3× FP32 speed**, 6–15% faster than FP16.
  Without QuantAdd, explicit INT8 was 1% slower than FP16 on Orin, where TensorRT's implicit PTQ was
  the fastest engine; with it, QDrop + QuantAdd overtakes both (0.651 ms vs. 0.663 ms implicit,
  0.695 ms FP16).
