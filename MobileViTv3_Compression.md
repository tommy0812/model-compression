# MobileViTv3 Compression — Measured Results

**ImageNet-100 · structured pruning → knowledge distillation → PTQ reconstruction → TensorRT (RTX 4090, Jetson AGX Orin) and Qualcomm AI Hub (Snapdragon NPU)**

*By [Tommy Chen (陳龍懷)](https://linkedin.com/in/tommychen19920812)*

Every number below comes from a `trtexec` log, a Qualcomm AI Hub profile, or a full 5,000-image validation run on the target runtime or device.

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
        ▼  Post-training quantization, Q/DQ placement chosen per target
   TensorRT : INT8 on Conv / Linear + conv residual adds, the rest in FP16
   NPU      : every tensor fixed-point — w8a8 or w8a16, INT32 bias
   Reconstruction: QDrop · APHQ-ViT · LS-ViT (block-wise)
   Baselines: TensorRT implicit PTQ · NVIDIA ModelOpt · AI Hub's own quantizer
        │
        ▼  Export & build
   BN folding → static QDQ ONNX ─┬─ trtexec          → TensorRT engine
                                 └─ AI Hub compile   → QNN DLC
        │
        ▼
RTX 4090 · Jetson AGX Orin (GPU)   |   Galaxy S25 Ultra (Hexagon NPU)
```

---

## TL;DR

| | |
|---|---|
| **Pruning** | 1.91 M → **0.88 M** params (**−54%**), 1,012 M → **350 M** MACs (**−65%**) |
| **Accuracy at the XXS budget** | **82.20%** Top-1 — **+1.22 pp** over the XXS-KD baseline |
| **P3B alone** | **+0.84 pp above the FP32 teacher** while removing 21% of its parameters |
| **TensorRT INT8** | Reconstruction engines: **+4.5–4.8 pp Top-1 over NVIDIA ModelOpt** and **4–11% lower latency than FP16** on RTX 4090 / Jetson AGX Orin, at most 0.42 pp below FP32 |
| **Snapdragon NPU, w8a8** | **+1.4–1.5 pp Top-1 over AI Hub's own quantizer** at near-equal latency; **38% lower latency than FP16** for a 0.6–0.7 pp drop |
| **Snapdragon NPU, w8a16** | **FP16 accuracy** (within 0.04 pp) at **29% lower latency** |
| **INT8 coverage (TensorRT)** | **95% of layers (77/81)**, residual adds included — against 53% (42/80) for TensorRT's implicit PTQ |

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
| Quantization | Per-channel symmetric INT8 weights, per-tensor activations. **TensorRT:** symmetric INT8 on Conv / Linear inputs and the conv residual adds; attention matmuls, LayerNorm and Softmax stay FP16. **NPU:** every tensor quantized — uint8 (w8a8) or uint16 (w8a16) activations with a zero-point, INT32 bias, symmetric `to_qkv` output |
| Calibration set | 2,500 images (25 per class) drawn once from the train split with a fixed seed, evaluation transform, no augmentation — the same set for every method, including AI Hub's quantizer |
| Reconstruction | Block-wise, 13 blocks × 20,000 iterations, batch 32, QDrop input mixing for all three methods |
| Export | BN folding → static QDQ ONNX, one per target placement |
| Engine build (TensorRT) | `trtexec`, TensorRT 10.10 (RTX 4090) / 10.3 (Jetson AGX Orin); each engine built 3 times — among builds within 0.5% of the fastest, the most accurate is kept |
| Compile (NPU) | Qualcomm AI Hub compile job → QNN DLC, run on a Samsung Galaxy S25 Ultra (Snapdragon 8 Elite); every layer runs on the Hexagon NPU |
| Benchmark | TensorRT: batch 1, static shape `1x3x256x256`, CUDA graph, mean GPU compute time — median of 3 rounds (10 on Jetson AGX Orin, where run-to-run variance is higher). NPU: batch 1, median of 100 on-device runs (AI Hub profile) |
| Accuracy | Top-1 on the full 5,000-image validation split — through the TensorRT runtime on each GPU, and on the phone itself for the NPU. PyTorch runs with TF32 disabled, so FP32 numbers match ONNX Runtime and TensorRT FP32 |

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

The pruned model is retrained from scratch with the same distillation recipe and epochs as the XXS-KD baseline.
This pruned model (0.88 M params, 350 M MACs) is the one quantized and deployed below.

---

## 2 · Quantization Reconstruction

Each target has its own Q/DQ placement, so each has its own PTQ starting point; within a target, all
three methods share that starting point and the same calibration set — only the objective that
re-optimises weight rounding differs. PTQ here is our own calibration (per-channel max weights,
MSE-clipped activations) before reconstruction. Top-1 (%), FP32 reference **82.20**.

**TensorRT INT8**

| Method | PTQ | Reconstructed | ONNX (ORT) | RTX 4090 | Jetson AGX Orin |
|---|---:|---:|---:|---:|---:|
| QDrop | 79.48 | 81.98 | 81.96 | **81.98** | **82.00** |
| APHQ-ViT | 79.48 | 81.84 | 81.76 | 81.88 | 81.78 |
| LS-ViT | 79.48 | 81.88 | 81.88 | 81.90 | 81.80 |

**Snapdragon NPU (fully fixed-point)**

| Method | PTQ | Reconstructed | ONNX (ORT) | Galaxy S25 Ultra |
|---|---:|---:|---:|---:|
| QDrop w8a8 | 79.38 | 81.32 | 81.22 | 81.44 |
| APHQ-ViT w8a8 | 79.38 | 81.38 | 81.60 | 81.50 |
| LS-ViT w8a8 | 79.38 | 81.36 | 81.34 | **81.54** |
| QDrop w8a16 | 82.14 | 82.18 | 82.18 | 82.10 |
| APHQ-ViT w8a16 | 82.14 | 82.14 | 82.14 | 82.06 |
| LS-ViT w8a16 | 82.14 | 82.20 | 82.18 | **82.12** |

- Reconstruction recovers **+2.4–2.5 pp** over plain PTQ on TensorRT and **+1.9–2.0 pp** on the NPU's
  w8a8 graph, where every tensor is 8-bit. With 16-bit activations, PTQ is already within 0.06 pp of FP32.
- **The simulation predicts the device.** TensorRT engines land within 0.12 pp of the ONNX simulation.
  On the phone, w8a8 lands within 0.22 pp — in a fully 8-bit graph, one-step rounding differences
  between platforms flip a few predictions — and w8a16 within 0.08 pp.

---

## 3 · Deployment

### TensorRT — RTX 4090 and Jetson AGX Orin

Top-1 (%) · mean GPU compute time, batch 1, `1x3x256x256`. Bold: most accurate INT8 engine per platform.

| Method | RTX 4090 | Jetson AGX Orin |
|---|---|---|
| FP32 | 82.20 · 0.409 ms | 82.20 · 1.491 ms |
| FP16 | 82.20 · 0.267 ms | 82.22 · 0.688 ms |
| TensorRT implicit PTQ | 81.22 · 0.260 ms | 81.26 · 0.657 ms |
| NVIDIA ModelOpt (explicit QDQ) | 77.42 · 0.248 ms | 77.24 · 0.697 ms |
| QDrop | **81.98** · 0.238 ms | **82.00** · 0.658 ms |
| APHQ-ViT | 81.88 · 0.239 ms | 81.78 · 0.656 ms |
| LS-ViT | 81.90 · 0.238 ms | 81.80 · 0.657 ms |

INT8 coverage: the reconstruction engines run **95% of layers in INT8 (77/81)** and ModelOpt 91%
(74/81); TensorRT's implicit PTQ reaches only **53% (42/80)**, 51% (41/80) on Orin.

### Snapdragon NPU — Qualcomm AI Hub, Samsung Galaxy S25 Ultra

Top-1 (%) measured on the phone · median latency of 100 runs, batch 1. Every layer runs on the Hexagon
NPU. "FP16" is the FP32 model compiled without quantization, which the NPU runs in FP16.
"AI Hub quantizer" is AI Hub's own quantize job on the same 2,500 calibration images.

| Method | Top-1 (%) | Latency | vs. FP16 |
|---|---:|---:|---:|
| FP16 | 82.10 | 6.034 ms | — |
| AI Hub quantizer, w8a8 | 80.02 | 3.679 ms | −39.0% |
| AI Hub quantizer, w8a16 | 82.08 | 4.263 ms | −29.4% |
| QDrop w8a8 | 81.44 | 3.741 ms | −38.0% |
| APHQ-ViT w8a8 | 81.50 | 3.745 ms | −37.9% |
| LS-ViT w8a8 | **81.54** | 3.735 ms | −38.1% |
| QDrop w8a16 | 82.10 | 4.283 ms | −29.0% |
| APHQ-ViT w8a16 | 82.06 | 4.281 ms | −29.1% |
| LS-ViT w8a16 | **82.12** | 4.275 ms | −29.2% |

### The same method on every target (QDrop)

| Platform | Runtime, precision | FP16: Top-1 · latency | QDrop: Top-1 · latency | Latency vs. FP16 | Top-1 vs. FP16 |
|---|---|---|---|---:|---:|
| RTX 4090 | TensorRT 10.10, INT8 | 82.20 · 0.267 ms | 81.98 · 0.238 ms | −10.9% | −0.22 |
| Jetson AGX Orin | TensorRT 10.3, INT8 | 82.22 · 0.688 ms | 82.00 · 0.658 ms | −4.4% | −0.22 |
| Galaxy S25 Ultra (NPU) | QNN, w8a8 | 82.10 · 6.034 ms | 81.44 · 3.741 ms | −38.0% | −0.66 |
| Galaxy S25 Ultra (NPU) | QNN, w8a16 | 82.10 · 6.034 ms | 82.10 · 4.283 ms | −29.0% | 0.00 |

**Results**

- **Reconstruction beats each vendor's own quantizer:** +4.5–4.8 pp over NVIDIA ModelOpt on TensorRT,
  and +1.4–1.5 pp over AI Hub's quantizer on the NPU. The three methods are tied with each other.
- **8-bit pays off far more on the NPU:** 38% lower latency than FP16, versus 4–11% on the GPUs;
  w8a16 keeps FP16 accuracy.
- **Q/DQ placement is per target:** TensorRT mixes INT8 and FP16, while the NPU needs a fully
  fixed-point graph.
- **Next bottleneck:** on the NPU, the unfold / fold reshapes around each MobileViT block take most of
  the critical path — data movement that MACs don't count.
