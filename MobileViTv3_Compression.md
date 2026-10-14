# MobileViTv3 Compression — Measured Results

**CIFAR-100 · structured pruning → knowledge distillation → INT8 PTQ → TensorRT on RTX 4090 and Jetson AGX Orin**

*By [Tommy Chen (陳龍懷)](https://linkedin.com/in/tommychen19920812)*

Every number below comes from a `trtexec` log or a full test-split run through the TensorRT runtime.

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
        │
        ▼  Export & build
   BN folding → explicit QDQ ONNX → trtexec
        │
        ▼
TensorRT FP32 / FP16 / INT8 engines
RTX 4090/4050 · NVIDIA Jetson AGX Orin
```

---

## Dataset and Experimental Setup

| Item | Value |
|---|---|
| Dataset | CIFAR-100 — 50,000 train / 10,000 test images, 100 classes |
| Input resolution | 72 × 72, upsampled from the native 32 × 32 (see below) |
| Models | MobileViTv3-XS — the model being compressed, and the FP32 teacher it is distilled from · MobileViTv3-XXS — reference small model |
| Augmentation | Random crop (32, padding 4) and horizontal flip applied at the native resolution, then resized to 72 |
| Normalisation | CIFAR-100 channel statistics, shared by training, evaluation and calibration |
| Pruning | P3B block pruning for the Transformer stages + GFP Fisher pruning for the CNN channels |
| Distillation | Vanilla KD from the FP32 XS teacher, α = 0.5, T = 4.0 |
| Quantization | INT8 weights and activations — per-channel weights, per-tensor activations |
| Calibration set | 2,500 images (25 per class) drawn once from the train split with a fixed seed, evaluation transform, no augmentation |
| Export | BN folding → ONNX with explicit QDQ nodes |
| Engine build | `trtexec`, TensorRT 10.3 (Jetson AGX Orin) / 10.10 (RTX 4090, RTX 4050) |
| Benchmark | batch 1, static shape `1x3x72x72`, mean GPU compute time from `trtexec` |
| Accuracy | Top-1 on the full 10,000-image test split, measured through the TensorRT runtime |

---

## 1 · Pruning + Distillation

Goal: prune MobileViTv3-XS down to the XXS compute budget and beat XXS at that budget.

| Model | Top-1 (%) | Top-5 (%) | Params | MACs |
|---|---:|---:|---:|---:|
| XS — teacher | 68.38 | 90.10 | 1.91 M | 90.2 M |
| XXS | 60.46 | 85.30 | 0.79 M | 30.3 M |
| XXS + KD — target baseline | 64.89 | 87.96 | 0.79 M | 30.3 M |
| XS + GFP + KD | 65.43 | 88.86 | 0.79 M | 29.9 M |
| XS + P3B + KD | **70.62** | 91.38 | 1.38 M | 75.2 M |
| **XS + GFP + P3B + KD** | **67.02** | **89.30** | 0.82 M | 30.1 M |

- **At the XXS budget: +2.13 pp Top-1 over the XXS-KD baseline** (67.02 vs. 64.89) at the same MACs —
  i.e. **−57% params and −67% MACs** against the XS teacher for −1.36 pp.
- **Pruning both halves matters.** GFP on the CNN stages alone reaches 65.43; adding P3B on the
  Transformer stages is worth **+1.59 pp** at the same budget.
- **P3B alone beats the teacher by +2.24 pp** while removing 28% of its parameters — the Transformer
  stages carry real redundancy, which is what makes the combined result possible.


---

## 2 · Quantization Reconstruction

All three methods start from the same PTQ initialisation and the same calibration set; only the
objective that re-optimises weight rounding differs. Top-1 (%), FP32 reference **67.01**.

| Method | PTQ | Reconstructed | Deployed | TensorRT engine |
|---|---:|---:|---:|---:|
| QDrop | 65.64 | 66.40 | 66.42 | 66.46 |
| **APHQ-ViT** | 65.64 | 66.58 | 66.57 | **66.71** |
| LS-ViT | 65.64 | 66.60 | 66.52 | 66.51 |

Reconstruction recovers **+0.8 to +1.1 pp** over plain PTQ, and the gain survives export and the
TensorRT build rather than evaporating on the way to the engine.

---

## 3 · Deployment

Top-1 (%) · mean GPU compute time, batch 1, `1x3x72x72`.

| Method | RTX 4090 | RTX 4050 | Jetson AGX Orin |
|---|---|---|---|
| FP32 | 67.01 · 0.299 ms | 67.01 · 0.371 ms | 67.01 · 0.706 ms |
| FP16 | 67.03 · 0.204 ms | 67.00 · 0.299 ms | 66.99 · 0.398 ms |
| TensorRT implicit PTQ | 65.89 · 0.202 ms | 65.80 · 0.241 ms | 65.82 · 0.428 ms |
| NVIDIA ModelOpt (explicit QDQ) | 64.83 · 0.195 ms | 64.78 · 0.248 ms | 65.17 · 0.403 ms |
| QDrop | 66.46 · 0.195 ms | 66.54 · 0.238 ms | 66.22 · 0.405 ms |
| **APHQ-ViT** | **66.71 · 0.195 ms** | **66.69 · 0.244 ms** | **66.49 · 0.403 ms** |
| LS-ViT | 66.51 · 0.196 ms | 66.57 · 0.244 ms | 66.46 · 0.405 ms |

INT8 coverage: every explicit-QDQ build runs **88% of layers in INT8 (66/75)**; TensorRT's implicit
PTQ reaches only **55% (41/74)**.

**Results**

- **Reconstruction is accuracy for free.** ModelOpt and APHQ-ViT ship the same explicit-QDQ graph,
  the same 88% INT8 coverage and the same 0.195 ms — APHQ-ViT is **+1.88 pp Top-1** (+1.91 on
  RTX 4050, +1.32 on Orin). Only the weight rounding differs.
- **Coverage alone is not enough.** ModelOpt puts 88% of layers in INT8 and is still the *least*
  accurate INT8 engine, below implicit PTQ at 55% coverage.
- **The best INT8 engine lands within 0.30 pp of FP32** (0.52 pp on Orin) at **1.5–1.75× FP32
  speed**, and the method ranking is identical on all three platforms.
- Against FP16 the latency win is modest and platform-dependent: **+4%** on RTX 4090, **+18%** on
  RTX 4050, **−1%** on Orin — where INT8 buys coverage and model size rather than speed.
