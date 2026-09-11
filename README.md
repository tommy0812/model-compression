## Tommy Chen (陳龍懷)

**Senior AI Algorithm Engineer — Model Compression & Edge Deployment**
Compal Electronics · M.S. Electrical and Computer Engineering, NYCU

I build vision models small enough and fast enough to run on embedded hardware, and I
measure every step of the way: structured pruning → knowledge distillation → post-training
quantization with block-wise reconstruction → ONNX Explicit QDQ → TensorRT INT8 →
NVIDIA Jetson AGX Orin.

---

### 📊 [Model Compression & Edge Deployment — Full Results](https://github.com/tommy0812/tommy0812/blob/main/Model_Compression_Results.md)

A complete, benchmarked compression pipeline on MobileViTv3 / ImageNet-100:

| | |
|---|---|
| **Pruning** | 1.91M → 0.61M params (**−68%**), 56.4M → 17.6M MACs (**−69%**) |
| **Accuracy recovery** | **62.74%** Top-1 — **+5.96 pp** over the XXS-KD baseline |
| **Quantization** | **62.30%** Top-1 INT8 — +0.56 pp vs. AdaRound, +0.26 pp vs. APHQ-ViT |
| **Jetson AGX Orin** | **62.40%** Top-1 (−0.34 pp vs. FP32), **0.864 ms** / **1,118 QPS** |
| **INT8 coverage** | **96.6%** of compute-bound layers (57/59) vs. 66.7% for NVIDIA ModelOpt |

**[→ Read the full experimental report](https://github.com/tommy0812/tommy0812/blob/main/Model_Compression_Results.md)** — per-method tables,
RTX 4090 vs. Jetson Orin engine comparison, the papers re-implemented, and how INT8 coverage
was actually measured.

---

### What I work on

- **Model compression** — structured pruning for CNNs (Group Fisher Pruning) and Vision
  Transformers (P3B block pruning), knowledge distillation, PTQ/QAT
- **Quantization** — Hessian-aware block-wise PTQ reconstruction (LS-ViT, APHQ-ViT,
  AdaRound), in-house INT8 fake-quantization framework with per-channel weights, BN folding,
  and automated calibration
- **Edge deployment** — PyTorch → ONNX Explicit QDQ → TensorRT 10.x, benchmarked on
  Jetson AGX Orin / AGX Xavier
- **Perception** — object detection, super-resolution-assisted small-object detection,
  monocular depth estimation, LiDAR / Radar / Camera sensor fusion on ROS

### Contact

[LinkedIn](https://linkedin.com/in/tommychen19920812) · a0912986300@gmail.com
