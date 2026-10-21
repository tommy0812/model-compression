# Model Compression

Measured compression studies — structured pruning, knowledge distillation and INT8 post-training
quantization, taken all the way to TensorRT engines on RTX GPUs and NVIDIA Jetson AGX Orin.

| Study | Dataset | Highlights |
|---|---|---|
| [MobileViTv3 — Vision Transformer Compression](./MobileViTv3_Compression.md) | ImageNet-100 | **−54% params, +1.22 pp Top-1** over the XXS-KD baseline at the same MACs · PTQ reconstruction with residual-add quantization: **+4.5–4.7 pp over NVIDIA ModelOpt**, **1.7–2.3× FP32 speed** and faster than FP16 on every platform |

*By [Tommy Chen (陳龍懷)](https://linkedin.com/in/tommychen19920812)*
