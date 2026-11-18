# Model Compression

Measured compression studies — structured pruning, knowledge distillation and post-training
quantization, taken all the way to TensorRT engines on RTX 4090 and NVIDIA Jetson AGX Orin, and to a
Snapdragon NPU through Qualcomm AI Hub.

| Study | Dataset | Highlights |
|---|---|---|
| [MobileViTv3 — Vision Transformer Compression](./MobileViTv3_Compression.md) | ImageNet-100 | **−54% params, +1.22 pp Top-1** over the XXS-KD baseline at the same MACs · PTQ reconstruction: **+4.5–4.8 pp over NVIDIA ModelOpt** on TensorRT, faster than FP16 on RTX 4090 and Jetson AGX Orin · **+1.4–1.5 pp over Qualcomm AI Hub's own w8a8 quantizer** on a Snapdragon NPU, **38% lower latency than FP16** |

*By [Tommy Chen (陳龍懷)](https://linkedin.com/in/tommychen19920812)*
