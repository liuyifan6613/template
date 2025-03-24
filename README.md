# template
Template for own work.


# Diffusion
BasicDiff1: diffusion通用框架，非diffuser库实现，基于DocDiff改动实现，用于图片的训练和推理。   
BasicDiff2: ControlNet通用框架，基于DiffBIR-V2.1改动实现，保留超分与复原类框架，[底模](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt)为SD2.1版本。


# Network
UNet: U-Net模型的单卡训练以及推理代码。

# Utils
mgpus: 多卡训练与推理框架。
