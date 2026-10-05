# Generative Decode

## 当前状态

目前 `scripts/run_image_eval.py` 的图片结果是 nearest-neighbor retrieval decode：

```text
target embedding -> gallery 中余弦最近的图片
```

这不是“从 embedding 直接生成新图片”。它的优点是稳定、可复现、便于测评；缺点是输出只能来自图库。

## 为什么不能直接把当前向量塞给生成模型

当前 baseline embedding 使用：

```text
openai/clip-vit-base-patch32
```

但生成式 decoder 通常要求它自己训练时使用的 embedding 空间。不同 CLIP/SigLIP/DINO 模型的维度、投影头和分布都不同。把 `clip-vit-base-patch32` 的数学运算结果直接塞给 Kandinsky、unCLIP 或 image-variation decoder，通常没有定义，也不会有可靠语义。

因此要生成有意义的新图，必须把“embedding backend”和“decoder”对齐。

## 第一条生成式路线：Kandinsky interpolation

Hugging Face Diffusers 文档说明，Kandinsky 可以在 image/text embeddings 之间做 interpolation，再把 embedding 解码成图像。这适合作为第一版“真正生成新图”的 baseline。

脚本：

```powershell
python scripts/generate_kandinsky_blend.py ^
  --image-a data\eval_gallery\green_square_solid.png ^
  --image-b data\eval_gallery\green_square_striped.png ^
  --weight-a 0.5 ^
  --weight-b 0.5 ^
  --output outputs\generated\green_square_blend.png
```

安装：

```powershell
python -m pip install -e .[generate]
```

注意：

- 当前机器的 PyTorch 是 CPU 版，Kandinsky 生成会非常慢。
- 如果要认真跑生成式 decode，建议安装 CUDA 版 PyTorch 并使用 GPU。
- 这个脚本是“weighted embedding blend -> generated image”的第一版，不是任意 `vector_ops.py` 操作的完整 decoder。

## 后续要做的真正版本

要支持所有数学操作并生成图像，需要一个 decoder-compatible embedding backend：

1. 用某个生成模型自己的 image encoder 得到 A、B 的 image embeddings。
2. 在这个 embedding space 中执行 `add`、`slerp`、`project`、`reflect` 等操作。
3. 把目标 embedding 传给同一个模型族的 decoder。
4. 同时保留 retrieval decode 作为 sanity check。

推荐顺序：

1. Kandinsky image/text embedding interpolation。
2. 扩展 Kandinsky backend，使它能暴露 image embeddings，然后接入 `vector_ops.py`。
3. 再看 Stable unCLIP / Stable Diffusion Image Variation。
4. 最后考虑训练自定义 decoder。
