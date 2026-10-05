# 图像 embedding 测评草案

## 第一阶段目标

本文仅针对 InnerMath 的图片 embedding 分支，研究数学变换后的向量对应什么图像语义。测评分成两层：

1. embedding 模型本身是否适合做图片语义空间。
2. 数学操作后的目标向量是否能被稳定、可解释地映射回图片。

第一版反向映射采用 nearest-neighbor retrieval：把候选图库编码成 embedding index，再把操作结果向量和图库向量做余弦相似度检索。

## 候选模型

### CLIP baseline

- 模型：`openai/clip-vit-base-patch32`
- 用途：快速 baseline、图文共同空间、zero-shot image classification、image-text similarity、image-to-image retrieval。
- 优点：生态成熟，Transformers 支持稳定，下载和 CPU demo 成本可接受。
- 局限：模型发布时间较早；模型卡明确提醒，真实任务需要做 task-specific testing；英文图文对齐更强。
- 来源：<https://huggingface.co/openai/clip-vit-base-patch32>

### SigLIP baseline

- 模型：`google/siglip-base-patch16-224`
- 用途：更现代的图文共同空间 baseline，可用于 zero-shot image classification 和 image-text retrieval。
- 优点：Hugging Face 模型卡说明 SigLIP 使用 sigmoid loss，不需要全局 pairwise normalization，便于扩展 batch size，并且可在较小 batch 下表现更好。
- 局限：作为第一版 demo，下载和兼容性验证应排在 CLIP baseline 之后。
- 来源：<https://huggingface.co/google/siglip-base-patch16-224>

## 推荐测评指标

### 基础检索质量

- Self retrieval：每张图片 embed 后，Top-1 是否能找回自己。
- Near duplicate retrieval：轻微裁剪、旋转、压缩、颜色扰动后，Top-K 是否能找回原图或同类图。
- Category retrieval：同一类别不同实例是否排在前面。
- Hard negative：外观相似但语义不同的图片是否被错误排前。

常用指标：

- Recall@K
- MRR
- nDCG@K
- Top-K similarity gap：`score_1 - score_2`，观察模型是否有清晰偏好。

### 数学操作测评

对每个操作记录：

- 输入图片 A、B。
- 操作名和参数，如 `slerp(t=0.5)`、`weighted-sum(alpha=1,beta=-0.25)`。
- 目标向量范数。
- Top-K 图片路径和相似度。
- 结果是否落回输入之一、同类图、混合语义图、无关图。

建议先测这些操作组：

- 插值组：`average`、`lerp`、`slerp`
- 代数组：`add`、`subtract`、`weighted-sum`、`hadamard`
- 几何组：`project-a-on-b`、`reject-a-from-b`、`reflect-a-across-b`、`cosine-bisector`

### 本项目特有的判断

普通检索模型测评只关心“搜得准不准”。本项目还要关心：

- 操作是否有可重复的视觉语义。
- 不同模型空间里，同一个数学操作是否产生相似趋势。
- 输出是否只是回到输入图片，还是能落到候选库中的“语义中间态”。
- 参数变化时，Top-K 是否沿着可解释轨迹变化。

## Demo 结果记录

使用环境：

- Python 3.11
- `openai/clip-vit-base-patch32`
- CPU PyTorch
- 8 张合成图片：红/蓝/紫/绿方形和圆形

命令：

```powershell
python scripts/create_demo_gallery.py

python -m innermath.cli `
  --image-a data\demo_gallery\red_square.png `
  --image-b data\demo_gallery\blue_square.png `
  --operation slerp `
  --t 0.5 `
  --gallery-dir data\demo_gallery `
  --gallery-index outputs\demo_gallery_clip_base.npz `
  --top-k 8 `
  --output-dir outputs\image-demo-slerp
```

结果：

| rank | image | cosine |
|---:|---|---:|
| 1 | `red_square.png` | 0.988896 |
| 2 | `blue_square.png` | 0.988896 |
| 3 | `green_square.png` | 0.963688 |
| 4 | `purple_square.png` | 0.955767 |
| 5 | `green_circle.png` | 0.907030 |
| 6 | `blue_circle.png` | 0.906669 |
| 7 | `purple_circle.png` | 0.900914 |
| 8 | `red_circle.png` | 0.803262 |

初步解释：

- 这次 `slerp(red_square, blue_square, t=0.5)` 没有优先返回 `purple_square`。
- 模型在这个合成图库中更明显地保持了“方形”结构，两个输入图片并列最高。
- 这个 demo 证明工程链路可跑，但不能作为模型质量结论。下一步需要自然图片和更大候选图库。

## 下一步测评集

建议建立一个 `data/eval_gallery`，每类 10 到 30 张图片：

- 颜色维度：红/蓝/紫/绿物体，但形状和类别尽量相同。
- 形状维度：圆形/方形/三角形/长条形，颜色尽量相同。
- 语义维度：猫/狗/车/房子/水果/家具。
- 风格维度：照片/插画/素描/低多边形。
- hard negatives：外观近似但语义不同，例如狼 vs 狗、柠檬 vs 网球。

然后对每个模型和每个操作生成统一 `results.json`，再汇总成表格。
