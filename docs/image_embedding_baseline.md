# 图片 embedding 基线

这是 InnerMath 的图片 embedding 探索分支。该分支聚焦以下闭环：

1. 给定两张图片，或直接给定它们的 embedding 向量。
2. 在 embedding space 中执行数学向量操作。
3. 把得到的新向量映射回图片结果。

当前版本采用一个务实的 baseline：**nearest-neighbor retrieval 反向映射**。也就是说，我们先把候选图片库编码成 embedding 矩阵，然后把数学操作得到的目标向量和候选图片做余弦相似度检索，返回最接近的图片。它不是生成式 decoder，但适合作为第一版实验基线，因为稳定、可解释、无需训练，并且能很快比较不同数学操作的视觉效果。

## 当前支持的向量操作

- `add`: `a + b`
- `subtract`: `a - b`
- `average`: `(a + b) / 2`
- `weighted-sum`: `alpha * a + beta * b`
- `lerp`: 线性插值 `(1 - t) * a + t * b`
- `slerp`: 球面线性插值，适合 L2-normalized embedding
- `hadamard`: 分量乘法 `a * b`
- `divide`: 安全分量除法 `a / b`
- `project-a-on-b`: `a` 在 `b` 方向上的投影
- `reject-a-from-b`: `a` 去掉 `b` 方向后的正交分量
- `reflect-a-across-b`: `a` 关于 `b` 方向的反射
- `cosine-bisector`: 两个归一化向量的角平分方向

默认会对运算结果做 L2 归一化，方便和 CLIP/SigLIP/DINO 这类常见 embedding 的余弦检索配合。可以用 `--no-normalize-output` 关闭。

## 安装

只运行向量运算和测试：

```bash
python -m pip install -e .[dev]
```

需要从图片计算 embedding：

```bash
python -m pip install -e .[image,dev]
```

默认图片模型是 `openai/clip-vit-base-patch32`。它不是最终推荐模型，只是生态稳定、适合作为第一版 baseline。后续可以切换为 SigLIP、OpenCLIP、DINO 等模型。

## 使用：直接操作已有 embedding

假设已经有两个向量文件：

```bash
innermath-image `
  --embedding-a data/a.npy ^
  --embedding-b data/b.npy ^
  --operation slerp ^
  --t 0.35 ^
  --output-dir outputs/slerp-vector-only
```

这会生成：

- `outputs/slerp-vector-only/target.npy`
- `outputs/slerp-vector-only/results.json`

如果没有提供 gallery，只会保存目标向量。

## 使用：图片 embedding + 图片库近邻结果

```bash
innermath-image `
  --image-a data/query_a.jpg ^
  --image-b data/query_b.jpg ^
  --operation reflect-a-across-b ^
  --gallery-dir data/gallery ^
  --gallery-index outputs/gallery_clip_base.npz ^
  --top-k 8 ^
  --output-dir outputs/reflect-demo
```

第一次运行时会把 `data/gallery` 下的图片编码并缓存到 `outputs/gallery_clip_base.npz`。之后相同 gallery 可以复用这个 index。

输出包括：

- `target.npy`: 数学操作后的目标 embedding。
- `results.json`: Top-K 图片路径和相似度分数。
- `best.<ext>`: 最接近的图片副本，方便快速查看。

## 批量评测

单次 CLI 适合快速试一个操作；系统研究需要批量跑多个图片对、多个数学操作和多个参数。当前可以用：

```bash
python scripts/create_demo_gallery.py
python scripts/run_image_eval.py configs/demo_eval.yml
```

这会生成：

```text
outputs/eval_runs/demo_clip_base/
  config.resolved.json
  results.jsonl
  summary.md
  target_vectors/
```

更大的合成评测图库：

```bash
python scripts/create_synthetic_eval_gallery.py
python scripts/run_image_eval.py configs/synthetic_eval.yml
```

`configs/synthetic_eval.yml` 会生成并使用 60 张颜色、形状、风格组合图片，批量比较 `average`、`lerp`、`slerp`、`weighted-sum`、投影、正交分量、反射和角平分方向。

## 图片分支研究路线

第一阶段建议保持这个 retrieval baseline，同时系统比较不同数学操作：

- 插值类：`lerp`、`slerp`、`average`
- 代数类：`add`、`subtract`、`weighted-sum`、`hadamard`、`divide`
- 几何类：`project-a-on-b`、`reject-a-from-b`、`reflect-a-across-b`、`cosine-bisector`

这样能先回答一个具体问题：**不同数学变换在同一个 image embedding space 中会把语义推向哪里？**

第二阶段可以加入更强的“向量到图片”方式：

- 用更大的候选图片库做 retrieval decode。
- 用 Stable Diffusion image variation / unCLIP / IP-Adapter 等生成式方法，把目标 embedding 作为条件生成图片。
- 针对某类数学操作训练小型 operator 或 decoder。
- 对每种操作记录统计指标：Top-K 相似度、类别变化、PCA/UMAP 轨迹、人工可解释标签。

## 测试

```bash
python -m pytest
```

## GitHub remote

本地仓库已关联到：

```text
https://github.com/assass1nlike/InnerMath.git
```
