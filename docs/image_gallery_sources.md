# Image Gallery Sources

## 推荐路线

项目需要的是“可复现、可解释、许可清楚”的图片库，而不是单纯漂亮图片。建议按三层推进：

1. 合成图库：用脚本生成，完全可复现，适合验证数学操作和评测代码。
2. Wikimedia Commons 小图库：通过 API 下载，记录作者、许可证和来源页，适合第一批真实图片实验。
3. 大型公开数据集：Open Images、COCO、Unsplash Lite 等，适合后续扩展，但要单独处理许可、下载规模和标签质量。

## Wikimedia Commons

优点：

- 可以通过 API 获取图片、来源页、作者、许可证、原图尺寸等元数据。
- 可以过滤 CC BY、CC BY-SA、CC0、Public Domain 等许可。
- 适合下载 50 到 500 张小型真实图库。

脚本：

```powershell
python scripts/download_commons_gallery.py --per-query 5 --sleep 2
python scripts/run_image_eval.py configs/commons_eval.generated.yml
```

输出：

```text
data/commons_gallery/
  manifest.json
configs/commons_eval.generated.yml
```

注意：

- 图片不应提交进 Git。
- `manifest.json` 应保留，用来追踪来源、作者和许可证。
- 搜索结果不是人工精选，第一次下载后需要人工删掉低质、不相关或重复图片。
- 如果 Wikimedia 返回 429，降低 `--per-query`，增大 `--sleep`，并用 `--user-agent` 提供包含联系方式的 User-Agent。

## Open Images

优点：

- 标签体系强，适合 object/category retrieval。
- 官方数据集规模大，适合后续系统测评。
- 官方说明图片使用 Creative Commons Attribution 2.0 许可。

局限：

- 规模很大。
- 下载和筛选流程比 Commons 更重。
- 需要按类别抽样，避免一次下载过多。

适合阶段：多模型比较稳定后，用 Open Images validation/test 子集做更大规模评测。

## COCO

优点：

- 图像、类别、检测框、caption 生态成熟。
- 适合语义层面的 image-text / image-image 对照。

局限：

- 图片来源和许可较复杂；使用前要按 COCO terms 和原始图片许可核验。
- 更适合作为研究基准，不适合把图片重新分发到项目仓库。

## Unsplash Lite

优点：

- 图片质量高，审美和自然图像质量非常好。
- 适合做“高质量照片风格”的 embedding 行为观察。

局限：

- Unsplash Dataset 有自己的条款，不等同于普通开源图片集。
- 不建议把图片或下载后的镜像重新分发到仓库。
- 适合本地研究，不适合作为项目默认公开数据。

## 实操建议

第一批真实图库建议用 Wikimedia Commons：

- 每个 query 下载 5 到 10 张。
- 总量控制在 50 到 100 张。
- 人工检查并删掉明显跑偏图片。
- 记录 manifest。
- 用 `configs/commons_eval.generated.yml` 先跑 CLIP baseline。

后续如果需要更高质量照片，再加 Unsplash Lite；如果需要更强标签和规模，再加 Open Images。

## 当前 smoke run

已用 Wikimedia Commons 跑通一版小规模真实图库：

```powershell
python scripts/download_commons_gallery.py --per-query 2 --sleep 1.0
python scripts/run_image_eval.py configs/commons_eval.generated.yml
```

输出：

```text
data/commons_gallery/manifest.json
outputs/eval_runs/commons_clip_base/results.jsonl
outputs/eval_runs/commons_clip_base/summary.md
```

这版因为 Wikimedia 限流，实际下载到 10 张图，足够验证真实图链路，但不够做结论。正式评测建议每类至少 10 到 30 张，并人工清洗。
