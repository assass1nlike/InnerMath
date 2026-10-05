# 图片 Embedding Space 调研

调研时间：2026-05-07

本文关注当前可在 Hugging Face 等平台获取权重、工程上较成熟的图片 embedding 模型。这里的“图片 embedding space”需要先拆成几类：

- 图文共同向量空间：适合 text-to-image retrieval、image-to-image retrieval、zero-shot 分类、图文匹配。
- 纯视觉表示空间：适合图像相似度、聚类、去重、异常检测、下游视觉任务。
- 文档图片检索空间：适合 PDF 页面、扫描件、截图、表格、图表等页面级检索，通常使用多向量 late-interaction。

## 快速结论

| 场景 | 首选 | 备选 |
|---|---|---|
| 通用图文检索、零样本分类、以文本搜图片 | `google/siglip-so400m-patch14-384` | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`, `openai/clip-vit-large-patch14` |
| 只做图像相似度、聚类、视觉特征、下游视觉任务 | `facebook/dinov3-vitb16-pretrain-lvd1689m` 或 `facebook/dinov3-vitl16-pretrain-lvd1689m` | `facebook/dinov2-with-registers-base`, `facebook/dinov2-with-registers-large`, `facebook/dinov2-large` |
| 中文/多语言文本搜图片 | `jinaai/jina-clip-v2` | `google/siglip-base-patch16-256-multilingual`, `visheratin/nllb-siglip-mrl-large` |
| 图片和文本统一 embedding，用于 RAG/搜索产品 | `nomic-ai/nomic-embed-vision-v1.5` | `jinaai/jina-clip-v2` |
| PDF 页面、截图、表格/图表/版面检索 | `vidore/colqwen2-v1.0-hf` / ColQwen 系列 | `vidore/colpali-v1.3` |

默认推荐组合：

- 图文检索 baseline：`google/siglip-so400m-patch14-384`
- 图文检索对照：`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
- 纯视觉特征：`facebook/dinov3-vitb16-pretrain-lvd1689m`
- 中文/多语言：`jinaai/jina-clip-v2`
- 文档图片 RAG：`vidore/colqwen2-v1.0-hf`

## 1. CLIP / OpenCLIP

CLIP / OpenCLIP 是图文共同向量空间里生态最成熟的一支。它不是当前所有指标上的最强方案，但因为教程、工具、向量库集成、评测基线和线上实践经验很多，仍然非常适合作为第一条 baseline。

代表模型：

- `openai/clip-vit-large-patch14`
- `openai/clip-vit-base-patch32`
- `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
- `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`

适合：

- 文本搜图片。
- 图片搜图片。
- zero-shot 分类。
- 需要一个可比较、可复现、生态成熟的 baseline。

优势：

- 工程生态成熟。
- 使用方式简单。
- 向量数据库、检索 demo、评测代码多。
- OpenCLIP / LAION 权重选择丰富。

局限：

- 原始 OpenAI CLIP 年代较早，英文为主。
- 对细粒度分类、复杂关系、计数、局部区域理解有限。
- OpenCLIP 大模型推理成本较高。

备注：

- `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` 的模型卡显示它基于 OpenCLIP，在 LAION-2B 英文子集上训练，license 为 MIT，并报告 ImageNet-1k zero-shot top-1 约 78.0。
- `openai/clip-vit-large-patch14` 是最常见的稳定 baseline，但如果是新项目，通常建议同时测试 SigLIP 或 OpenCLIP ViT-H。

来源：

- https://huggingface.co/openai/clip-vit-large-patch14
- https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K
- https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k

## 2. SigLIP

SigLIP 是当前通用图文 embedding 的强实用选择。它和 CLIP 一样使用图像编码器与文本编码器生成共同空间表示，但训练目标从 CLIP 的 softmax 对比损失改为 sigmoid loss。Transformers 文档说明，这种目标让训练更容易扩展到大 batch，并且在较小 batch 下也有更好表现。

代表模型：

- `google/siglip-so400m-patch14-384`
- `google/siglip-base-patch16-224`
- `google/siglip-base-patch16-256`
- `google/siglip-base-patch16-256-multilingual`

适合：

- 通用图文检索。
- zero-shot 图像分类。
- caption/image 匹配。
- 新项目中的主力图文 embedding 候选。

优势：

- 比原始 CLIP 更新，效果通常更强。
- Transformers 原生支持，工程接入简单。
- `google/siglip-so400m-patch14-384` 是一个质量和可用性都较好的强基线。
- 有 multilingual 版本可用于多语言文本查询。

局限：

- 与 CLIP 相比，生态历史略短。
- 多语言能力需要具体评测，不能直接假设中文业务效果一定足够好。

备注：

- `google/siglip-so400m-patch14-384` 的模型卡显示它使用 384x384 输入、SoViT-400m 架构，license 为 Apache-2.0，可用于 zero-shot image classification 和 image-text retrieval。

来源：

- https://huggingface.co/docs/transformers/en/model_doc/siglip
- https://huggingface.co/google/siglip-so400m-patch14-384
- https://huggingface.co/google/siglip-base-patch16-256-multilingual

## 3. DINOv2 / DINOv3

DINO 系列更适合“图像模态本身”的 embedding，而不是图文对齐。如果任务不需要文本查询，只需要图片之间的相似度、聚类、去重、异常检测、局部视觉特征或下游视觉任务，DINO 往往比 CLIP/SigLIP 更合适。

### DINOv2

代表模型：

- `facebook/dinov2-base`
- `facebook/dinov2-large`
- `facebook/dinov2-giant`
- `facebook/dinov2-with-registers-base`
- `facebook/dinov2-with-registers-large`
- `facebook/dinov2-with-registers-giant`

适合：

- image-to-image 相似度。
- 图像聚类。
- 去重。
- frozen feature extractor。
- 视觉下游任务，如分类、分割、深度估计等。
- patch-level local embeddings。

优势：

- 自监督视觉 foundation model，视觉特征质量稳定。
- 可提取整图 CLS embedding，也可提取 patch-level features。
- with-registers 版本改善了 ViT attention artifact，attention map 更干净。

来源：

- https://huggingface.co/docs/transformers/en/model_doc/dinov2
- https://huggingface.co/docs/transformers/main/model_doc/dinov2_with_registers
- https://huggingface.co/facebook/dinov2-with-registers-large

### DINOv3

DINOv3 是更新一代的视觉 foundation model。Transformers 文档显示 DINOv3 于 2025-08-13 发布，2025-08-14 加入 Transformers，强调高质量 dense features，并面向多种视觉任务。

代表模型：

- `facebook/dinov3-vits16-pretrain-lvd1689m`
- `facebook/dinov3-vitb16-pretrain-lvd1689m`
- `facebook/dinov3-vitl16-pretrain-lvd1689m`
- `facebook/dinov3-vith16plus-pretrain-lvd1689m`
- `facebook/dinov3-vit7b16-pretrain-lvd1689m`

适合：

- 更强的纯视觉 embedding。
- dense feature extraction。
- image retrieval。
- classification、segmentation、depth、object discovery 等视觉任务。

优势：

- 相比 DINOv2 更新，能力更强。
- 模型卡说明可输出 class token、patch tokens、register tokens。
- 对局部特征和 dense prediction 更友好。

注意：

- DINOv3 使用 `dinov3-license`，不是 Apache/MIT，需要单独检查商业使用条款。
- `facebook/dinov3-vit7b16-pretrain-lvd1689m` 等大模型可能是 gated，需要同意条件后才能访问。

来源：

- https://huggingface.co/docs/transformers/model_doc/dinov3
- https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
- https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m/tree/main

## 4. Jina CLIP v2

`jinaai/jina-clip-v2` 是非常实用的多语言多模态 embedding 模型，适合中文、英文和多语言场景中的图文检索。

适合：

- 中文文本搜图片。
- 多语言文本搜图片。
- 跨语言图文检索。
- 图文混合检索。

优势：

- 模型卡显示支持 89 种语言。
- 图片输入为 512x512。
- 输出 64 到 1024 维 Matryoshka 表示，便于质量和成本折中。
- 图像和文本在共同空间对齐。
- 有 `sentence-transformers`、Transformers.js、ONNX 示例。

风险：

- HF 模型卡显示 license 为 `cc-by-nc-4.0`。如果需要商业自托管，不能直接按开源商用处理，需要走 Jina API 或联系授权。

来源：

- https://huggingface.co/jinaai/jina-clip-v2

## 5. Nomic Embed Vision

`nomic-ai/nomic-embed-vision-v1.5` 是偏产品化搜索/RAG 的统一图文 embedding 模型。它和 `nomic-embed-text-v1.5` 共享同一个 embedding space，可以 embed 图片，也可以用文本向量检索图片。

适合：

- 图片/文本统一检索。
- 多模态 RAG。
- 资产库、知识库、搜索产品。
- 需要 Apache-2.0 license 的工程场景。

优势：

- HF 模型卡显示 license 为 Apache-2.0。
- 和 Nomic 文本 embedding 对齐，便于统一索引。
- 定位贴近语义搜索产品，而不只是研究模型。

局限：

- 生态和可比评测不如 CLIP/SigLIP/DINO 普遍。
- 具体业务效果需要和 SigLIP、Jina CLIP v2 做 A/B。

来源：

- https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5
- https://www.nomic.ai/news/nomic-embed-vision

## 6. EVA-CLIP

EVA-CLIP 是开源 CLIP 家族里规模很大的强模型。`BAAI/EVA-CLIP-18B` 模型卡显示它有 18B 参数，license 为 Apache-2.0，并报告在 27 个图像分类 benchmark 上 zero-shot top-1 平均 80.7%。

适合：

- 离线高质量 embedding。
- benchmark。
- 大算力环境中的图文检索实验。
- 对质量要求高、对推理成本不敏感的场景。

不太适合：

- 多数线上实时检索服务。
- 预算受限的系统。
- 需要快速迭代的小型项目。

来源：

- https://huggingface.co/BAAI/EVA-CLIP-18B

## 7. ColPali / ColQwen

如果“图片”其实是 PDF 页面、扫描件、截图、报告、表格或图表，那么普通 CLIP/SigLIP/DINO 往往不是最优。这个场景更推荐 visual document retrieval 模型，尤其是 ColPali / ColQwen 系列。

这类模型通常把文档页当作图片，不依赖传统 OCR/layout pipeline，用 VLM 生成多向量 embeddings，并通过 ColBERT-style late interaction 计算查询与页面的相似度。

代表模型：

- `vidore/colqwen2-v1.0-hf`
- `vidore/colqwen2.5-v0.2`
- `vidore/colpali-v1.3`
- `vidore/colSmol-256M`
- `vidore/colSmol-500M`

适合：

- PDF 页面检索。
- 扫描件检索。
- 截图检索。
- 表格、图表、版面密集文档检索。
- 文档图片 RAG。

优势：

- 比普通单向量图片 embedding 更适合文档页面。
- 能利用页面版面、视觉结构、文字块、表格和图表信息。
- 多向量 late-interaction 通常有更好的召回质量。

代价：

- 索引和查询成本更高。
- 向量数量更多，系统复杂度高于单向量 embedding。
- 需要配合专门的检索实现，而不只是简单余弦相似度。

来源：

- https://huggingface.co/docs/transformers/model_doc/colpali
- https://huggingface.co/docs/transformers/model_doc/colqwen2
- https://huggingface.co/vidore

## 工程选型建议

### 通用图片库，支持文本搜图

优先测试：

1. `google/siglip-so400m-patch14-384`
2. `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
3. `openai/clip-vit-large-patch14`

理由：

- SigLIP 作为主力候选。
- OpenCLIP ViT-H 作为强 baseline。
- OpenAI CLIP 作为生态最成熟的传统 baseline。

### 中文或多语言文本搜图片

优先测试：

1. `jinaai/jina-clip-v2`
2. `google/siglip-base-patch16-256-multilingual`
3. `visheratin/nllb-siglip-mrl-large`

注意：

- Jina CLIP v2 的 license 是 `cc-by-nc-4.0`，商业自托管需要额外处理授权。
- 多语言效果一定要用业务查询集评测，不能只看英文 benchmark。

### 只做 image-to-image

优先测试：

1. `facebook/dinov3-vitb16-pretrain-lvd1689m`
2. `facebook/dinov3-vitl16-pretrain-lvd1689m`
3. `facebook/dinov2-with-registers-large`
4. `facebook/dinov2-large`

理由：

- 这类任务不需要图文对齐，DINO 的纯视觉特征更自然。
- 如果需要局部视觉结构或 dense feature，DINO 系列通常比 CLIP/SigLIP 更合适。

### 文档图片 RAG

优先测试：

1. `vidore/colqwen2-v1.0-hf`
2. `vidore/colqwen2.5-v0.2`
3. `vidore/colpali-v1.3`

理由：

- 文档页不是普通自然图片。
- 版面、表格、图表、文字块对检索质量影响很大。
- ColPali/ColQwen 的多向量 late-interaction 更贴近文档检索问题。

### 预算很紧

候选：

- `openai/clip-vit-base-patch32`
- `google/siglip-base-patch16-224`
- `google/siglip-base-patch16-256`
- `facebook/dinov2-base`
- `facebook/dinov3-vits16-pretrain-lvd1689m`

### 预算充足，追求质量

候选：

- `google/siglip-so400m-patch14-384`
- `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
- `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`
- `facebook/dinov3-vitl16-pretrain-lvd1689m`
- `facebook/dinov3-vith16plus-pretrain-lvd1689m`
- `BAAI/EVA-CLIP-18B`

## 实验建议

建议不要只看公开 benchmark。更稳妥的方式是构造一个小型业务评测集：

- 100 到 500 条真实文本查询。
- 每条查询标注 3 到 10 张相关图片。
- 加入中英文查询、短查询、长查询、口语化查询、同义词查询。
- 对 image-to-image 场景，准备近重复、同类不同实例、外观相似但语义不同的 hard negative。
- 对文档页面，单独准备表格、图表、截图、扫描件、文字密集页。

常用指标：

- Recall@K
- nDCG@K
- MRR
- 人工抽检 Top-K
- 推理延迟
- 索引体积
- GPU/CPU 成本

一个实用的第一轮实验矩阵：

| 任务 | 模型 |
|---|---|
| 文本搜自然图片 | SigLIP So400M, OpenCLIP ViT-H, OpenAI CLIP L/14 |
| 中文文本搜图片 | Jina CLIP v2, multilingual SigLIP |
| 图片搜图片 | DINOv3 ViT-B, DINOv2 with registers large, SigLIP So400M |
| 文档页检索 | ColQwen2, ColPali |

## 最终判断

- 如果要最稳的通用图文 embedding：先试 `google/siglip-so400m-patch14-384`。
- 如果要最成熟可比的 baseline：保留 `openai/clip-vit-large-patch14` 和 OpenCLIP。
- 如果要图像模态本身的特征空间：优先 DINOv3，其次 DINOv2 with registers。
- 如果要中文/多语言图文检索：优先 Jina CLIP v2，但要注意 license。
- 如果要文档图片检索：直接看 ColQwen/ColPali，不要默认用普通 CLIP/SigLIP。
