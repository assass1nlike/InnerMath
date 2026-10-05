# Hugging Face 上可获得的多模态 / 全模态 Embedding Space 调研

调研日期: 2026-05-07

## 1. 调研范围与判断标准

这份调研只看 `Hugging Face 上能直接拿到模型权重` 的 embedding space，不把“只有论文、没有可直接使用模型”混进来。

这里的“成熟”主要按下面几个维度综合判断：

- 是否有可直接使用的 HF 模型卡和权重
- 是否已有稳定生态支持，例如 `transformers`、`sentence-transformers`、`open_clip`
- 是否已经被广泛用于检索、RAG、跨模态相似度、zero-shot 等实际任务
- 是否有相对清晰的 benchmark 或模型卡结果
- 是否有明显的工程风险，例如只支持社区移植、许可证偏紧、模态并不完全对称

一个重要结论先写在前面：

- `image-text shared space` 这条线已经非常成熟
- `text/image/video/document` 这类更广义统一空间开始成熟，但生态还没有 CLIP 系那么稳
- 真正 `omni` 到 `audio/depth/thermal/IMU` 的“全模态统一空间”，目前在 HF 上更多还是研究成熟，不是工业成熟

## 2. 总体结论

如果只按“现在就能在 HF 上拿来干活”的标准排序：

1. `image-text` 方向最成熟，首选仍然是 `SigLIP2`、`Jina CLIP v2`、`Nomic Embed Vision/Text v1.5`、`OpenCLIP`
2. 更广义的统一空间里，`Ops-MM-embedding-v1` 是目前最值得认真看的开源候选
3. 音视频统一空间里，`PE-AV` 已经具备较强可用性，但它是专门做 `audio/video/text`，不是通用 omni-space
4. `LanguageBind` 和 `ImageBind` 依然是“全模态统一空间”的代表作，但在 HF 上更像研究资产，不是最顺手的工程选项

## 3. Image-Text Shared Space: 目前最成熟的一层

### 3.1 Google SigLIP2

- HF: <https://huggingface.co/google/siglip2-base-patch16-224>
- 论文: <https://arxiv.org/abs/2502.14786>
- 模态: `image + text`
- 许可证: `Apache-2.0`
- 生态: `transformers` 原生支持
- 成熟度判断: `很高`

优点：

- 这是目前 HF 上最稳的一类通用 `image-text` 共享空间之一
- 模型卡明确支持 `image-text retrieval`、`zero-shot image classification`
- 许可证友好，工程集成成本低
- HF 模型卡显示下载量高，说明生态接受度很强

限制：

- 本质上仍然是 `image-text`，不是更广义的多模态统一空间
- 如果任务是 `video/doc/audio`，就不该把它当成“通用多模态空间”

适用场景：

- 图文检索
- 多语言图文对齐
- 视觉语义召回
- 作为 VLM 的视觉编码器

### 3.2 Jina CLIP v2

- HF: <https://huggingface.co/jinaai/jina-clip-v2>
- 论文: <https://arxiv.org/abs/2412.08802>
- 模态: `image + text`
- 许可证: `CC BY-NC 4.0`
- 生态: `transformers`、`sentence-transformers`、`Transformers.js`、`ONNX`
- 成熟度判断: `很高`

优点：

- 明确支持 `94 languages`
- 同时兼顾图像和文本检索，工程感很强
- 支持 `Matryoshka` 截断，方便做向量降维和存储压缩
- 对多语言检索和跨语种图文搜索很有吸引力

限制：

- 许可证是 `non-commercial`
- 如果你的场景是可商用部署，需要先确认许可证边界

适用场景：

- 多语言图文检索
- 国际化商品搜索
- 多语言视觉搜索
- 需要 `sentence-transformers` 生态的多模态检索

### 3.3 Nomic Embed Vision/Text v1.5

- HF: <https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5>
- 论文: <https://arxiv.org/abs/2406.18587>
- 模态: `image + text`
- 许可证: `Apache-2.0`
- 生态: `transformers`，文本侧可与 `nomic-embed-text-v1.5` 直接共用空间
- 成熟度判断: `高`

优点：

- 图像 embedding 明确和 `nomic-embed-text-v1.5` 共享同一个空间
- 很适合把图像并入现有文本 RAG 或文本检索系统
- 模型较小，工程成本低
- 许可证友好

限制：

- 更像“把图像对齐到一个强文本空间”，不是广义 omni-space
- 文本检索时需要遵循前缀约定，例如 `search_query:`

适用场景：

- 多模态 RAG
- 图像并入文本向量库
- 轻量级图文检索

### 3.4 OpenCLIP / LAION CLIP

- HF: <https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K>
- 项目: <https://github.com/mlfoundations/open_clip>
- 模态: `image + text`
- 许可证: `MIT`
- 生态: `OpenCLIP`
- 成熟度判断: `很高`

优点：

- 这是最经典、最 battle-tested 的开源图文共享空间之一
- 上下游兼容性最好，很多检索和多模态方案都会默认把它作为基线
- 适合做对照实验和快速搭系统

限制：

- 语种上更偏英文
- 效果未必是今天最强的一档
- 原模型卡也明确提醒它更偏研究输出，不应在缺少充分测试时直接无约束部署

适用场景：

- 基线模型
- 快速原型
- 老系统兼容
- 需要 OpenCLIP 生态的项目

### 3.5 Image-Text 层的推荐顺序

如果只看今天在 HF 上的可用性和工程成熟度，我会这样排：

1. `SigLIP2`
2. `Jina CLIP v2`
3. `Nomic Embed Vision/Text v1.5`
4. `OpenCLIP`

简化理解：

- 要最稳、最通用: `SigLIP2`
- 要多语言检索: `Jina CLIP v2`
- 要并入已有文本向量体系: `Nomic`
- 要老牌基线和最高兼容性: `OpenCLIP`

## 4. 其它多模态 Embedding Space

这里把“不是全模态 omni-space，但确实已经跨越两种以上模态、并能直接在 HF 上用”的模型单独列出来。

### 4.1 Ops-MM-embedding-v1

- HF: <https://huggingface.co/OpenSearch-AI/Ops-MM-embedding-v1-2B>
- 模态: `text + image + text-image pair + visual document + video`
- 许可证: `Apache-2.0`
- 基座: `Qwen2-VL` 微调
- 成熟度判断: `中高`

优点：

- 这是目前 HF 上最像“广义统一空间”的开源模型之一
- 它不是只做图文，而是把 `text / image / text-image pair / visual document / video` 编到同一个空间
- 模型卡直接给了 `MMEB-V2`、`MMEB-Image`、`ViDoRe-v2` 结果
- 如果你的核心任务是 `图文检索 + 视觉文档检索 + 视频检索`，它比单纯 CLIP 更接近真实需求

限制：

- 新模型，生态成熟度还不如 CLIP 系
- 更像“VLM 派生的 embedding 模型”，不是经典对称式双塔
- 下载量和社区采用度还没有拉开数量级优势

适用场景：

- 多模态 RAG
- 文档页检索
- 视频检索
- 图文混合索引

结论：

- 如果你要的不是“最成熟 image-text”，而是“更接近 universal embedding space 的可用开源模型”，`Ops-MM-embedding-v1` 是当前最值得优先试的一个

### 4.2 PE-AV

- HF: <https://huggingface.co/facebook/pe-av-base>
- 论文: <https://arxiv.org/abs/2512.19687>
- 模态: `audio + video + audio-video + text`
- 许可证: `Apache-2.0`
- 生态: `transformers` 直接支持
- 成熟度判断: `中高`

优点：

- 来自 Meta，模型卡质量高，接口清晰
- 能同时编码 `audio`、`video`、`audio-video` 和 `text`
- 对音视频检索、跨模态匹配非常实用
- 提供多个尺寸和帧数版本

限制：

- 它不是图像、文档、深度、热成像那种通用 omni-space
- 文本侧其实是针对不同配对关系提供不同 embedding，严格说并不是一个完全对称的“单一文本向量”体系

适用场景：

- 音视频检索
- audio-text 检索
- video-text 检索
- audio-video-text 联合相似度

结论：

- 如果任务落在 `audio/video/text` 这一簇，`PE-AV` 比拿通用 CLIP 硬改更靠谱

### 4.3 CLAP

- HF: <https://huggingface.co/laion/clap-htsat-unfused>
- 论文: <https://arxiv.org/abs/2211.06687>
- 模态: `audio + text`
- 许可证: `Apache-2.0`
- 生态: `transformers` 直接支持
- 成熟度判断: `很高`

优点：

- 这是音频版的 CLIP，已经相当成熟
- HF 下载量很高，音频检索和 zero-shot 音频分类生态稳定
- 如果你的任务只涉及 `audio-text`，它往往比更大的通用模型更直接

限制：

- 不是通用多模态空间
- 只适合 `audio-text`

适用场景：

- text-to-audio retrieval
- audio-text similarity
- zero-shot 音频分类

结论：

- `CLAP` 不属于“全模态统一空间”，但在专业音频语义空间里，它已经是非常成熟的 HF 资产

## 5. 全模态 / Omni-Space: 研究代表作与 HF 可用性

### 5.1 LanguageBind

- HF 入口:
  - <https://huggingface.co/LanguageBind/LanguageBind_Video_FT>
  - <https://huggingface.co/LanguageBind/LanguageBind_Audio_FT>
  - <https://huggingface.co/LanguageBind/LanguageBind_Image>
  - <https://huggingface.co/LanguageBind/LanguageBind_Thermal>
  - <https://huggingface.co/LanguageBind/LanguageBind_Depth>
- 论文: <https://arxiv.org/abs/2310.01852>
- 模态: `video + audio + depth + thermal + image + language`
- 许可证: `MIT`
- 成熟度判断: `研究成熟度高，HF 工程成熟度中等`

优点：

- 这是目前 HF 上最完整的 `language-centered N-modality` 家族之一
- 核心思想很清晰: 用 `language` 做 bind，把多个模态对齐到共享语义空间
- 模型卡直接给了多模态联合推理示例
- 对 `video/audio/depth/thermal` 这类非标准模态很友好

限制：

- 它更像“多分支模型家族”，不是一个像 CLIP 那样极顺手的一体化产品
- 工程体验、社区封装、下游生态都明显不如 CLIP 系
- 模型卡明确说明 `image` 分支本身并未单独训练，而是从 OpenCLIP 初始化，这说明它的“全模态统一”仍然带有拼接式色彩

适用场景：

- 想做全模态研究
- 想看 `language-as-pivot` 的统一空间范式
- 任务涉及 `depth / thermal / audio / video`

结论：

- 如果你的目标是“研究 omni embedding”，LanguageBind 很值得看
- 如果你的目标是“今天在产品里稳定上线”，它还不是最省事的选择

### 5.2 ImageBind

- HF 社区移植: <https://huggingface.co/handaber/imagebind>
- 官方论文: <https://arxiv.org/abs/2305.05665>
- Meta 项目页: <https://ai.meta.com/research/publications/imagebind-one-embedding-space-to-bind-them-all/>
- 模态: `image + text + audio + depth + thermal + IMU`
- 许可证: `CC BY-NC 4.0`
- 成熟度判断: `论文成熟度高，HF 工程成熟度偏低`

优点：

- 这是“统一嵌入空间”路线最经典的代表作之一
- 它在概念上非常接近大家直觉中的 omni-space
- 模态覆盖广，包含 `depth / thermal / IMU`

限制：

- HF 上目前更常见的是社区移植，不是官方一等公民式维护
- 许可证偏紧
- 工程生态、维护和兼容性都不如现在主流的 HF 检索模型

适用场景：

- 研究型实验
- 演示 unified embedding 的论文思路
- 小规模 cross-modal 验证

结论：

- `ImageBind` 仍然是最重要的 omni-space 代表作之一
- 但如果问题是“HF 上最成熟、最好上手的模型”，它不在第一梯队

## 6. 选型建议

### 6.1 如果目标是今天就上线

- 纯图文检索: `SigLIP2`
- 多语言图文检索: `Jina CLIP v2`
- 想把图像并入文本向量库: `Nomic Embed Vision/Text v1.5`
- 需要老牌强基线: `OpenCLIP`

### 6.2 如果目标是更广义多模态 RAG

- 首先试 `Ops-MM-embedding-v1`
- 如果任务里有大量视觉文档页、图像、视频混合内容，它比 CLIP 系更贴题

### 6.3 如果目标是音视频方向

- `audio-text` 为主: `CLAP`
- `audio/video/text` 联合空间: `PE-AV`

### 6.4 如果目标是真正的 omni-space 研究

- 首先看 `LanguageBind`
- 同时参考 `ImageBind`
- 这两者更像“研究范式的代表”，而不是“HF 上最省事的生产模型”

## 7. 最终判断

截至 2026-05-07，HF 上“最成熟”的多模态 embedding space 不能一概而论，应该分层看：

- 最成熟的仍然是 `image-text` shared space
- 最值得部署的图文模型是 `SigLIP2 / Jina CLIP v2 / Nomic / OpenCLIP`
- 最接近“广义统一多模态空间”的可用模型，是 `Ops-MM-embedding-v1`
- 最强的音视频统一空间候选，是 `PE-AV`
- 最典型的全模态研究路线，是 `LanguageBind` 和 `ImageBind`

一句话总结：

> 如果你要“现在就能稳定用”，优先看 `SigLIP2`、`Jina CLIP v2`、`Nomic`、`Ops-MM`。
> 如果你要“真正全模态统一空间”的研究代表，优先看 `LanguageBind` 和 `ImageBind`。

## 8. 参考链接

- SigLIP2: <https://huggingface.co/google/siglip2-base-patch16-224>
- Jina CLIP v2: <https://huggingface.co/jinaai/jina-clip-v2>
- Nomic Embed Vision v1.5: <https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5>
- OpenCLIP LAION: <https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K>
- Ops-MM-embedding-v1-2B: <https://huggingface.co/OpenSearch-AI/Ops-MM-embedding-v1-2B>
- PE-AV: <https://huggingface.co/facebook/pe-av-base>
- CLAP: <https://huggingface.co/laion/clap-htsat-unfused>
- LanguageBind: <https://huggingface.co/LanguageBind/LanguageBind_Video_V1.5_FT>
- ImageBind 社区移植: <https://huggingface.co/handaber/imagebind>
- ImageBind 论文: <https://arxiv.org/abs/2305.05665>
- LanguageBind 论文: <https://arxiv.org/abs/2310.01852>
