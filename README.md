# InnerMath

**Exploring Mathematical Structures in Model Representations**
**探索模型内部表示中的数学结构。**

InnerMath 关注模型内部是否存在非平凡的数学结构，以及这些结构与模型行为有什么关系。模型不限于 LLM；研究对象不限于 embedding 或 hidden-state 向量，也可以是注意力、线性算子、子空间及其相互关系。

我们希望把群表示、代数、拓扑和几何中的具体问题转化成可检验的实验：明确对象和假设，拟合或提取结构，检查定义要求的关系，比较对照，并记录失败案例。对向量进行数学操作可以作为探索工具，但本身不等于发现了模型内部的数学结构。

## 当前内容

| 分支 | 研究问题 | 状态 |
|---|---|---|
| 循环群表示 | 学习模加法的 MLP，其隐藏层能否用同一个线性算子表示输入的循环平移？ | 可训练、可复现的 NumPy 实验，包含留出轨道和对照 |
| 图片 embedding | 向量运算会如何改变图片检索结果？ | 保留原有 12 种运算、图片编码与批量评测 |
| 研究规划 | 如何将其他纯数学结构变为可证伪的研究问题？ | [研究路线](docs/project_roadmap.md)与[实验规范](docs/research_protocol.md) |

## 快速开始：循环群实验

Python >= 3.10。基础实验只需要 NumPy，无需 GPU、模型下载或 API key。

```powershell
python -m pip install -e ".[dev]"
innermath --modulus 11 --hidden 48 --steps 2000 --seed 0 --output-dir outputs/cyclic/seed0
```

也可以使用 `python -m innermath.cyclic`，参数相同。

实验学习任务为 `(a + b) mod n`。网络输入是两个 one-hot 向量，经过可训练的 tanh 隐藏层和分类层，**没有预先植入旋转或傅里叶表示**。训练结束后，对固定 `b` 的隐藏状态轨道拟合生成元 `T`，检查：

- 未参与拟合的 `b` 轨道上，`h(a,b) T ≈ h(a+1,b)` 是否成立。
- `T^k` 是否对应平移 `k`，`T^n` 是否在观测状态上接近恒等作用。
- 算子的特征值与 n 次单位根的距离，以及轨道在循环群特征标上的能量分布。
- 训练前网络、轨道内随机打乱顺序、已知精确群表示三种对照。

输出 `report.json`（完整指标、随机种子、划分、版本）、`artifacts.npz`（模型参数、激活和算子）、`summary.md`（简报）。

**这是检验结构的实验框架，不是“已经发现隐藏层形成群表示”的结论。** 模加法的输入对称性并不保证隐藏层的线性等变性；任务泛化和结构拟合泛化也不同。公式、指标和限制见[实验说明](docs/cyclic_representations.md)，实际运行结果见[基线报告](docs/cyclic_baseline.md)。

## 图片 embedding 分支

```powershell
python -m pip install -e ".[image,dev]"
innermath-image --embedding-a data/a.npy --embedding-b data/b.npy --operation slerp --t 0.35 --output-dir outputs/slerp
```

图片输入、图库检索、生成式解码探索与批量评测见[图片基线说明](docs/image_embedding_baseline.md)、[评测文档](docs/image_embedding_evaluation.md)和[生成式解码](docs/generative_decode.md)。图片模型所需权重在首次使用时下载。

## 项目结构

```text
innermath/
  structures.py       # 循环群生成元拟合、群关系检查、特征标能量
  cyclic.py           # NumPy MLP、实验划分、对照与结果导出
  vector_ops.py       # 原有图片/向量实验工具
  embedder.py, gallery.py, cli.py
docs/                 # 研究规范、数学定义、实验说明与结果
investigations/       # 调研及历史研究笔记
scripts/, configs/    # 图片实验脚本与配置
tests/                # 数学性质、训练梯度和实验流程测试
```

Python 包名和基础环境名为 `innermath`；研究 CLI 为 `innermath`，图片 CLI 为 `innermath-image`。早期版本中的 `math_embedding` 导入路径已统一为 `innermath`。

## 验证

```powershell
python -m pytest
```

仓库：[assass1nlike/InnerMath](https://github.com/assass1nlike/InnerMath)。项目研究范围以 [AGENTS.md](AGENTS.md) 为准。
