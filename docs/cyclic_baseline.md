# 循环群探针：首轮基线

这是一组实际运行的负结果，目的是建立可复现的研究起点。没有据此声称发现了模型内部的群表示。

设置：模数 11、隐藏维度 48、2000 步 Adam、学习率 0.01、80% 任务训练输入、ridge=1e-6。运行环境：Python 3.13.9、NumPy 2.3.5。

| Seed | 训练准确率 | 任务留出准确率 | trained 等变误差 | untrained 等变误差 | shuffled 等变误差 | 精确对照等变误差 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 100.00% | 0.00% | 1.4281 | 1.1753 | 2.0603 | 1.27e-06 |
| 1 | 100.00% | 0.00% | 1.3350 | 1.4695 | 2.1803 | 1.42e-06 |
| 2 | 100.00% | 0.00% | 1.4903 | 1.7210 | 2.3283 | 1.83e-06 |

三次运行均记住了任务训练集，但未泛化到留出的模加法输入对。trained 的等变误差较大，群结构假设没有获得支持；精确对照的误差很小，说明探针在已知表示上可以工作。未训练网络的误差也必须一起观察，避免把输入编码带来的结构归因于学习。

这三个种子只是初始基线，不能代表所有网络、训练方法或非线性表示。下一步应研究任务泛化更好的模型、隐藏层分解和因果干预。

复现命令：

```powershell
innermath --seed 0 --output-dir outputs/cyclic/seed0
innermath --seed 1 --output-dir outputs/cyclic/seed1
innermath --seed 2 --output-dir outputs/cyclic/seed2
```

完整指标与划分：[seed 0](results/cyclic_seed0.json)、[seed 1](results/cyclic_seed1.json)、[seed 2](results/cyclic_seed2.json)。指标定义见[实验说明](cyclic_representations.md)。
