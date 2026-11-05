# 制作 SeisT / MagNet 的2.88秒数据集

在IDE直接运行 `build_dataset_288.py`，或从工作区执行：

```powershell
& "$env:USERPROFILE/.conda/envs/lightweight/python.exe" discussion/model_comparison/build_dataset_288.py
```

脚本顶部可修改输入路径、16进程设置、批大小和压缩方式。
输入为 `dataset/dataseth5/J_K_KIKnet_3000_Raw_100Hz.h5`；同名 `_log.csv` 是制作日志，不包含波形。

输出到 `discussion/model_comparison/data/SeisT_MagNet_288_Causal_<时间戳>/`：

- `SeisT_MagNet_288.h5`：每条记录 `(288,3)` float32，Z/N/E、gal、100Hz。
- `cohort.npz`：原有序记录键、mag/dis/azi 标签和固定划分。
- `manifest.json`：状态、输入身份、代码/清单/输出哈希、数量和处理说明。

记录集合严格复用 CompactGRU 的 `plotting/Fig01_Data_Distribution/labels.npz`：训练605166、验证84564、测试51772。不会只保留训练集；验证和测试保留用于之后的模型选择和公平比较。

保留P对齐起点后的前288点，复用 CompactGRU 的 `CausalWavePreprocessor`：每条记录重置状态，扣除首个采样点作为固定基线，使用100Hz下0.75–40Hz、Butterworth设计阶数4（带通总阶数8）的单向SOS滤波，float64状态、float32输出。不去线性趋势、不做全窗归一化、不用双向滤波。

执行方式为每条记录一次 `step` 输入全部288点，不按32点循环。滤波递推及初始条件与 CompactGRU 保持一致，因此与保留状态的9次32点处理具有相同的数学结果；它是整窗批处理执行方式，并非严格流式调度。后续训练加载器应避免重复滤波。本文件尚未接入上游原生训练加载器。
沿用原名和标签属性，并更新长度、时长、形状和split；保留P对齐偏移。2.88秒是现有P对齐窗口的名义时长，不消除原裁剪约0–0.009秒偏移。

只读取288点，之后的数据不参与当前窗口计算；原始制作脚本中200Hz抽点为100Hz的方式保持不变。
从原始H5读取，再按上述与 CompactGRU 相同的处理生成新H5，避免读取已滤波数据后重复滤波。没有E3WS数据输出。

每个读取进程独立打开只读源H5，任务队列有界，主进程独占写入；任何记录错误即停止，不静默丢样本、不补零、不覆盖旧数据。失败保留partial文件及错误manifest，完整写入后才改为正式文件名。磁盘需预留波形约2.6GB及HDF5逐记录属性/索引开销的额外空间。

提供 `check_build_dataset_288.py`，可自行运行，核对288点一次处理与9次32点处理的一致性、串行/多进程输出、标签和划分。此次修改仅编写代码，没有运行修改后的检查或真实数据制作。
