# 讨论实验：ELS-EEW规模—精度—计算资源权衡

本目录是独立讨论实验，复用 `els_eew` 的 **CompactGRU** 实现及原始训练逻辑。仅改变CNN宽度和GRU隐藏维度，不采用旧S4架构，不覆盖原模型、日志或数据。

## Windows一键入口

1. **先双击 `check_scaling_experiment.bat`**：检查依赖、H5路径、参考权重、五个模型的实际2976点前向/流式一致性、参数量、MACs、缓存，以及少量合成数据恢复测试。不会读取完整数据集、不会对真实数据训练、不会创建正式实验完成状态。
2. **准备好后双击 `run_scaling_experiment.bat`**：执行完整实验，可能持续数十小时。默认调用 `%USERPROFILE%\.conda\envs\lightweight\python.exe`，可通过环境变量 `SCALING_PYTHON` 指定其他Python。两个批处理文件结束或报错均会保留窗口。
3. 再次双击正式入口，会恢复 `active_run.json` 指向的实验，跳过已经完整训练和测试且文件校验通过的模型。

IDE或命令行同样可运行：

```powershell
& "$env:USERPROFILE\.conda\envs\lightweight\python.exe" discussion/scaling_study/run_scaling_experiment.py --check
& "$env:USERPROFILE\.conda\envs\lightweight\python.exe" discussion/scaling_study/run_scaling_experiment.py
```

需要新建完整实验而非续跑时，添加 `--new-run`。每个正式实验独立保存在 `results/YYYYMMDD_HHMMSS_microseconds_seed42/`；预检仅写入 `checks/`。路径均以整个 `Light_model` 工作区为根解析，不依赖盘符或shell当前目录。

## 五种规模（预检已核实）

|配置|四个残差块CNN宽度|两层GRU隐藏维度|CNN参数|GRU参数|输出层参数|任务权重参数|总参数|MACs/32点更新|状态KiB|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|ELS-2K|6|12|336|1656|52|3|2047|55920|0.796875|
|ELS-4K|12|16|960|3072|68|3|4103|116416|1.531250|
|ELS-8K|16|24|1536|6624|100|3|8263|243296|2.062500|
|ELS-16K|22|34|2640|13056|140|3|15839|477384|2.843750|
|ELS-32K|34|48|5712|26208|196|3|32119|984704|4.359375|

上表总参数包含3个训练用任务权重，与原检查点的8263口径一致；纯推理权重另外保存 `inference_weights_kib`。四个CNN块均为核宽3的depthwise因果卷积→逐时刻通道LayerNorm→GELU→pointwise→残差；膨胀率1、2、4、8。GRU为两层，层间Dropout=0.1，逐采样点递推，每32点读取一次四维输出。

配置搜索枚举偶数CNN宽度4–96和GRU隐藏维度4–128，保留参数偏差≤8%、宽度比例对数偏差≤0.3的组合，按 `参数相对偏差 + 0.15×比例对数偏差` 排序。只使用参数公式，不接触验证/测试误差。8K固定为参考结构16/24。各目标前8个候选及最终选择保存为CSV/JSON。

## 参考权重和公平性

默认只读加载：

`els_eew/outputs/experiment_20261012_223631_490021/CompactGRU_seed42/checkpoint_CompactGRU_20261012_235806/best_CompactGRU_20261012_235806.pth`

SHA256为 `c20841d6c4d76776dae64b83847cb5e788d287f59f52adab592f183639557f80`。参考模型最佳轮次258，训练完成278轮。预检严格加载全部权重，核对保存的运行配置、AdamW参数组、结构和数据来源。完整运行会重新计算8K的验证/测试时间点指标，不直接拼接旧表。

`experiment_config.yaml` 中 `retrain_8k: false` 默认不重训8K。若参考配置不兼容，默认阻止混合比较并写入审计；可显式开启 `retrain_8k: true` 后在本目录新实验中训练8K，原权重不覆盖。`allow_reference_mismatch` 默认false，不建议无理由打开。原检查点没有保存当时训练源代码快照，因此只能核验保存配置、优化器和现有实现，不能声称证明了历史源码逐字一致。

前期已查看过8K测试结果，本实验明确记录这一事实，不将整个研发过程包装成未接触测试集。新规模的选择只依据验证表现与预先设定的资源预算。

## 正式执行计划

1. 预检环境和数据，生成并打印五种配置，核对参考模型。
2. 新建带时间戳的正式目录，取得OS文件锁；记录配置、代码哈希、原权重哈希及数据文件身份。
3. 使用原 `SeismicDataset` 与16进程H5读取功能加载数据一次，读取进程全部结束后由主进程训练。原筛选、长度处理和年份划分不变，不再滤波或做归一化。
4. 逐项核对全部有序训练、验证和测试记录名称，确认与原标签缓存相同；预期605166/84564/51772条。缓存仅用于核对，不用它替代真实波形或训练标签。
5. 先加载现有8K并统一评估；再按2K→4K→16K→32K顺序训练并评估，不并行训练。若显式开启重训8K，则8K也在独立目录训练。
6. 每个模型完成后立即保存独立输出并重建汇总表和初步图；重启时跳过已完成模型。

全数据波形本身约26.5 GB，原训练函数还会保留逐输出预测/标签用于指标，另需若干GB内存。预检不分配这些完整数据。没有自动缩短输入、降低batch、修改模型或梯度累积替代。GPU显存不足、NaN、坏输出或缺文件时停止并保存错误日志；请处理原因后重新运行。

## 训练策略：复用原实现

- 调用原 `train_one_epoch`、`evaluate`、`multi_task_weighted_mse_loss`、`safe_clamp_preds` 和 `collate_fn`。
- 100 Hz、2976点、93输出，完整序列监督；震级、距离MSE及sin/cos平均MSE。
- 四阶段时间权重 `[0.7,0.8,0.9,1.0]`；三个log_sigma初始化 `-1,4,-1.5`，参与优化。
- AdamW，batch512，seed42，weight_decay=1e-5，betas=(0.9,0.999)，eps=1e-8。
- CNN基础/峰值学习率2e-4/6e-4；GRU、输出层与任务权重1e-3/3e-3。
- CyclicLR triangular2，10轮上升+10轮下降，**每个成功训练批次之后**更新，不是每轮更新。
- 全局梯度L2裁剪0.8，最多300轮；至少两项验证MAE改善超过1e-6才保存，并且只在该条件满足时更新改善项的历史最佳值；连续20轮未满足即早停。
- 训练初始化及随机种子设置顺序与原流程一致；每个模型重新固定seed，再初始化模型，不在初始化后额外重置训练随机数流。
- 原训练代码遇到坏批次会跳过；本实验的包装器在首次跳过告警时立即终止。正常有限批次的计算和调度没有变化。原测试代码会用数值替换NaN/Inf，本实验改为明确报错，不掩盖异常；有限输出仍执行原范围裁剪M[0,10]、D[0,200]、sin/cos[-1,1]。

每个训练轮次结束后原子提交恢复检查点，包含当前权重、AdamW状态、CyclicLR状态、轮次、完整历史、三个历史最佳值、早停计数、最佳模型快照以及Python/NumPy/PyTorch/全部CUDA RNG状态。进程在某轮中途退出时，重新运行会从上一个完整提交轮次重做该轮，不能称为批次级无损续跑。CPU合成测试已验证恢复路径与不中断训练逐权重、优化器和历史完全相同；不承诺跨硬件/CUDA版本逐位一致。

## 资源定义

参数数目由实际模块枚举，并与解析公式交叉核对。设CNN宽度w、GRU隐藏维度h：

```
CNN parameters = 4w² + 32w
GRU parameters = 3hw + 9h² + 12h
Output parameters = 4h + 4
Total parameters = 4w² + 32w + 3hw + 9h² + 16h + 7
CNN MACs/update = 32 × (4w² + 15w)
GRU MACs/update = 32 × (3hw + 9h²)
Output MACs/update = 4h
Conv cache FP32 values = 30w
GRU hidden FP32 values = 2h
Persistent state KiB = 4 × (30w + 2h) / 1024
```

统计时实际运行batch=1的 `stream_step`，在Conv1d、GRU和Linear注册hook验证公式。**GRU每次更新执行32个递归时间步、两层均计入**，不是每32点只做一次GRU。MACs包括卷积和矩阵乘加，不含偏置加法、LayerNorm、非线性、门控逐元素运算、残差加法及内存复制；不应冒充完整FLOPs或实测延迟。

持久状态只计算模型卷积历史和GRU隐藏状态；输入H5已经预处理，不再次增加滤波缓存。临时激活/峰值工作内存未测量，不记成0，不与权重或持久状态混合。可选 `model_scaling.benchmark_cpu` 提供真实batch=1流式CPU计时，正式流程默认不执行。

## 结果与图

每个正式目录包含：

- `model_configs.csv`、`compute_resources.csv`、`fairness_audit.json`、`experiment_identity.json`、`split_audit.json`。
- 每个模型子目录：配置、日志、状态、最佳权重 `best.pth`、恢复权重 `resume_checkpoint.pth`、训练历史、统一时间点验证/测试误差、资源JSON。**复用8K时不生成伪造的新最佳权重，而记录原权重来源**。
- 根目录：`training_summary.csv`、`accuracy_by_time.csv`、`pareto_results.csv`、`validation_budget_recommendations.csv`、`summary.md`。
- `figures/`：参数量—精度、MACs—精度、模型资源成本三组初步图，各有中英文PDF和600 DPI PNG。只用真实完成模型的结果，不用合成值填缺失模型。

3/5/10/20/29.76秒分别使用2.88/4.80/9.92/19.84/29.76秒输出。表中保存请求时间、实际输出时间和chunk编号。时间是原H5的P对齐窗时间，真实标注P与裁剪点可另有0–0.009秒偏移。

帕累托结果分别针对验证/测试、3/20秒、三种资源，计算单任务误差与资源二维非支配关系，以及三任务误差+资源四维非支配关系。相同点允许同时非支配；只称候选集合经验前沿。不同单位的MAE不直接相加。

预算建议仅使用验证集：每个预设预算给出分别最小化各任务误差的候选及三任务非支配候选，不根据测试误差挑最终主模型。汇总会如实报告32K相对2K的收益、任务差异、早晚期收益、容量增加却退化的情形及8K所处位置；未完成全部模型时明确标注部分结果。

## 文件职责

- `run_scaling_experiment.py`：Windows spawn安全入口。
- `study.py`：预检、历史公平性审计、数据与状态管理、顺序调度。
- `model_scaling.py`：原模块宽度配置、资源统计和流式验证、可选CPU benchmark。
- `training_adapter.py`：复用原epoch函数，补充完整断点恢复及严格错误处理。
- `reporting.py`：实际结果汇总、验证集预算建议、帕累托及初步图。
- `synthetic_checks.py`：少量合成数据的恢复/早停/坏批次/帕累托检查，不画合成实验结果。
- `experiment_config.yaml`：集中配置；修改训练配置或源码后不会混入旧结果，需 `--new-run`。

预检报告：`checks/preflight_report.json`、`checks/fairness_audit.json`、`checks/model_configs.csv`、`checks/compute_resources.csv`、`checks/synthetic_checks.json` 和 `checks/preflight.log`。
