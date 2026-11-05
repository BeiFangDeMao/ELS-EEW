# 288点四模型训练

## 启动

先等待 `build_dataset_288.py` 制作完成，manifest状态必须为completed。

1. 可双击 `check_training.bat`：只用合成数据检查四个模型的288点前向、损失和反向梯度，不开始正式训练。
2. 双击 `run_training.bat`：自动选取最新完成的 `data/SeisT_MagNet_288_Causal_*/`，按顺序训练四个模型，不并行占用GPU。
3. 也可在IDE直接运行 `train_comparison.py`。

默认解释器为 `%USERPROFILE%\.conda\envs\lightweight\python.exe`，可设置 `COMPARISON_PYTHON` 指向其他环境。不会自动安装包。

## 四个任务

|模型|标签|损失|验证/测试指标|
|---|---|---|---|
|seist_s_emg|mag|Huber，delta=1|震级MAE|
|seist_s_dis|dis，km|Huber，delta=1|震中距MAE/km|
|seist_s_baz|azi，度|原始角度标量Huber，delta=1|最小圆周角误差MAE/度|
|magnet|mag|官方MousaviLoss，按样本数取均值|震级MAE|

SeisT选用官方Small骨干，约98283参数；MagNet为SeisT仓库的PyTorch复现，114418参数。SeisT的方位角损失保持官方标量Huber，边界0/360度不是圆周损失，评价指标使用圆周误差；必须在结果说明中披露。

SeisT震级原输出范围0–8不足以覆盖本数据最高9级，因此输出激活缩放设置为0–10（配置中的magnitude_output_max）。其余骨干及输出保持上游实现。此项是本项目适配，不宣称逐项复现官方实验。

## 数据接口

`comparison_dataset.py` 严格读取已完成的因果288点H5及同目录cohort.npz，返回float32 `(3,288)`、单任务 `(1,)` 标签和记录键。
每个Windows读取进程独立打开只读H5；不再滤波、归一化或随机增强。H5与清单的split、任务标签不一致或数据非有限时立即报错。
固定原有训练/验证/测试划分；没有重新随机划分。测试只在验证集选定最佳权重后执行。

## 配置与结果

所有训练设置集中在 `train_config.json`。默认batch256、AdamW固定lr=1e-3、weight_decay=1e-5、梯度裁剪0.8、seed42、最多300轮、验证MAE早停patience20、最小改善1e-6。没有学习率调度器。
这些是本项目共同训练协议，未照搬SeisT官方CyclicLR方案，也未完成学习率搜索。后续若比较最优性能，可在相当的验证集搜索预算下调参，不能依据测试误差选设置。

输出在 `training_results/<时间戳>/<模型名>/`，包含训练日志、逐轮JSON、最佳权重、恢复权重、测试逐记录预测和完成指标；根目录summary.json汇总四模型结果。

再次启动默认续跑active_run.json指向的实验，跳过已经完成的模型。恢复粒度为完整epoch；中断轮次重新执行。保留优化器、早停与随机状态，但不承诺跨硬件/版本逐位重现。
如配置、数据或源码改变，拒绝混入旧实验；新实验运行：

```powershell
& "$env:USERPROFILE/.conda/envs/lightweight/python.exe" discussion/model_comparison/train_comparison.py --new-run
```

dataset_dir为auto时采用最新完成数据；建议正式训练前将其改成目标数据目录的绝对路径以固定来源。
CUDA不可用、OOM、无效梯度或损失会停止，不自动缩小batch、不跳过坏批次。

## 已验证范围

当前lightweight环境下，四个模型均已通过合成 `(2,3,288)` 输入的前向、损失、反向梯度检查。未启动正式训练，未对尚在制作的数据集执行完整读取。启动正式训练时还会重新做模型预检并读取每个任务的一条训练记录。

E3WS不在此次训练范围内；CompactGRU不重训。这里是三个单任务SeisT和一个震级MagNet，不能将四个任务误写为四个三任务模型。
