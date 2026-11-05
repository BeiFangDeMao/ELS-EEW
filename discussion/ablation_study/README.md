# 讨论实验：ELS-EEW 8K完整消融

本目录与原训练、模型规模实验隔离。**只围绕现有8K CompactGRU：CNN宽度16、两层GRU隐藏维度24、4个残差块膨胀率1/2/4/8、卷积核3、GRU层间dropout0.1**。不训练16K/32K，不覆盖原权重或结果。

## 启动

1. 可先双击 `check_ablation.bat`。这只进行模型构造、已有权重加载、两个真实样本的前向/流式检查，以及小型合成数据前后向、断点恢复和缓存检查。**不会开始正式训练。**
2. 准备好后双击 **`run_ablation.bat`**，自动按顺序运行全部启用的实验。
3. VS Code直接打开 **`run_ablation.py`** 点击“运行Python文件”，效果与正式bat相同。

默认Python为 `%USERPROFILE%\.conda\envs\lightweight\python.exe`，可设置 `ABLATION_PYTHON`。bat结束或报错都会暂停保留窗口。入口采用Windows spawn安全写法，16个H5读取进程不会误启动训练。

命令行：

```powershell
& "$env:USERPROFILE\.conda\envs\lightweight\python.exe" discussion/ablation_study/run_ablation.py --check
& "$env:USERPROFILE\.conda\envs\lightweight\python.exe" discussion/ablation_study/run_ablation.py
```

只运行8K缓存一致性与CPU测试、不训练任何模型：

```powershell
& "$env:USERPROFILE\.conda\envs\lightweight\python.exe" discussion/ablation_study/run_ablation.py --cache-only
```

结果保存在 `results/YYYYMMDD_HHMMSS_microseconds_seed42/`。默认续跑 `active_run.json` 指向的目录，已完成且哈希验证通过的模型自动跳过。需要全新实验时使用 `--new-run`。改变代码、数据或训练配置后，拒绝混入原实验；不要为了续跑删除身份校验。

## 哪些需要训练

|实验|动作|输出维数|实际总参数|
|---|---|---:|---:|
|ELS8K_Full，兼作A0/B0/C3|复用原8K最佳权重，跳过训练|4|8263|
|PureGRU_Existing，A1|复用原PureGRU类和最佳权重，跳过训练|4|5791|
|Single_magnitude|从头训练，8K原骨干|1|8186|
|Single_distance|从头训练，8K原骨干|1|8186|
|Single_azimuth|从头训练，8K原骨干|2|8211|
|C0_Uniform_Fixed|从头训练，统一时间权重，冻结q|4|8263（可训练8260）|
|C1_Staged_Fixed|从头训练，原四阶段，冻结q|4|8263（可训练8260）|
|C2_Uniform_Learned|从头训练，统一时间权重，可学习q|4|8263|
|缓存D0/D1/D2|只使用原8K最佳权重，不训练|4|8263|

**默认共6个新增训练模型，最多各300轮。** 不重新训练完整8K或PureGRU。参数量包含训练任务权重，计数口径与原检查点一致。

没有发现现成的参数量匹配PureGRU。现有PureGRU为5791参数，与8263相差约30%，不能称参数量匹配。依照补充说明，**不重新实现PureGRU、不扩展其隐藏维度、不新增训练**；架构表将 `ParameterMatchedPureGRU` 标记为unavailable，而不是填假指标。CNN作用可作现有结构对比，严格参数匹配结论保持开放。

## 独立开关

`ablation_config.yaml` 的 `groups.architecture / multitask / loss / cache` 可分别启用关闭。完整8K作为共用参考只评估一次，不重复训练。首次运行前设置开关；一个运行开始后如改变开关，使用 `--new-run` 保持身份清楚。CPU缓存阶段出错不会删除之前训练完成的结果。

## 原模型与已有PureGRU

8K只读加载原 `CompactGRU_seed42/checkpoint_CompactGRU_20261012_235806/best_CompactGRU_20261012_235806.pth`，原最佳轮次258。

PureGRU只读加载原 `PureGRU_seed42/checkpoint_PureGRU_20261012_223634/best_PureGRU_20261012_223634.pth`，最佳轮次64、训练完成84轮。原类直接导入 `models.pure_gru.PureGRU`，未重新实现。它是两层hidden24，三分量直接进入GRU，每32点读输出；原CNN参数组为空，实际GRU学习率沿用head设置。

预检核对原run_config、结构、数据集、序列长度、seed、batch、轮次预算、早停和优化器。若PureGRU的历史设置有差异，写入 `reference_audit.json`，仍优先用已有权重统一测试并保留差异说明，不偷偷重训。历史训练源码快照未保存在旧权重内，不能证明历史源码逐字一致。

原表缺少本次所需的定位误差、全部稳定性统计，因此正式流程会重新加载8K和PureGRU权重，**统一推理原独立测试集**，不重复训练。原指标文件作为预检与溯源参考，新的结果写入本目录。

## 数据、训练与恢复

复用规模实验配置及原 `training.Trainstudents.SeismicDataset`、H5读取、collate、指标、优化器/CyclicLR与随机数状态辅助函数。保持原筛选、ZNE、gal、100Hz、2976点、32点输出、93次监督，不额外滤波或做全窗口归一化。

原年份划分：2019–2022验证，2023–2024测试，其余合格年份训练，605166/84564/51772记录。正式载入后逐条核对有序记录键与原选择缓存，绝不重新随机划分。数据原有P对齐离散偏移约0–0.009 s，图表使用名义P后观测时间。

固定AdamW、batch512、seed42、weight_decay1e-5、betas(0.9,0.999)、eps1e-8；CNN基础/峰值LR2e-4/6e-4，GRU/输出及q为1e-3/3e-3。triangular2每个成功批次更新，10轮升/10轮降，全局梯度L2裁剪0.8，最多300轮，patience20。没有更换此前讨论的早停门槛。

共享的活动输出epoch适配器支持1/2/4维头，保留原监督、钳制和完整序列指标定义。原三任务四阶段损失直接调用原函数；其他消融只在声明的时间权重/活动任务上改变。预检逐值核对完整损失及梯度与原实现相同。NaN/Inf、坏梯度、OOM明确停止，不自动修改batch、结构或输入长度。

三任务沿用原“至少两项历史最佳改善超过1e-6”，仅在满足该条件时更新改善项的历史最佳值；单任务按自身验证MAE改善选择。**选择准则不同，必须披露**。

每轮原子保存当前模型、AdamW、CyclicLR、轮次、早停计数、历史最佳值、完整历史、最佳模型快照、Python/NumPy/Torch/CUDA RNG。中途退出从上一个完整轮次恢复，部分轮次重新执行。合成CPU恢复检查与不中断训练逐位对比。达到终止轮次的检查点不执行额外训练轮次。

完整H5读取复用原eager框架，波形约26.5GB，加上训练指标数组还需额外内存；预检和 `--cache-only` 不加载整个训练集。

## 时间与任务权重

原四阶段为 `[0.7,0.8,0.9,1.0]`，分段公式与原代码相同。统一时间权重保留要求的0.85。**93个输出实际分成24/23/23/23，原平均权重0.848387；0.85是近似等尺度，不能写成精确相同。**

q初始化为震级-1、距离4、方位角-1.5；损失仍为 `L_j/(2*exp(q_j)^2)+q_j`。固定权重版本只冻结原q，不换一套任意损失比值。单任务只保留对应q，默认学习自己的q，不保留其他任务q或输出；对应sin/cos平均MSE。可用 `single_task_learn_own_q` 显式关闭自身q学习，但默认与原机制保持一致。

稳定性计算相邻0.32秒输出的绝对变化，角度使用最小圆周差；早期窗口0–5 s，后期10–29.76 s，两个端点均须在窗口内。它不是单位时间导数，表中明确每次更新的单位；同时报告对应预测MAE，常数模型不会被解释为既稳定又准确。

## 缓存对比

- D0：每次从零状态计算完整已到达前缀32/64/…/2976点，取最新输出，无未来输入。
- D1：原 `stream_step`，卷积特征缓存+GRU隐藏状态，新增32点一次更新。
- D2：保留GRU隐藏状态、最近30点**原始波形上下文**，重算局部CNN并只把当前32点特征送入GRU。历史30点来自实际感受野 `sum(2*dilation)=30`，没有把原始输入与特征缓存混淆。
- 一次完整序列因果forward仅用作93次输出数值参考，不能被当成可提前获取未来波形的在线方案。

逐条记录重置状态，逐输出分别保存M/D/sin/cos的最大、平均绝对差，容限rtol2e-4/atol2e-5；缓存错误会中止该阶段，不产生伪造效率结论。

MACs用真实Conv/GRU/Linear hook统计，计入32次GRU递推。D0每次计算量随前缀增长；D1恒定243296MACs/update；D2首块243296、之后281216。93次更新D0总量为D1的47倍（理论乘加，不是实际延迟加速比）。激活、归一化、残差加法和内存复制不属于本MAC口径。

分别记录模型状态、输入历史缓冲、两者合计：D1模型状态2.0625KiB；D2原始上下文0.3515625KiB+GRU0.1875KiB；D0模型历史状态0，但需存储不断增长的完整输入前缀，最终34.875KiB，不能写成“无存储”。临时工作内存不填成0。

默认本机CPU单线程、batch1、FP32、eval+inference_mode；8条真实测试记录、预热2条、重复5次。输入和权重预加载，计时不包括文件读取、日志或绘图；初始化分开测量，更新内的必要缓存拼接/复制计入真实开销。保留逐重复逐时间原始延迟，统计中位数、P95和整条93次累计时间。只说明本机CPU的模型计算能力，不宣称嵌入式部署或端到端系统时延。

不默认计时预处理：输入H5已经因果处理，不能重复滤波。`end_to_end_preprocessing` 当前明确拒绝启用；未来若有原始未处理输入，应另实现单独端到端口径，不能混入本结果。

## 输出

根目录：`architecture_ablation.csv`、`multitask_ablation.csv`、`multitask_system_costs.csv`、`loss_ablation.csv`、`loss_ablation_key_times.csv`、`stability_ablation.csv`、`cache_consistency.csv`、`cache_efficiency.csv`、`ablation_summary.md`，以及源配置、权重哈希、公平性审计、逐模型资源表。

每模型目录：配置、训练日志、训练历史、最佳和恢复权重（仅新训练模型）、93个时刻的误差指标、完整预测npy、真值、稳定性、阶段状态和输出哈希。复用模型只记录原权重来源，不复制伪造训练记录。

缓存目录：逐更新时间的MACs、缓冲和状态内存、数值一致性、原始CPU延迟、环境说明。此阶段失败不会删除训练模型。

`figures/`：架构、任务、训练权重、缓存效率四组初步分析图，中英文PDF与600 DPI PNG，只使用实际已完成结果。任务完整系统成本按三套单任务骨干合计，不能拿单个单任务模型当作完整三任务系统成本。

自动分析逐项报告PureGRU是否有优势、联合任务竞争、资源节约、时间/q权重效果、稳定性与误差、缓存数值差异、理论成本和实际CPU加速、P95相对320ms更新周期；未完成的项目明确保留，绝不填模拟值。参数匹配PureGRU问题未开展，不能给肯定结论。

## 代码分工

- `ablation_runner.py`：审计、统一调度、独立开关和续跑。
- `ablation_models.py`：原模块的单任务头、冻结q、损失适配与资源统计；PureGRU直接导入原类。
- `ablation_training.py`：活动输出epoch及完整训练恢复。
- `ablation_evaluation.py`：统一预测误差、球面定位、稳定性。
- `cache_benchmark.py`：可独立调用的D0/D1/D2一致性、MACs与CPU时间函数。
- `ablation_reporting.py`：CSV、真实数据绘图、自动讨论。
- `ablation_checks.py`：短时间预检，不开展真实数据训练。

依赖使用原lightweight环境，见 `requirements.txt`；额外需要现有 `discussion/scaling_study` 和原项目公共代码，不更改它们。字体检查要求SimSun和Times New Roman。预检报告保存在 `checks/`。


## 2026-10-21 ???????????

????????0.8???????????????????FP32???????FP64??????????????????????????????????????????????????????RNG?`gradient_diagnostics/`?????????RNG????TF32???cuDNN???????????????????????????????????????/??????????????NaN???????????????????????????/??????JSON?????dropout????????????????????????????????????TF32????????????????batch??????

??????`gradient_repair_compatibility.json`??????????????/??/?????????????????????????????`run_ablation.bat`???????????Single_azimuth??????49????????????50??????????????`--new-run`?????????????????
