# 架构与单任务消融：最终测试集MAE及R²

主表：`Ablation_Test_MAE_R2_2p88s_29p76s.csv`。9行、6列，与用户指定表格一致；每行重复填写Task，便于机器读取。CSV使用UTF-8 BOM，数值保留计算精度；论文展示可将震级及R²保留4位小数、距离和角度MAE保留2位小数。

数据来自 `../results/20261021_003609_054421_seed42`，使用已保存的最佳权重测试预测 `predictions.npy` 和真实标签 `truth_labels.npy`。ELS-EEW对应`ELS8K_Full`，PureGRU对应`PureGRU_Existing`，三个单任务模型分别对应`Single_magnitude`、`Single_distance`、`Single_azimuth`。

所有结果均为同一组51,772条测试波形记录，非独立地震事件数。两个实际输出时刻为2.88 s和29.76 s，即100 Hz、每32点更新下的第9、第93个输出。沿用已有P对齐及预测裁剪规则，不添加新的归一化、过滤或预测修正。

## 指标定义

- 震级MAE：`mean(abs(predicted_magnitude - true_magnitude))`，无量纲。
- 震中距MAE：同上，单位km。
- 方位角列中的MAE实际是MAAE：从保存的sin/cos通过`atan2(sin, cos)`恢复角度，取最小圆周角差的均值，单位°。表头保留用户指定的MAE。
- 震级、震中距R²：`1 - sum((prediction-truth)^2) / sum((truth-mean(truth))^2)`。
- **方位角R²使用用户确认的sin/cos联合定义**：令`y=(sin(theta), cos(theta))`，`p`为模型保存的两个分量，则`R² = 1 - sum_i ||p_i-y_i||² / sum_i ||y_i-mean(y)||²`。等价于scikit-learn的`r2_score(y, p, multioutput='variance_weighted')`，不是分别计算两项R²后的算术平均。
- 方位角R²基于保存的sin/cos分量，不进行额外单位长度归一化。因此它也受到分量幅值偏差影响；它不是对0–360°角度计算的普通R²，不应与该口径混用。
- 保留可能出现的负R²，不做截断。若真实值总方差为零，应记NaN；本表未出现该情况。

## 验证和复现

`compute_test_table.py`只读取原结果，检查五个模型的标签及顺序完全一致、所有状态complete、样本数一致。18个任务/模型/时间组合的MAE逐项核对已有`metrics_by_time.csv`，全部R²与scikit-learn独立核对。输入文件哈希在计算前后检查且与原状态文件核对；不改动权重、训练配置、预测或已有指标。

先用lightweight环境运行`compute_test_table.py`生成`test_table_values.json`，再用可加载`@oai/artifact-tool`的Node运行`export_test_table.mjs`导出CSV。本次没有重新推理或训练，没有启动此前停止的第二轮实验。
