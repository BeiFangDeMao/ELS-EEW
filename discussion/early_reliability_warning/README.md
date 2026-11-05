# 讨论实验1：早期参数可靠性与不可撤回预警决策

模型固定为 `els_eew/outputs/experiment_20261012_223631_490021/CompactGRU_seed42`
中的原始最佳 CompactGRU 权重（第258轮），不训练、不微调、不改模型。

直接运行 `run_experiment.py`：先做16条跨测试集抽取的小样本流式/全窗口一致性及后缀因果检查，
再对完整独立测试集重新推理，最后统计。每次使用新时间戳结果目录，不覆盖旧实验。

命令行：

```
python run_experiment.py --validate-only
python run_experiment.py --run-dir 已有小样本验证结果目录
python run_experiment.py --statistics-only 已有完整预测结果目录
```

依赖现有模型工程和源H5，以及 NumPy/Pandas/PyTorch/h5py/PyArrow/SciPy/scikit-learn/tqdm。
模型输入 `(B,3,2976)`，Z/N/E，gal；模型原始四输出为震级、距离/km、方位sin/cos。
保持原测试裁剪：震级[0,10]、距离[0,200]、sin/cos[-1,1]。
既存H5已完成因果处理，不增加归一化、滤波或积分。
完整窗口的现有因果 forward 只有通过与已有 stream_step 一致性检查后才用于批量推理。
每条记录每个32点块的结果对应名义P对齐后0.32、0.64……29.76秒；精确标注P后的时间还保留裁剪对齐偏移。

本实验读取经过原加载器筛选并缓存的测试键清单，同时检查H5大小/修改时间、每条波形与真值，
并重新推理全部51772条2023/2024测试记录，核对原评估的逐chunk震级/距离MAE。
推理不是从原预测表复制；旧定位结果仅用于数值一致性核对。

文件职责：

- `config.py`：原模型/数据路径、时间点、容差和记录级CI设置。
- `inference.py`：最小验证、完整推理、元数据审计、球面坐标与Parquet保存。
- `analysis_metrics.py`：回归、互斥距离组、当前得分AUC、不可撤回策略、首次报警与CI。
- `results/<时间戳>/summary.md`：完整结果、样本支持及解释边界。

按用户最新要求，全部统计以单条波形为单位，不计算事件或台站ID，不做事件归组、首台选择或跨台时间比较。
每条波形独立运行9种不可撤回报警策略。记录键只用于逐波形结果追溯。
达标率、Precision和Recall的CI采用95% Wilson二项比例区间，统计单位为波形记录；
该区间不代表事件级区间，也未考虑不同记录可能存在的相关性。
头段绝对时间仅作为原始信息保留，不用于发震后或跨台排序；决策均基于各条波形P到时后的输入。

不绘制正式论文图片，不做双台融合。NaN在CSV中为空单元格，JSON中为null；未定义原因记录在statistics_manifest.json。
