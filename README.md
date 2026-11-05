# ELS-EEW / CompactGRU

精简版：仅保留CompactGRU主训练链与discussion实验代码。

## 主训练

在IDE运行 `els_eew/train_CompactGRU.py`。
数据路径、训练参数、随机种子等在 `els_eew/experiment_config.py` 设置。
批量入口默认也只训练CompactGRU，没有启动任何训练。
主模型实现：`els_eew/models/compact_gru.py`。
数据加载、预处理及损失：`els_eew/training/`。
模型目录内其他实现是统一注册器或讨论实验依赖，不是默认训练任务。
原目录名保留，以兼容discussion的相对路径。

## 讨论实验

- `discussion/early_reliability_warning`：早期可靠性和预警评价。
- `discussion/scaling_study`：模型规模实验。
- `discussion/ablation_study`：消融和缓存效率。
- `discussion/model_comparison`：对比模型、CPU延迟和内存测试；upstream为第三方基线依赖。
- `discussion/real_events`：真实事件回放。
- `discussion/two_station_streaming_eew`：双台站方案文档。

`plotting`仅保留讨论实验需要的共享publication_style.py。
H5、权重、预测结果、labels.npz和地图文件不在源码包中，需自行配置。
部分讨论脚本仍引用原plotting位置的数据文件；删除了绘图工程，但不改变这些输入位置。
第三方许可参见THIRD_PARTY_NOTICES.md。本次没有发布或上传GitHub。
