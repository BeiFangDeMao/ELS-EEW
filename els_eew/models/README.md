# 当前17个完整模型文件

以下文件均直接定义该模型的前端、输出头、完整模型类、forward和stream_step。
核心结构可在对应文件中审阅；仅复用CausalConv、逐时刻通道LN及无参数状态校验/块平均工具。
所有模型在每32点完成后输出一次；下面的时间单位是历史状态内部更新的频率。

|模型编号|实际文件|核心结构|状态更新单位|
|---|---|---|---|
|S4_stream_LN|[students_streaming.py](students_streaming.py)|S规模DS卷积＋GRU32×2|采样点|
|CompactGRU|[compact_gru.py](compact_gru.py)|DS卷积＋GRU24×2|采样点|
|MobileGRU|[mobile_gru.py](mobile_gru.py)|倒残差卷积＋GRU24×2|采样点|
|MultiScaleGRU|[multiscale_gru.py](multiscale_gru.py)|3/7/15多尺度卷积＋GRU24×2|采样点|
|BlockQRNN|[block_qrnn.py](block_qrnn.py)|DS卷积＋两层fo-QRNN|完成32点块|
|PhysicsGRU|[physics_gru.py](physics_gru.py)|DS卷积＋GRU24×2＋因果统计|采样点|
|LongDS_TCN|[ds_tcn.py](ds_tcn.py)|11层膨胀DS-TCN＋线性头|无循环单元|
|PartialMixGRU|[partial_mix_gru.py](partial_mix_gru.py)|部分普通时序卷积＋GRU24×2|采样点|
|ShuffleGroupGRU|[shuffle_group_gru.py](shuffle_group_gru.py)|分组卷积/洗牌＋GRU24×2|采样点|
|GatedDSGRU|[gated_ds_gru.py](gated_ds_gru.py)|门控深度卷积＋GRU24×2|采样点|
|PrefixSEGRU|[prefix_se_gru.py](prefix_se_gru.py)|前缀SE卷积＋GRU24×2|采样点|
|DynamicKernelGRU|[dynamic_kernel_gru.py](dynamic_kernel_gru.py)|动态核混合卷积＋GRU24×2|采样点|
|LowRankGRU|[low_rank_gru.py](low_rank_gru.py)|DS卷积＋两层低秩GRU|完成32点块|
|BlockSRU|[block_sru.py](block_sru.py)|DS卷积＋两层SRU|完成32点块|
|BlockIndRNN|[block_indrnn.py](block_indrnn.py)|DS卷积＋两层独立循环单元|完成32点块|
|BlockLinearAttention|[block_linear_attention.py](block_linear_attention.py)|DS卷积＋核注意力累计状态|完成32点块|
|SoftForestGRU|[soft_forest_gru.py](soft_forest_gru.py)|DS卷积＋GRU24×2＋概率树头|采样点|

## 接口与文件定位

- forward(wave)：输入(B,3,32*K)，返回预测(B,K,4)与卷积特征。
- stream_step(wave_chunk,state=None)：eval模式下只输入新增(B,3,32)，返回(B,1,4)及新状态。
- 新记录以state=None开始；持续记录将返回状态传入下次调用。
- 四维预测为震级、距离、方向sin/cos；2976点产生93次预测。
- CompactGRU有独立文件；physics_gru.py中的旧导入仍保留兼容。
- S4_stream_LN是项目S4编号的LN对照，不是Structured State Space模型。
- 模型注册见[candidates.py](candidates.py)；单模型训练入口在上一级train_模型名.py。

## 核对范围

纯静态检查已通过：53个Python文件语法/相对导入、17个模型本地前端/输出头、
显式forward/stream_step、注册与训练入口，不存在pass/NotImplementedError占位。
代码准备阶段没有运行模型前向、反向或正式训练。
数值等价性、梯度和实际效果需运行[可选模型检查](../tools/verify_models.py)及训练后确认。
