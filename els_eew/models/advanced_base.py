"""无参数公共工具：缓存张量校验、已完成32点块的特征均值。

模型前端、输出头及forward/stream_step均已展开在对应模型文件。
"""
def check_state_tensor(tensor, reference, shape):
    if tuple(tensor.shape) != tuple(shape) or tensor.device != reference.device or tensor.dtype != reference.dtype:
        raise ValueError("State shape/device/dtype mismatch; reset per independent record")


def completed_block_features(features):
    """仅在块末使用完整32点的均值；返回(B,K,C)。"""
    b, c, length = features.shape
    return features.reshape(b, c, length // 32, 32).mean(-1).transpose(1, 2)
