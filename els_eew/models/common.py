"""模型共享基础算子：带显式历史缓存的因果卷积。

各候选的网络主体和流式接口已展开在对应模型文件。
advanced_base.py仅保留缓存校验与完成块均值的无参数工具。
"""
import torch
from torch import nn


class CausalConv(nn.Module):
    def __init__(self, ci, co, kernel=3, dilation=1, groups=1):
        super().__init__()
        self.history = (kernel-1)*dilation
        self.conv = nn.Conv1d(ci, co, kernel, dilation=dilation, groups=groups)

    def forward(self, x, cache=None):
        if cache is None:
            cache = x.new_zeros(x.size(0), x.size(1), self.history)
        expected = (x.size(0), x.size(1), self.history)
        if tuple(cache.shape) != expected or cache.device != x.device or cache.dtype != x.dtype:
            raise ValueError("Invalid convolution state")
        combined = torch.cat([cache, x], -1)
        state = combined[:, :, -self.history:].clone() if self.history else combined[:, :, :0]
        return self.conv(combined), state


