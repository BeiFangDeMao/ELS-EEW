# -*- coding: utf-8 -*-
"""DynamicKernelGRU：每个时刻根据当前特征混合两个depthwise卷积专家。

Dynamic Convolution依据 Chen et al., CVPR2020；不用整段全局池化。
计算两个专家的响应再混合，不声称原论文聚合核的FLOPs/速度。
详见 docs/DynamicKernelGRU.md。
完整前端/输出头/forward/stream_step在本文件展开；仅复用基础算子和无参数工具。
"""
import torch
from torch import nn
from torch.nn import functional as F
from .common import CausalConv
from .students import ChannelLayerNorm

CONFIG = dict(width=16, hidden=24, dilations=(1, 2, 4, 8), experts=2,
              mechanism="causal_dynamic_depthwise", recurrent_rate="sample")


class DynamicKernelBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.num_experts = CONFIG['experts']
        if self.num_experts < 1:
            raise ValueError("experts must be positive")
        self.experts = nn.ModuleList([CausalConv(width, width, 3, dilation, width)
                                      for _ in range(self.num_experts)])
        self.router = nn.Conv1d(width, self.num_experts, 1)
        self.norm = ChannelLayerNorm(width)
        self.mix = nn.Conv1d(width, width, 1)

    def forward(self, x, state=None):
        states = [None] * self.num_experts if state is None else state
        if len(states) != self.num_experts:
            raise ValueError("Dynamic convolution expert cache count mismatch")
        outputs, caches = [], []
        for expert, old in zip(self.experts, states):
            y, cache = expert(x, old)
            outputs.append(y)
            caches.append(cache)
        # 路由器仅看当前时刻通道，不看未来；两个专家共享输入但权重不同。
        weights = self.router(x).softmax(dim=1)
        mixed = sum(y * weights[:, i:i+1] for i, y in enumerate(outputs))
        return x + self.mix(F.gelu(self.norm(mixed))), tuple(caches)


class WaveformFrontend(nn.Module):
    """逐点输入投影；各残差块分别保存自己的历史缓存。"""
    def __init__(self, width, dilations):
        super().__init__()
        self.input = nn.Conv1d(3, width, 1)
        self.blocks = nn.ModuleList([DynamicKernelBlock(width, d) for d in dilations])

    def forward(self, wave, state=None):
        x = self.input(wave)
        states = [None] * len(self.blocks) if state is None else state
        if len(states) != len(self.blocks):
            raise ValueError("前端缓存数量错误")
        new_states = []
        for block, old in zip(self.blocks, states):
            x, new = block(x, old)
            new_states.append(new)
        return x, tuple(new_states)


class GRUOutputHead(nn.Module):
    """两层 24 维 GRU 按采样点递推；每 32 点读取一次输出。"""
    def __init__(self, width, hidden):
        super().__init__()
        self.gru = nn.GRU(width, hidden, 2, batch_first=True, dropout=.1)
        self.proj = nn.Linear(hidden, 4)

    def forward(self, features, wave, state=None):
        hidden, stats = (None, None) if state is None else state
        output, hidden = self.gru(features.transpose(1, 2), hidden)
        return self.proj(output[:, 31::32]), (hidden, stats)


class DynamicKernelGRU(nn.Module):
    """完整模型：前端、输出头和状态处理均在本文件展开。"""
    chunk_size = 32

    def __init__(self):
        super().__init__()
        self.architecture_config = dict(CONFIG)
        self.feature_extractor = WaveformFrontend(CONFIG['width'], CONFIG['dilations'])
        self.output_head = GRUOutputHead(CONFIG['width'], CONFIG['hidden'])
        self.log_sigma_mag = nn.Parameter(torch.tensor(-1.))
        self.log_sigma_dis = nn.Parameter(torch.tensor(4.))
        self.log_sigma_azi = nn.Parameter(torch.tensor(-1.5))

    def _compute(self, wave, state=None):
        if wave.ndim != 3 or wave.size(1) != 3 or wave.size(-1) < 32 or wave.size(-1) % 32:
            raise ValueError("输入需要为 (B,3,L)，L 为正的 32 倍数")
        front_state, head_state = (None, None) if state is None else state
        features, front_state = self.feature_extractor(wave, front_state)
        predictions, head_state = self.output_head(features, wave, head_state)
        return predictions, features, (front_state, head_state)

    def forward(self, wave):
        predictions, features, _ = self._compute(wave)
        return predictions, features

    @torch.no_grad()
    def stream_step(self, wave_chunk, state=None):
        """每次只处理新增的 32 点；state=None 表示开始独立的新记录。"""
        if self.training:
            raise RuntimeError("流式推理前请调用 model.eval()")
        if wave_chunk.ndim != 3 or wave_chunk.shape[1:] != (3, 32):
            raise ValueError("每次流式调用只接受新增的 32 点")
        reference = self.feature_extractor.input.weight
        if wave_chunk.device != reference.device or wave_chunk.dtype != reference.dtype:
            raise ValueError("输入的 device/dtype 必须与模型一致")
        predictions, _, new_state = self._compute(wave_chunk, state)
        return predictions, new_state
