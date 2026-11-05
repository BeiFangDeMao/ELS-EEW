# -*- coding: utf-8 -*-
"""完整 CompactGRU 对照：16通道因果DS前端 + 两层24维GRU，不加辅助特征。

与PhysicsGRU使用同一基础结构，独立文件便于审阅和消融。
默认参数路径、输出与状态格式保持一致。输入2976点输出93次。
"""
import torch
from torch import nn
from torch.nn import functional as F
from .common import CausalConv
from .students import ChannelLayerNorm

CONFIG = dict(width=16, hidden=24, kind="ds", dilations=(1, 2, 4, 8), mode="gru")


class DSResidualBlock(nn.Module):
    """核宽 3 的因果 depthwise -> 逐时刻 LN/GELU -> pointwise -> 残差。"""
    def __init__(self, width, dilation):
        super().__init__()
        self.convs = nn.ModuleList([CausalConv(width, width, 3, dilation, width)])
        self.norms = nn.ModuleList([ChannelLayerNorm(width)])
        self.project = nn.Conv1d(width, width, 1)

    def forward(self, x, state=None):
        if state is not None and len(state) != 1:
            raise ValueError("DS 残差块需要一个卷积缓存")
        y, cache = self.convs[0](x, None if state is None else state[0])
        return x + self.project(F.gelu(self.norms[0](y))), (cache,)


class WaveformFrontend(nn.Module):
    """逐点输入投影；各残差块分别保存自己的历史缓存。"""
    def __init__(self, width, dilations):
        super().__init__()
        self.input = nn.Conv1d(3, width, 1)
        self.blocks = nn.ModuleList([DSResidualBlock(width, d) for d in dilations])

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


class CompactGRU(nn.Module):
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
