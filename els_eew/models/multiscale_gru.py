# -*- coding: utf-8 -*-
"""完整 MultiScaleGRU：三个因果尺度分支 + 两层 GRU。

借鉴 InceptionTime 的多尺度思想；不是原集成结构/全局池化复现。
缓存保留每个分支各自需要的历史，详见 docs/MultiScaleGRU.md。
"""
import torch
from torch import nn
from torch.nn import functional as F
from .common import CausalConv
from .students import ChannelLayerNorm

CONFIG = dict(width=16, hidden=24, kind="multi", dilations=(1, 2, 4), mode="gru")


class MultiScaleResidualBlock(nn.Module):
    """三个核宽 3/7/15 的因果 depthwise 分支，拼接后 pointwise 混合。"""
    def __init__(self, width, dilation):
        super().__init__()
        self.convs = nn.ModuleList([CausalConv(width, width, k, dilation, width)
                                    for k in (3, 7, 15)])
        self.norms = nn.ModuleList([ChannelLayerNorm(width) for _ in range(3)])
        self.project = nn.Conv1d(3 * width, width, 1)

    def forward(self, x, state=None):
        states = [None] * 3 if state is None else state
        if len(states) != 3:
            raise ValueError("多尺度残差块需要三个分支缓存")
        outputs, new_states = [], []
        for conv, norm, old in zip(self.convs, self.norms, states):
            y, new = conv(x, old)
            outputs.append(F.gelu(norm(y)))
            new_states.append(new)
        return x + self.project(torch.cat(outputs, dim=1)), tuple(new_states)


class WaveformFrontend(nn.Module):
    """逐点输入投影；各残差块分别保存自己的历史缓存。"""
    def __init__(self, width, dilations):
        super().__init__()
        self.input = nn.Conv1d(3, width, 1)
        self.blocks = nn.ModuleList([MultiScaleResidualBlock(width, d) for d in dilations])

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


class MultiScaleGRU(nn.Module):
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
