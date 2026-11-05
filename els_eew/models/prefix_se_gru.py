# -*- coding: utf-8 -*-
"""PrefixSEGRU：用前缀均值替换SE的整段池化，逐时刻重标定通道。

SENet依据 Hu et al., CVPR2018；前缀累计是本项目严格因果适配。
详见 docs/PrefixSEGRU.md。
完整前端/输出头/forward/stream_step在本文件展开；仅复用基础算子和无参数工具。
"""
import torch
from torch import nn
from torch.nn import functional as F
from .advanced_base import check_state_tensor
from .common import CausalConv
from .students import ChannelLayerNorm

CONFIG = dict(width=16, hidden=24, dilations=(1, 2, 4, 8), reduction=4,
              mechanism="prefix_squeeze_excitation", recurrent_rate="sample")


class PrefixSEBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.temporal = CausalConv(width, width, 3, dilation, width)
        self.norm = ChannelLayerNorm(width)
        self.mix = nn.Conv1d(width, width, 1)
        reduction = CONFIG['reduction']
        if reduction < 1:
            raise ValueError("reduction must be positive")
        reduced = max(1, width // reduction)
        self.squeeze = nn.Conv1d(width, reduced, 1)
        self.excite = nn.Conv1d(reduced, width, 1)

    def forward(self, x, state=None):
        cache, total, count = (None, None, 0) if state is None else state
        y, cache = self.temporal(x, cache)
        y = self.mix(F.gelu(self.norm(y)))
        if total is None:
            total = y.new_zeros(y.size(0), y.size(1))
        check_state_tensor(total, y, (y.size(0), y.size(1)))
        summed = y.cumsum(-1) + total.unsqueeze(-1)
        n = torch.arange(count + 1, count + y.size(-1) + 1, device=y.device, dtype=y.dtype)
        prefix_mean = summed / n
        gate = self.excite(F.relu(self.squeeze(prefix_mean))).sigmoid()
        return x + y * gate, (cache, summed[..., -1].clone(), count + y.size(-1))


class WaveformFrontend(nn.Module):
    """逐点输入投影；各残差块分别保存自己的历史缓存。"""
    def __init__(self, width, dilations):
        super().__init__()
        self.input = nn.Conv1d(3, width, 1)
        self.blocks = nn.ModuleList([PrefixSEBlock(width, d) for d in dilations])

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


class PrefixSEGRU(nn.Module):
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
