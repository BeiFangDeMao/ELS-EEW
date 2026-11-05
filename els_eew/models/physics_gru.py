# -*- coding: utf-8 -*-
"""完整 PhysicsGRU：DS 卷积/GRU + 10 维历史幅值与交叉矩特征。

物理启发的特征融合，不是物理约束损失；详见 docs/PhysicsGRU.md。
CompactGRU 使用相同可学习前端/GRU，去掉辅助特征，作为必要消融。
依赖输入三分量的物理单位与 100 Hz 时间约定；不消除旧 H5 的预处理泄露。
"""
import torch
from torch import nn
from torch.nn import functional as F
from .common import CausalConv
from .students import ChannelLayerNorm

CONFIG = dict(width=16, hidden=24, kind="ds", dilations=(1, 2, 4, 8), mode="physical")


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


def physical_features(wave, state=None):
    """仅累计当前及历史样本：3 RMS、3 峰值、3 归一化交叉矩、1 时间。

    state=(平方和,交叉乘积和,历史峰值,已接收点数)。
    不按整段波形归一化，也不读取未来峰值。
    """
    b, _, length = wave.shape
    products = torch.stack((wave[:, 0] * wave[:, 1], wave[:, 0] * wave[:, 2],
                            wave[:, 1] * wave[:, 2]), dim=1)
    if state is None:
        sums, cross, peaks = [wave.new_zeros(b, 3) for _ in range(3)]
        count = 0
    else:
        sums, cross, peaks, count = state
        if count < 0:
            raise ValueError("历史样本计数不能为负")
        for tensor in (sums, cross, peaks):
            if tuple(tensor.shape) != (b, 3) or tensor.device != wave.device or tensor.dtype != wave.dtype:
                raise ValueError("物理特征缓存与输入不匹配")
    sq = wave.square().cumsum(dim=-1) + sums.unsqueeze(-1)
    cp = products.cumsum(dim=-1) + cross.unsqueeze(-1)
    pk = torch.maximum(wave.abs().cummax(dim=-1).values, peaks.unsqueeze(-1))
    n = torch.arange(count + 1, count + length + 1, device=wave.device, dtype=wave.dtype)
    rms = (sq / n).clamp_min(1e-12).sqrt()
    denom = torch.stack((sq[:, 0] * sq[:, 1], sq[:, 0] * sq[:, 2],
                         sq[:, 1] * sq[:, 2]), dim=1).clamp_min(1e-12).sqrt()
    corr = (cp / denom).clamp(-1, 1)
    elapsed = torch.log1p(n / 100).view(1, 1, -1).expand(b, 1, -1)
    auxiliary = torch.cat((torch.log1p(rms), torch.log1p(pk), corr, elapsed), dim=1)
    new_state = (sq[..., -1].clone(), cp[..., -1].clone(), pk[..., -1].clone(), count + length)
    return auxiliary[:, :, 31::32].transpose(1, 2), new_state


class PhysicsOutputHead(nn.Module):
    """两层 GRU24；可选在四维预测层前拼接 10 维因果辅助统计。"""
    def __init__(self, width=16, hidden=24, physical=True):
        super().__init__()
        self.physical = physical
        self.gru = nn.GRU(width, hidden, 2, batch_first=True, dropout=.1)
        self.proj = nn.Linear(hidden + (10 if physical else 0), 4)

    def forward(self, features, wave, state=None):
        hidden, statistics = (None, None) if state is None else state
        output, hidden = self.gru(features.transpose(1, 2), hidden)
        sampled = output[:, 31::32]
        if self.physical:
            auxiliary, statistics = physical_features(wave, statistics)
            sampled = torch.cat((sampled, auxiliary), dim=-1)
        return self.proj(sampled), (hidden, statistics)


class PhysicsGRU(nn.Module):
    chunk_size = 32

    def __init__(self, physical=True):
        super().__init__()
        self.architecture_config = dict(CONFIG, mode="physical" if physical else "gru")
        self.feature_extractor = WaveformFrontend(CONFIG['width'], CONFIG['dilations'])
        self.output_head = PhysicsOutputHead(CONFIG['width'], CONFIG['hidden'], physical)
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


# 兼容旧导入路径；完整CompactGRU已展开在独立的compact_gru.py中。
from .compact_gru import CompactGRU
