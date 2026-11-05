# -*- coding: utf-8 -*-
"""BlockSRU：两层SRU，仅使用逐通道历史连接而非完整循环矩阵。

依据 Lei et al., EMNLP2018；完成块均值/因果DS前端是本项目适配。
顺序PyTorch版本，不使用论文的专用并行内核。详见 docs/BlockSRU.md。
完整前端/输出头/forward/stream_step在本文件展开；仅复用基础算子和无参数工具。
"""
import torch
from torch import nn
from torch.nn import functional as F
from .advanced_base import completed_block_features, check_state_tensor
from .students import ChannelLayerNorm
from .common import CausalConv

CONFIG = dict(width=24, hidden=24, dilations=(1, 2, 4, 8),
              mechanism="sru_diagonal_gates", recurrent_rate="block_mean")


class SRULayer(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.map = nn.Linear(width, 3 * width)
        self.v_forget = nn.Parameter(torch.zeros(width))
        self.v_reset = nn.Parameter(torch.zeros(width))

    def forward(self, x, cell=None):
        cell = torch.zeros_like(x[:, 0]) if cell is None else cell
        check_state_tensor(cell, x, x[:, 0].shape)
        value, forget, reset = self.map(x).chunk(3, -1)
        outputs = []
        for k in range(x.size(1)):
            # 两个门的对角历史连接都读取更新前cell，符合SRU公式。
            f = (forget[:, k] + self.v_forget * cell).sigmoid()
            r = (reset[:, k] + self.v_reset * cell).sigmoid()
            cell = f * cell + (1 - f) * value[:, k]
            outputs.append(r * cell.tanh() + (1 - r) * x[:, k])
        return torch.stack(outputs, dim=1), cell


class SRUHead(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.layers = nn.ModuleList([SRULayer(width) for _ in range(2)])
        self.dropout = nn.Dropout(.1)
        self.proj = nn.Linear(width, 4)

    def forward(self, features, wave, state=None):
        x = completed_block_features(features)
        states = [None, None] if state is None else state
        if len(states) != 2:
            raise ValueError("SRU requires two cells")
        new_states = []
        for i, (layer, cell) in enumerate(zip(self.layers, states)):
            x, cell = layer(x, cell)
            new_states.append(cell)
            if i == 0:
                x = self.dropout(x)
        return self.proj(x), tuple(new_states)


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


class BlockSRU(nn.Module):
    """完整模型：前端、输出头和状态处理均在本文件展开。"""
    chunk_size = 32

    def __init__(self):
        super().__init__()
        self.architecture_config = dict(CONFIG)
        self.feature_extractor = WaveformFrontend(CONFIG['width'], CONFIG['dilations'])
        self.output_head = SRUHead(CONFIG['width'])
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
