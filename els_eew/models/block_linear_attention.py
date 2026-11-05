# -*- coding: utf-8 -*-
"""BlockLinearAttention：因果核注意力，缓存KV累计矩阵与K累计向量。

依据 Katharopoulos et al., ICML2020；phi=ELU+1，非softmax注意力。
无长度增长的KV列表；单头8维key/24维value。详见对应模型文档。
完整前端/输出头/forward/stream_step在本文件展开；仅复用基础算子和无参数工具。
"""
import torch
from torch import nn
from torch.nn import functional as F
from .advanced_base import completed_block_features, check_state_tensor
from .students import ChannelLayerNorm
from .common import CausalConv

CONFIG = dict(width=24, dilations=(1, 2, 4, 8), key_dim=8, value_dim=24,
              mechanism="causal_kernel_attention", recurrent_rate="block_mean")


class CausalLinearAttentionHead(nn.Module):
    def __init__(self, width=24, key_dim=8):
        super().__init__()
        self.key_dim = key_dim
        self.norm1 = nn.LayerNorm(width)
        self.query = nn.Linear(width, key_dim)
        self.key = nn.Linear(width, key_dim)
        self.value = nn.Linear(width, width)
        self.out = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                 nn.Dropout(.1), nn.Linear(2 * width, width))
        self.proj = nn.Linear(width, 4)

    def forward(self, features, wave, state=None):
        x = completed_block_features(features)
        normalized = self.norm1(x)  # 仅通道归一化，不跨块统计
        q = F.elu(self.query(normalized)) + 1
        k = F.elu(self.key(normalized)) + 1
        v = self.value(normalized)
        b, _, width = x.shape
        if state is None:
            matrix = x.new_zeros(b, self.key_dim, width)
            normalizer = x.new_zeros(b, self.key_dim)
        else:
            matrix, normalizer = state
        check_state_tensor(matrix, x, (b, self.key_dim, width))
        check_state_tensor(normalizer, x, (b, self.key_dim))
        outputs = []
        for t in range(x.size(1)):
            # 当前块也属于已知历史；不读取任何后续块。
            matrix = matrix + k[:, t].unsqueeze(-1) * v[:, t].unsqueeze(1)
            normalizer = normalizer + k[:, t]
            numerator = torch.einsum('bd,bdv->bv', q[:, t], matrix)
            denominator = (q[:, t] * normalizer).sum(-1, keepdim=True).clamp_min(1e-6)
            outputs.append(numerator / denominator)
        x = x + self.out(torch.stack(outputs, dim=1))
        x = x + self.ffn(self.norm2(x))
        return self.proj(x), (matrix, normalizer)


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


class BlockLinearAttention(nn.Module):
    """完整模型：前端、输出头和状态处理均在本文件展开。"""
    chunk_size = 32

    def __init__(self):
        super().__init__()
        self.architecture_config = dict(CONFIG)
        self.feature_extractor = WaveformFrontend(CONFIG['width'], CONFIG['dilations'])
        self.output_head = CausalLinearAttentionHead(CONFIG['width'], CONFIG['key_dim'])
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
