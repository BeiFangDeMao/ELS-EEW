# -*- coding: utf-8 -*-
"""完整S4编号流式LN对照：因果DS卷积前端 + 两层GRU32。

本文件展开原students.py中S规模前端和GRU输出头，便于直接审阅。
S4是本项目模型编号，不表示Structured State Space序列模型。
保留原参数路径、逐时刻LN与每32点输出；2976点对应93次输出。
"""
from dataclasses import dataclass
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F
from .students import ChannelLayerNorm

CONFIG = dict(channel_dims=(16, 16, 24, 32, 32), dilations=(1, 2, 4, 8),
              gru_hidden_dim=32, gru_num_layers=2, chunk_size=32, num_task=4,
              normalization="per_timestep_channel_layernorm")


class S4CausalDSConv(nn.Module):
    """因果核3 depthwise+pointwise；离线左补零，在线使用显式历史缓存。"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = dilation * (kernel_size - 1)
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   dilation=dilation, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(F.pad(x, (self.padding, 0))))


class S4Frontend(nn.Module):
    """3→16，四块16→16→24→32→32，三处逐时刻LN/GELU，全局残差。"""
    def __init__(self):
        super().__init__()
        self.init_conv = nn.Conv1d(3, 16, 1)
        self.residual_from = 0
        self.blocks = nn.ModuleList()
        # bns为兼容旧checkpoint的名字，实际全部是ChannelLayerNorm。
        self.bns = nn.ModuleList()
        channels, dilations = CONFIG['channel_dims'], CONFIG['dilations']
        for i, dilation in enumerate(dilations):
            self.blocks.append(S4CausalDSConv(channels[i], channels[i+1], 3, dilation))
            if i < len(dilations) - 1:
                self.bns.append(ChannelLayerNorm(channels[i+1]))
        self.activation = nn.GELU()
        self.residual_conv = nn.Conv1d(16, 32, 1)

    def forward(self, wave):
        out = self.init_conv(wave)
        identity = self.residual_conv(out)
        for i, block in enumerate(self.blocks):
            out = block(out)
            if i < len(self.bns):
                out = self.activation(self.bns[i](out))
        return out + identity


class S4GRUHead(nn.Module):
    """两层GRU32逐采样点更新；输出次数由实际长度计算，没有1024点固定限制。"""
    def __init__(self):
        super().__init__()
        self.chunk_size = CONFIG['chunk_size']
        self.gru = nn.GRU(32, CONFIG['gru_hidden_dim'], CONFIG['gru_num_layers'],
                          batch_first=True, dropout=.1)
        self.proj = nn.Linear(CONFIG['gru_hidden_dim'], CONFIG['num_task'])

    def forward(self, features):
        output, _ = self.gru(features.transpose(1, 2))
        indices = torch.arange(self.chunk_size - 1, features.size(-1),
                               self.chunk_size, device=features.device)
        return self.proj(output[:, indices])


@dataclass(frozen=True)
class StreamState:
    conv_caches: Tuple[torch.Tensor, ...]
    gru_hidden: torch.Tensor


class StreamingS4(nn.Module):
    chunk_size = 32

    def __init__(self):
        super().__init__()
        self.architecture_config = dict(CONFIG)
        self.feature_extractor = S4Frontend()
        self.log_sigma_mag = nn.Parameter(torch.tensor(-1.))
        self.log_sigma_dis = nn.Parameter(torch.tensor(4.))
        self.log_sigma_azi = nn.Parameter(torch.tensor(-1.5))
        self.output_head = S4GRUHead()

    def forward(self, wave):
        """(B,3,32*K) -> (B,K,4), 卷积特征；每条记录从零历史开始。"""
        if wave.ndim != 3 or wave.size(1) != 3 or wave.size(-1) < 32 or wave.size(-1) % 32:
            raise ValueError("Expected (B,3,L), positive L divisible by 32")
        features = self.feature_extractor(wave)
        return self.output_head(features), features

    def initial_state(self, batch_size=1):
        """Create zero state on the model's device/dtype; reset per new record."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        reference = self.feature_extractor.init_conv.weight
        caches = tuple(reference.new_zeros(batch_size, block.depthwise.in_channels,
                                          block.padding)
                       for block in self.feature_extractor.blocks)
        hidden = reference.new_zeros(2, batch_size, 32)
        return StreamState(caches, hidden)

    @torch.no_grad()
    def stream_step(self, wave_chunk, state=None):
        """(B,3,32) -> prediction (B,1,4), new independent StreamState.

        Pass returned state into the next call. No state is stored in the model,
        so independent stations may use separate states with the same weights.
        GRU runs only the 32 new timesteps. Each convolution retains its input
        history of dilation*(kernel_size-1) points.
        """
        if self.training:
            raise RuntimeError("stream_step requires model.eval() (LayerNorm/dropout)")
        if wave_chunk.ndim != 3 or wave_chunk.shape[1:] != (3, self.chunk_size):
            raise ValueError("Expected wave_chunk shape (B, 3, 32)")
        reference = self.feature_extractor.init_conv.weight
        if wave_chunk.device != reference.device or wave_chunk.dtype != reference.dtype:
            raise ValueError("Input must match model device and dtype")
        batch = wave_chunk.shape[0]
        if state is None:
            state = self.initial_state(batch)
        if len(state.conv_caches) != len(self.feature_extractor.blocks):
            raise ValueError("Invalid number of convolution caches")
        if state.gru_hidden.shape != (2, batch, 32):
            raise ValueError("Invalid GRU hidden shape; reset state when batch changes")
        tensors = (*state.conv_caches, state.gru_hidden)
        if any(t.device != wave_chunk.device or t.dtype != wave_chunk.dtype for t in tensors):
            raise ValueError("State must match input device and dtype; reset after moving model")

        extractor = self.feature_extractor
        out = extractor.init_conv(wave_chunk)
        identity = extractor.residual_conv(out)
        caches = []
        for i, (block, cache) in enumerate(zip(extractor.blocks, state.conv_caches)):
            expected = (batch, block.depthwise.in_channels, block.padding)
            if tuple(cache.shape) != expected:
                raise ValueError(f"Cache {i} expected {expected}, got {tuple(cache.shape)}")
            combined = torch.cat((cache, out), dim=-1)
            # Clone to keep only the required history, not the combined storage.
            caches.append(combined[:, :, -block.padding:].clone())
            out = block.pointwise(block.depthwise(combined))
            if i < len(extractor.bns):
                out = extractor.activation(extractor.bns[i](out))
        features = out + identity
        output, hidden = self.output_head.gru(
            features.transpose(1, 2), state.gru_hidden)
        prediction = self.output_head.proj(output[:, -1:, :])
        return prediction, StreamState(tuple(caches), hidden)


MODEL_IDS = ["S4_stream_LN"]


def create_student_model(model_id="S4_stream_LN"):
    if model_id not in MODEL_IDS:
        raise ValueError(f"Expected one of {MODEL_IDS}, got {model_id}")
    return StreamingS4()
