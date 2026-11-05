# -*- coding: utf-8 -*-
"""完整 BlockQRNN：因果 DS 前端 + 两层块率 fo-QRNN + 三任务回归。

输入 (B,3,32*K)，输出 (B,K,4)：震级、距离、方向 sin/cos。
每次 stream_step 仅处理新增 32 点；调用者传递卷积缓存与 cell。
QRNN 来自 Bradbury et al., ICLR 2017；块平均是本项目适配。
使用逐块递推，不复现论文融合 CUDA 加速；详见 docs/BlockQRNN.md。
展开原 common.py 实现，保留原参数路径及结构。
"""
import torch
from torch import nn
from torch.nn import functional as F
from .students import ChannelLayerNorm

CONFIG = dict(width=24, hidden=0, kind="ds", dilations=(1, 2, 4, 8), mode="qrnn")


class CachedCausalConv(nn.Module):
    """缓存 d*(k-1) 个输入；记录开始时以零初始化历史。"""
    def __init__(self, ci, co, kernel=3, dilation=1, groups=1):
        super().__init__()
        self.history = (kernel - 1) * dilation
        self.conv = nn.Conv1d(ci, co, kernel, dilation=dilation, groups=groups)

    def forward(self, x, cache=None):
        if cache is None:
            cache = x.new_zeros(x.size(0), x.size(1), self.history)
        if (tuple(cache.shape) != (x.size(0), x.size(1), self.history)
                or cache.device != x.device or cache.dtype != x.dtype):
            raise ValueError("卷积缓存与输入不匹配，请重置记录状态")
        combined = torch.cat((cache, x), dim=-1)
        new_cache = combined[..., -self.history:].clone() if self.history else combined[..., :0]
        return self.conv(combined), new_cache


class DSResidualBlock(nn.Module):
    """depthwise(k=3) -> 逐时刻 LN -> GELU -> pointwise -> 残差。"""
    def __init__(self, width, dilation):
        super().__init__()
        self.convs = nn.ModuleList([CachedCausalConv(width, width, 3, dilation, width)])
        self.norms = nn.ModuleList([ChannelLayerNorm(width)])
        self.project = nn.Conv1d(width, width, 1)

    def forward(self, x, state=None):
        if state is not None and len(state) != 1:
            raise ValueError("DS 残差块需要一个卷积缓存")
        cache = None if state is None else state[0]
        y, cache = self.convs[0](x, cache)
        return x + self.project(F.gelu(self.norms[0](y))), (cache,)


class QRNNFrontend(nn.Module):
    """3→24 的输入投影及膨胀率 1/2/4/8 的四个残差块。"""
    def __init__(self, width=24, dilations=(1, 2, 4, 8)):
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


class FOPoolingQRNN(nn.Module):
    """每层保留一个块的卷积输入与一个 cell，门卷积宽度为 2。"""
    def __init__(self, width=24):
        super().__init__()
        self.gates = nn.ModuleList([CachedCausalConv(width, 3 * width, 2) for _ in range(2)])
        self.proj = nn.Linear(width, 4)

    def forward(self, features, state=None):
        b, c, length = features.shape
        # 只在该块全部到达后求均值，不产生块内中间时刻的预测。
        x = features.reshape(b, c, length // 32, 32).mean(dim=-1)
        states = [None, None] if state is None else state
        if len(states) != 2:
            raise ValueError("QRNN 需要两层状态")
        new_states = []
        for gate, old in zip(self.gates, states):
            conv_cache, cell = (None, None) if old is None else old
            values, conv_cache = gate(x, conv_cache)
            z, f, o = values.chunk(3, dim=1)
            z, f, o = z.tanh(), f.sigmoid(), o.sigmoid()
            if cell is None:
                cell = z.new_zeros(b, c)
            if tuple(cell.shape) != (b, c) or cell.device != x.device or cell.dtype != x.dtype:
                raise ValueError("QRNN cell 状态与输入不匹配")
            outputs = []
            for k in range(x.size(-1)):
                cell = f[..., k] * cell + (1 - f[..., k]) * z[..., k]
                outputs.append(o[..., k] * cell)
            x = torch.stack(outputs, dim=-1)
            new_states.append((conv_cache, cell))
        return self.proj(x.transpose(1, 2)), tuple(new_states)


class QRNNOutputHead(nn.Module):
    """保留原 output_head.qrnn 参数路径。"""
    def __init__(self, width=24):
        super().__init__()
        self.qrnn = FOPoolingQRNN(width)

    def forward(self, features, wave, state=None):
        return self.qrnn(features, state)


class BlockQRNN(nn.Module):
    """2976 点产生 93 个输出；状态由调用者管理，每条记录重新置空。"""
    chunk_size = 32

    def __init__(self):
        super().__init__()
        self.architecture_config = dict(CONFIG)
        self.feature_extractor = QRNNFrontend(CONFIG['width'], CONFIG['dilations'])
        self.output_head = QRNNOutputHead(CONFIG['width'])
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
        """(B,3,32) -> (B,1,4), 新状态；None 表示一条新记录。"""
        if self.training:
            raise RuntimeError("流式推理前请调用 model.eval()")
        if wave_chunk.ndim != 3 or wave_chunk.shape[1:] != (3, 32):
            raise ValueError("每次流式调用只接受新增的 32 点")
        reference = self.feature_extractor.input.weight
        if wave_chunk.device != reference.device or wave_chunk.dtype != reference.dtype:
            raise ValueError("输入的 device/dtype 必须与模型一致")
        predictions, _, new_state = self._compute(wave_chunk, state)
        return predictions, new_state
