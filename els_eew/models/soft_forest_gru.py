# -*- coding: utf-8 -*-
"""SoftForestGRU：四棵深度3的可微概率树替换最后的线性输出层。

依据 Deep Neural Decision Forests (ICCV2015)，本项目改成多任务回归叶值。
不是sklearn随机森林，不使用硬路由/随机bagging；没有额外树监督损失。
详见 docs/SoftForestGRU.md。
完整前端/输出头/forward/stream_step在本文件展开；仅复用基础算子和无参数工具。
"""
import torch
from torch import nn
from torch.nn import functional as F
from .students import ChannelLayerNorm
from .common import CausalConv

CONFIG = dict(width=16, hidden=24, dilations=(1, 2, 4, 8), trees=4, depth=3,
              mechanism="soft_regression_forest", recurrent_rate="sample")


class SoftRegressionForest(nn.Module):
    def __init__(self, hidden=24, trees=4, depth=3, outputs=4):
        super().__init__()
        self.trees, self.depth = trees, depth
        self.nodes_per_tree = 2 ** depth - 1
        self.router = nn.Linear(hidden, trees * self.nodes_per_tree)
        leaves = torch.randn(trees, 2 ** depth, outputs) * .05
        # 小正值避免原训练clamp把震级/距离初始梯度全截断；不引入目录标签先验。
        leaves[..., :2] = leaves[..., :2].abs() + .05
        self.leaves = nn.Parameter(leaves)

    def forward(self, x):
        b, blocks, _ = x.shape
        gates = self.router(x).reshape(b, blocks, self.trees, self.nodes_per_tree).sigmoid()
        probabilities = x.new_ones(b, blocks, self.trees, 1)
        for level in range(self.depth):
            offset = 2 ** level - 1
            count = 2 ** level
            split = gates[..., offset:offset + count]
            # 子节点按(left0,right0,left1,right1,...)排列，匹配下一层BFS索引。
            probabilities = torch.stack((probabilities * split,
                                          probabilities * (1 - split)), dim=-1).flatten(-2)
        per_tree = torch.einsum('bkfl,flo->bkfo', probabilities, self.leaves)
        return per_tree.mean(dim=2)


class ForestOutputHead(nn.Module):
    def __init__(self, width=16, hidden=24):
        super().__init__()
        self.gru = nn.GRU(width, hidden, 2, batch_first=True, dropout=.1)
        self.forest = SoftRegressionForest(hidden, CONFIG['trees'], CONFIG['depth'], 4)

    def forward(self, features, wave, state=None):
        hidden, stats = (None, None) if state is None else state
        output, hidden = self.gru(features.transpose(1, 2), hidden)
        return self.forest(output[:, 31::32]), (hidden, stats)


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


class SoftForestGRU(nn.Module):
    """完整模型：前端、输出头和状态处理均在本文件展开。"""
    chunk_size = 32

    def __init__(self):
        super().__init__()
        self.architecture_config = dict(CONFIG)
        self.feature_extractor = WaveformFrontend(CONFIG['width'], CONFIG['dilations'])
        self.output_head = ForestOutputHead(CONFIG['width'], CONFIG['hidden'])
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
