# -*- coding: utf-8 -*-
"""
StudentModels.py
12个轻量化学生模型变体 (S1~S4, M1~M4, L1~L4)
结构统一：动态深度可分离卷积特征提取器 + GRU输出头
通过 model_id 工厂函数创建

修改说明：所有卷积层已改为因果卷积（左侧填充），确保严格因果推理。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 基础组件 ====================

class ChannelLayerNorm(nn.Module):
    """Normalize channels independently at each timestep: (B,C,L)."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-5)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class DepthwiseSeparableConv1d(nn.Module):
    """
    因果深度可分离卷积 (Causal Depthwise Separable Conv1d)
    每个输出点只依赖当前及左侧的点，不依赖未来。
    padding 参数被忽略，固定使用因果填充。
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=None):
        super().__init__()
        # 计算左侧填充量，使得卷积后长度不变
        self.padding = dilation * (kernel_size - 1)
        # 深度卷积：无填充，但会在forward中手动填充左侧
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=0, dilation=dilation, groups=in_channels
        )
        # 逐点卷积
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C, L)
        # 左侧填充 self.padding 个零
        x_pad = F.pad(x, (self.padding, 0))  # (B, C, L + padding)
        out = self.depthwise(x_pad)          # 输出长度 = L + padding - (kernel_size-1)*dilation = L
        out = self.pointwise(out)
        return out


class LightweightGRUHead(nn.Module):
    """
    轻量级 GRU 输出头
    输入: (B, 32, L)  -> 转置为 (B, L, 32)
    输出: (B, num_chunks, num_task)
    """
    def __init__(self, feature_dim=32, chunk_size=32, hidden_dim=16, num_layers=1, num_task=4):
        super().__init__()
        self.chunk_size = chunk_size
        self.num_chunks = 1024 // chunk_size

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else 0.1
        )
        self.proj = nn.Linear(hidden_dim, num_task)

    def forward(self, x):
        # x: (B, 32, L)
        x = x.permute(0, 2, 1)          # (B, L, 32)
        out, _ = self.gru(x)            # (B, L, hidden_dim)
        L = x.size(1)
        # 取每个 chunk 的最后一个点（索引 31, 63, 95 ...）
        indices = torch.arange(self.chunk_size - 1, L, self.chunk_size, device=x.device)
        sampled = out[:, indices, :]    # (B, num_chunks, hidden_dim)
        pred = self.proj(sampled)       # (B, num_chunks, num_task)
        return pred


# ==================== 动态特征提取器 (因果版本) ====================

class DynamicFeatureExtractor(nn.Module):
    """
    因果动态特征提取器，使用因果深度可分离卷积堆栈。
    配置与原始版本相同。
    """
    def __init__(self, in_channels=3, out_channels=32, config=None):
        super().__init__()
        if config is None:
            config = {
                'channel_dims': [16, 16, 24, 32, 32],
                'dilations': [1, 2, 4, 8],
                'residual_from': 0
            }

        # 1x1 卷积本身是因果的（无历史依赖）
        self.init_conv = nn.Conv1d(in_channels, 16, kernel_size=1)

        channel_dims = config['channel_dims']
        dilations = config['dilations']
        self.residual_from = config['residual_from']

        self.blocks = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(len(dilations)):
            in_ch = channel_dims[i]
            out_ch = channel_dims[i+1]
            dil = dilations[i]
            block = DepthwiseSeparableConv1d(in_ch, out_ch, kernel_size=3, dilation=dil)
            self.blocks.append(block)
            if i < len(dilations) - 1:
                self.bns.append(ChannelLayerNorm(out_ch))

        self.activation = nn.GELU()
        # 残差投影也是 1x1，因果
        self.residual_conv = nn.Conv1d(16, 32, kernel_size=1)

    def forward(self, x):
        out = self.init_conv(x)          # (B, 16, L)
        identity = self.residual_conv(out)  # (B, 32, L)

        for i, block in enumerate(self.blocks):
            out = block(out)
            if i < len(self.bns):
                out = self.activation(self.bns[i](out))

        out = out + identity             # 残差连接
        return out                       # (B, 32, L)


# ==================== 完整动态学生模型 ====================

class DynamicStudentModel(nn.Module):
    """
    完整学生模型，包含因果特征提取器和GRU输出头
    """
    def __init__(self, cnn_config, gru_hidden_dim, gru_num_layers, num_task=4):
        super().__init__()
        self.feature_extractor = DynamicFeatureExtractor(config=cnn_config)
        self.log_sigma_mag = nn.Parameter(torch.tensor(-1.0))
        self.log_sigma_dis = nn.Parameter(torch.tensor(4.0))
        self.log_sigma_azi = nn.Parameter(torch.tensor(-1.5))
        self.output_head = LightweightGRUHead(
            feature_dim=32,
            chunk_size=32,
            hidden_dim=gru_hidden_dim,
            num_layers=gru_num_layers,
            num_task=num_task
        )

    def forward(self, x):
        feat_map = self.feature_extractor(x)   # (B, 32, L)
        pred = self.output_head(feat_map)      # (B, 32, num_task)
        return pred, feat_map                   # 返回预测和特征图


# ==================== 12个模型配置注册 ====================

# ---------- CNN 规模定义 ----------
CNN_CONFIGS = {
    'S': {  # 4层，小
        'channel_dims': [16, 16, 24, 32, 32],
        'dilations': [1, 2, 4, 8],
        'residual_from': 0
    },
    'M': {  # 6层，中
        'channel_dims': [16, 16, 24, 32, 32, 32, 32],
        'dilations': [1, 2, 4, 8, 12, 16],
        'residual_from': 0
    },
    'L': {  # 8层，大
        'channel_dims': [16, 16, 24, 32, 32, 32, 32, 32, 32],
        'dilations': [1, 2, 4, 8, 12, 16, 20, 24],
        'residual_from': 0
    }
}

# ---------- GRU 配置 ----------
GRU_CONFIGS = {
    '1': {'hidden_dim': 16, 'num_layers': 1},
    '2': {'hidden_dim': 16, 'num_layers': 2},
    '3': {'hidden_dim': 32, 'num_layers': 1},
    '4': {'hidden_dim': 32, 'num_layers': 2},
}

# ---------- 模型ID映射 ----------
MODEL_IDS = [
    'S1', 'S2', 'S3', 'S4',
    'M1', 'M2', 'M3', 'M4',
    'L1', 'L2', 'L3', 'L4'
]

# ---------- 工厂函数 ----------
def create_student_model(model_id: str):
    """
    根据模型ID创建对应的学生模型实例
    model_id: 格式 "{CNN规模}{GRU编号}"，如 "S1", "M4", "L3"
    """
    if model_id not in MODEL_IDS:
        raise ValueError(f"未知模型ID: {model_id}，可选: {MODEL_IDS}")

    cnn_scale = model_id[0]      # 'S', 'M', 'L'
    gru_id = model_id[1]         # '1', '2', '3', '4'

    cnn_config = CNN_CONFIGS[cnn_scale]
    gru_params = GRU_CONFIGS[gru_id]

    model = DynamicStudentModel(
        cnn_config=cnn_config,
        gru_hidden_dim=gru_params['hidden_dim'],
        gru_num_layers=gru_params['num_layers']
    )
    return model


# ==================== 注册字典 ====================
MODEL_REGISTRY = {mid: create_student_model for mid in MODEL_IDS}


# ==================== 测试（打印参数量） ====================
if __name__ == "__main__":
    print("学生模型各模块参数量统计（因果版本）：")
    print("-" * 60)
    print(f"{'模型ID':<6} {'特征提取器':>12} {'输出头(GRU)':>14} {'额外(log_sigma)':>16} {'总计':>10}")
    print("-" * 60)

    for mid in MODEL_IDS:
        model = create_student_model(mid)

        params_feat = sum(p.numel() for p in model.feature_extractor.parameters())
        params_head = sum(p.numel() for p in model.output_head.parameters())
        params_extra = sum(p.numel() for p in model.parameters()) - params_feat - params_head
        total = params_feat + params_head + params_extra

        print(f"{mid:<6} {params_feat:>12,} {params_head:>14,} {params_extra:>16,} {total:>10,}")

    print("-" * 60)
    print("说明：所有卷积已改为因果卷积（左侧填充），确保严格因果推理。")
    print("      特征提取器 = DynamicFeatureExtractor（CNN部分）")
    print("      输出头     = LightweightGRUHead（GRU + 线性投影）")
    print("      额外参数   = 三个log_sigma（不确定性权重）")