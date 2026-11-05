# -*- coding: utf-8 -*-
"""
CausalStudentModel.py (接口对齐版)
严格因果流式学生模型 （独立监督训练）
设计原则：
1. 物理切块 (1024 -> 32个独立的32点块)，彻底杜绝跨块信息泄露
2. 块内使用教师模型 (modelB) 的 ChunkEmbeddingHeadV2 风格编码 (1x1 Conv + MLP)
3. GRU 在块间传递隐藏状态，建模时序依赖
4. ★★★ 接口与 DynamicStudentModel (students.py) 完全一致:
   - 具有 feature_extractor 和 output_head 两个子模块
   - forward 返回 (pred, feat)
   - 工厂函数 create_student_model(model_id)
   - 模型ID列表 MODEL_IDS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 块编码器 (复用教师逻辑) ====================

class TeacherStyleChunkEncoder(nn.Module):
    """
    完全复用教师模型 (modelB) 的 ChunkEmbeddingHeadV2 核心逻辑
    专门用于处理单个 Chunk (B, 3, 32) -> 输出特征向量 (B, out_channels)
    严格物理隔离：只处理块内数据，绝不跨块
    """
    def __init__(self, in_channels=3, mid_channels=32, out_channels=32, chunk_size=32):
        super().__init__()
        self.chunk_size = chunk_size

        # 1. 1x1 卷积升维（通道混合）
        self.expand = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, mid_channels, kernel_size=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # 2. 双层 MLP 时间混合（块内全局感受野）
        # 注意：输入是 (B, C, 32)，我们将时间维展平后送入 MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(chunk_size, chunk_size * 2, bias=False),
            nn.GELU(),
            nn.Linear(chunk_size * 2, chunk_size, bias=False),
        )

        # 3. 将块内 32 个时间步压缩为 1 个向量（平均池化）
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 4. 最终投影（如果 mid_channels != out_channels）
        if mid_channels != out_channels:
            self.proj = nn.Linear(mid_channels, out_channels)
        else:
            self.proj = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        # 与教师模型保持一致的初始化
        for module in self.time_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)

    def forward(self, x):
        # x: (B, 3, 32)  -> 一批独立的 Chunk
        # 1. 通道混合
        x = self.expand(x)                      # (B, mid_channels, 32)

        # 2. 时间混合 (MLP)
        B, C, T = x.shape
        # 展平为 (B*C, T) 以便 MLP 处理每个通道的时间维度
        x = x.reshape(B * C, T)                 # (B*C, 32)
        x = self.time_mlp(x)                    # (B*C, 32)
        x = x.reshape(B, C, T)                  # (B, C, 32)

        # 3. 全局平均池化压缩时间维 -> (B, C, 1)
        x = self.pool(x).squeeze(-1)            # (B, C)

        # 4. 最终投影
        x = self.proj(x)                        # (B, out_channels)
        return x


# ==================== GRU 输出头 (封装为独立模块) ====================

class GRUOutputHead(nn.Module):
    """
    将 GRU 和线性输出投影封装为一个模块，方便训练脚本按 head 分组学习率
    """
    def __init__(self, input_size=32, hidden_dim=16, num_layers=1, num_task=4):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else 0.1
        )
        self.proj = nn.Linear(hidden_dim, num_task)

    def forward(self, x):
        # x: (B, num_chunks, 32)
        out, _ = self.gru(x)        # (B, num_chunks, hidden_dim)
        pred = self.proj(out)       # (B, num_chunks, num_task)
        return pred


# ==================== 完整因果学生模型 (接口对齐) ====================

class CausalStudentModel(nn.Module):
    """
    完整因果流式学生模型
    - 输入: (B, 3, 1024)
    - 输出: (B, 32, 4) 以及特征图 (B, 32, 32)
    - ★★★ 属性与 DynamicStudentModel 对齐: feature_extractor, output_head ★★★
    """
    def __init__(self,
                 chunk_size=32,
                 num_chunks=32,
                 gru_hidden_dim=16,
                 gru_num_layers=1,
                 num_task=4):
        super().__init__()
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks

        # ★★★ 特征提取器 (对齐命名) ★★★
        self.feature_extractor = TeacherStyleChunkEncoder(
            in_channels=3,
            mid_channels=32,
            out_channels=32,
            chunk_size=chunk_size
        )

        # ★★★ 输出头 (对齐命名) ★★★
        self.output_head = GRUOutputHead(
            input_size=32,
            hidden_dim=gru_hidden_dim,
            num_layers=gru_num_layers,
            num_task=num_task
        )

        # ★★★ 不确定性参数 （多任务损失用） ★★★
        self.log_sigma_mag = nn.Parameter(torch.tensor(-1.0))
        self.log_sigma_dis = nn.Parameter(torch.tensor(4.0))
        self.log_sigma_azi = nn.Parameter(torch.tensor(-1.5))

    def forward(self, x):
        # x: (B, 3, 1024)
        B, C, L = x.shape
        assert L == self.chunk_size * self.num_chunks, \
            f"输入长度 {L} 必须是 {self.chunk_size * self.num_chunks}"

        # ========== 1. 物理切块 ==========
        # (B, 3, 1024) -> (B*32, 3, 32)
        x = x.view(B, C, self.num_chunks, self.chunk_size)  # (B, 3, 32, 32)
        x = x.permute(0, 2, 1, 3)                           # (B, 32, 3, 32)
        x = x.reshape(B * self.num_chunks, C, self.chunk_size)  # (B*32, 3, 32)

        # ========== 2. 块内特征提取 ==========
        feat = self.feature_extractor(x)                    # (B*32, 32)

        # ========== 3. 恢复序列形状 ==========
        feat_seq = feat.view(B, self.num_chunks, -1)        # (B, 32, 32)

        # ========== 4. GRU 跨块建模 (通过 output_head) ==========
        pred = self.output_head(feat_seq)                   # (B, 32, num_task)

        # feat_seq 保留特征输出以兼容模型接口
        return pred, feat_seq


# ==================== 工厂函数与配置 ====================

# ★★★ 模型ID列表 (对齐训练脚本) ★★★
MODEL_IDS = ['C1', 'C2', 'C3', 'C4']

def create_student_model(model_id: str = "C1"):
    """
    根据模型ID创建因果学生模型
    支持: C1, C2, C3, C4 (对应不同的GRU规模)
    """
    configs = {
        'C1': {'hidden_dim': 16, 'num_layers': 1},
        'C2': {'hidden_dim': 16, 'num_layers': 2},
        'C3': {'hidden_dim': 32, 'num_layers': 1},
        'C4': {'hidden_dim': 32, 'num_layers': 2},
    }
    if model_id not in configs:
        raise ValueError(f"未知模型ID: {model_id}，可选: {list(configs.keys())}")

    params = configs[model_id]
    model = CausalStudentModel(
        chunk_size=32,
        num_chunks=32,
        gru_hidden_dim=params['hidden_dim'],
        gru_num_layers=params['num_layers'],
        num_task=4
    )
    return model


# # ==================== 参数统计测试 ====================
# if __name__ == "__main__":
#     print("=" * 60)
#     print("因果学生模型 (CausalStudentModel) 参数量统计")
#     print("=" * 60)
#     print(f"{'模型ID':<6} {'特征提取器':>12} {'输出头':>12} {'log_sigma':>12} {'总计':>12}")
#     print("-" * 72)
#
#     for mid in ['C1', 'C2', 'C3', 'C4']:
#         model = create_student_model(mid)
#
#         params_encoder = sum(p.numel() for p in model.feature_extractor.parameters())
#         params_head = sum(p.numel() for p in model.output_head.parameters())
#         params_extra = sum(p.numel() for p in [model.log_sigma_mag, model.log_sigma_dis, model.log_sigma_azi])
#         total = params_encoder + params_head + params_extra
#
#         print(f"{mid:<6} {params_encoder:>12,} {params_head:>12,} {params_extra:>12,} {total:>12,}")
#
#     print("-" * 72)
#     print("说明:")
#     print("  - 特征提取器: TeacherStyleChunkEncoder (1x1 Conv + MLP + Pooling)")
#     print("  - 输出头: GRU + 线性投影")
#     print("  - log_sigma: 三个不确定性参数 （多任务损失用）")
#     print("=" * 60)
#     print("\n使用方法 (与训练脚本完全兼容):")
#     print("  from students_mlp import create_student_model, MODEL_IDS")
#     print("  model = create_student_model('C1')")
#     print("  pred, feat = model(torch.randn(2, 3, 1024))")
#     print(f"  pred.shape: (2, 32, 4), feat.shape: (2, 32, 32)")
#     print("\n运行训练脚本时，请传入对应的模型ID:")
#     print("  python Trainstudents_mlp.py --model_id C1")