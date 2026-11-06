"""Per-timestep normalization and compatibility factory for ELS-EEW."""
import torch
from torch import nn

class ChannelLayerNorm(nn.Module):
    """Normalize channels independently at each timestep: (B,C,L)."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-5)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)

MODEL_IDS = ["CompactGRU"]

def create_student_model(model_id="CompactGRU"):
    if model_id not in MODEL_IDS:
        raise ValueError(f"Expected CompactGRU, got {model_id}")
    from .compact_gru import CompactGRU
    return CompactGRU()
