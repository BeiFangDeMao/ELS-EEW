"""Unified registry: independent candidates and existing S4 LN control."""
from .students_streaming import StreamingS4
from .mobile_gru import MobileGRU
from .multiscale_gru import MultiScaleGRU
from .ds_tcn import LongDSTCN
from .block_qrnn import BlockQRNN
from .physics_gru import PhysicsGRU
from .compact_gru import CompactGRU
from .pure_gru import PureGRU
from .partial_mix_gru import PartialMixGRU
from .shuffle_group_gru import ShuffleGroupGRU
from .gated_ds_gru import GatedDSGRU
from .prefix_se_gru import PrefixSEGRU
from .dynamic_kernel_gru import DynamicKernelGRU
from .low_rank_gru import LowRankGRU
from .block_sru import BlockSRU
from .block_indrnn import BlockIndRNN
from .block_linear_attention import BlockLinearAttention
from .soft_forest_gru import SoftForestGRU

MODEL_REGISTRY = {
    "S4_stream_LN": StreamingS4,
    "CompactGRU": CompactGRU,
    "PureGRU": PureGRU,
    "MobileGRU": MobileGRU,
    "MultiScaleGRU": MultiScaleGRU,
    "BlockQRNN": BlockQRNN,
    "PhysicsGRU": PhysicsGRU,
    "LongDS_TCN": LongDSTCN,
    "PartialMixGRU": PartialMixGRU,
    "ShuffleGroupGRU": ShuffleGroupGRU,
    "GatedDSGRU": GatedDSGRU,
    "PrefixSEGRU": PrefixSEGRU,
    "DynamicKernelGRU": DynamicKernelGRU,
    "LowRankGRU": LowRankGRU,
    "BlockSRU": BlockSRU,
    "BlockIndRNN": BlockIndRNN,
    "BlockLinearAttention": BlockLinearAttention,
    "SoftForestGRU": SoftForestGRU,
}
MODEL_IDS = list(MODEL_REGISTRY)


def create_student_model(model_id):
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model_id}; choices: {MODEL_IDS}")
    return MODEL_REGISTRY[model_id]()
