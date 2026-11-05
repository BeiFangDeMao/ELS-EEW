"""纯标准库模型名单：配置与批量入口共用，导入时不加载torch。"""
ORIGINAL_MODELS = (
    "S4_stream_LN",
      "CompactGRU", "MobileGRU", "MultiScaleGRU",
    "BlockQRNN", "PhysicsGRU", "LongDS_TCN",
)
NEW_MODELS = (
    "PartialMixGRU",
    "ShuffleGroupGRU",
    "GatedDSGRU",
    "PrefixSEGRU",
    "DynamicKernelGRU",
    "LowRankGRU",
    "BlockSRU",
    "BlockIndRNN",
    "BlockLinearAttention",
    "SoftForestGRU",
)
ABLATION_MODELS = ("PureGRU",)
ALL_MODEL_IDS = ORIGINAL_MODELS + NEW_MODELS + ABLATION_MODELS
