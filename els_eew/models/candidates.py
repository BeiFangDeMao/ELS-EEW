"""ELS-EEW training registry."""
from .compact_gru import CompactGRU

MODEL_REGISTRY = {"CompactGRU": CompactGRU}
MODEL_IDS = list(MODEL_REGISTRY)

def create_student_model(model_id="CompactGRU"):
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Expected CompactGRU, got {model_id}")
    return MODEL_REGISTRY[model_id]()
