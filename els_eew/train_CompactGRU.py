"""IDE entry: train only CompactGRU, shared experiment settings."""
from run_all_models import run_models

if __name__ == "__main__":
    run_models(["CompactGRU"])
