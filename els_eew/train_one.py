"""Worker used by IDE launcher; each model trains in an independent process."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    from training.Trainstudents import main as train
    from models.candidates import MODEL_IDS
    if cfg['model_id'] not in MODEL_IDS:
        raise ValueError("Unregistered model")
    output = Path(cfg['output_root'])
    train(model_id=cfg['model_id'], train_h5_path=cfg['train_h5'],
          epochs=cfg['epochs'], batch_size=cfg['batch_size'], output_root=str(output),
          skip_final_test=cfg['skip_final_test'], patience=cfg['patience'], seed=cfg['seed'],
          model_family="explore", sequence_length=cfg['sequence_length'],
          grad_clip=cfg['grad_clip'], exclude_200hz=cfg['exclude_200hz'],
          h5_read_processes=cfg.get('h5_read_processes', 16))
    checkpoints = list(output.glob('checkpoint_*/best_*.pth'))
    metrics = list(output.glob('checkpoint_*/test_metrics_by_chunk.json'))
    if len(checkpoints) != 1:
        raise RuntimeError("Training returned without exactly one best checkpoint; inspect logs")
    if not cfg['skip_final_test'] and len(metrics) != 1:
        raise RuntimeError("Final test did not produce metrics; inspect logs")
    if not cfg['skip_final_test']:
        figures = list(output.glob('checkpoint_*/evaluation_*.png'))
        if len(figures) != 3:
            raise RuntimeError("Final test must produce exactly three task figures; inspect logs")


if __name__ == "__main__":
    main()
