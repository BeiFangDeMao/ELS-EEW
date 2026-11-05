# -*- coding: utf-8 -*-
"""在IDE直接运行；按experiment_config逐个启动独立训练进程。"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import experiment_config as config
from model_catalog import ALL_MODEL_IDS
from tools.summarize_results import summarize

# 无需导入torch来显示候选列表或编排训练。
AVAILABLE_MODELS = ALL_MODEL_IDS


def run_models(models=None):
    models = list(config.MODELS_TO_RUN if models is None else models)
    if not models or len(models) != len(set(models)) or any(m not in AVAILABLE_MODELS for m in models):
        raise ValueError(f"模型列表为空、重复或无效；可选{AVAILABLE_MODELS}")
    if not config.SEEDS or len(config.SEEDS) != len(set(config.SEEDS)):
        raise ValueError("SEEDS不能为空或重复")
    for path in (config.TRAIN_H5,):
        if not Path(path).is_file():
            raise FileNotFoundError(f"请核对experiment_config中的H5路径：{path}")
    settings = dict(train_h5=str(config.TRAIN_H5),
                    sequence_length=config.SEQUENCE_LENGTH, batch_size=config.BATCH_SIZE,
                    epochs=config.EPOCHS, patience=config.PATIENCE, grad_clip=config.GRAD_CLIP,
                    exclude_200hz=config.EXCLUDE_200HZ, skip_final_test=config.SKIP_FINAL_TEST,
                    h5_read_processes=config.H5_READ_PROCESSES)
    if settings['sequence_length'] <= 0 or settings['sequence_length'] % 32:
        raise ValueError("输入长度须为正且能被32整除")
    if (not isinstance(settings['h5_read_processes'], int)
            or isinstance(settings['h5_read_processes'], bool) or settings['h5_read_processes'] < 1):
        raise ValueError("H5_READ_PROCESSES必须为正整数")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    campaign = Path(config.OUTPUT_ROOT) / f"experiment_{stamp}"
    campaign.mkdir(parents=True)
    manifest = dict(created=stamp, settings=settings, models=models, seeds=config.SEEDS, runs=[])
    manifest_path = campaign / "experiment_manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def refresh_summary():
        try:
            summarize(campaign, reexport_missing=False, export_excel=False)
            manifest.pop('summary_export_error', None)
        except Exception as exc:
            manifest['summary_export_error'] = f"{type(exc).__name__}: {exc}"
            print(f"结果已保留，但指标CSV汇总失败：{exc}", flush=True)
        save()

    save()
    refresh_summary()
    for model_id in models:
        for seed in config.SEEDS:
            run_dir = campaign / f"{model_id}_seed{seed}"
            run_dir.mkdir()
            payload = dict(settings, model_id=model_id, seed=seed, output_root=str(run_dir))
            settings_path = run_dir / "run_config.json"
            settings_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            item = dict(model_id=model_id, seed=seed, output_dir=str(run_dir), status="running")
            manifest['runs'].append(item)
            save()
            print(f"\n开始训练：{model_id} / seed={seed}\n结果目录：{run_dir}", flush=True)
            try:
                result = subprocess.run([sys.executable, "-u", str(config.PROJECT_DIR / "train_one.py"),
                                         "--config", str(settings_path)], cwd=str(config.PROJECT_DIR))
            except KeyboardInterrupt:
                item['status'] = 'interrupted'
                save()
                refresh_summary()
                raise
            item['exit_code'] = result.returncode
            item['status'] = 'completed' if result.returncode == 0 else 'failed'
            save()
            refresh_summary()
            if result.returncode and not config.CONTINUE_ON_ERROR:
                raise RuntimeError(f"{model_id}失败，详见{run_dir}")
    # 只汇总已完成测试的结果；不将失败训练伪装为有效模型结果。
    refresh_summary()
    print(f"实验结束；逐模型状态见：{manifest_path}", flush=True)
    return campaign


if __name__ == "__main__":
    run_models()
