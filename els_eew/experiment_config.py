# -*- coding: utf-8 -*-
"""IDE配置区：所有候选使用同一组实验设置。"""
from pathlib import Path
from model_catalog import ALL_MODEL_IDS

PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parent

# 本次仅按以下顺序训练五个模型，其余暂不训练。
MODELS_TO_RUN = ["CompactGRU"]
# 只训练第二轮十个候选时，运行run_new_models.py，无需修改此处。
SEEDS = [42]
TRAIN_H5 = WORKSPACE_DIR / "dataset/dataseth5/J_K_KIKnet_2976_Causal_100Hz_20261012_153132_099924/J_K_KIKnet_2976_Causal_100Hz.h5"
OUTPUT_ROOT = PROJECT_DIR / "outputs"
SEQUENCE_LENGTH = 2976
BATCH_SIZE = 512
# 训练前的H5读取进程数；主进程列名称，均分后读取，全部退出后按原顺序合并。
# 不影响训练DataLoader的num_workers；设为1可恢复串行，4/8可对照磁盘读取速度。
H5_READ_PROCESSES = 16
EPOCHS = 300
PATIENCE = 20
GRAD_CLIP = 0.8
EXCLUDE_200HZ = True
SKIP_FINAL_TEST = False
# 测试后只生成三张任务误差PNG和小体积指标文件，不自动导出Excel/逐样本CSV。
# 单个候选异常时记录失败并继续其他候选；Ctrl+C仍会停止整个实验。
CONTINUE_ON_ERROR = True
