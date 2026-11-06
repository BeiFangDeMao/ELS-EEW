#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多任务学生模型训练脚本（无蒸馏）
- 使用学生模型（来自 students.py）同时预测震级、震中距、方位角 (sin/cos)
- 数据加载/划分方式与 TrainM3-twotask.py 完全一致
- 学习率策略：CyclicLR（循环学习率）
- 早停策略：三项指标至少两项突破历史最佳
- 可视化：仅三张逐输出点测试误差曲线；不自动导出Excel或逐样本CSV
用法: python train_CompactGRU.py
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from tqdm import tqdm
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # Save figures without requiring a GUI/Tk installation.
import matplotlib.pyplot as plt
import argparse
import math

from models.students import create_student_model, MODEL_IDS

# ---------- 字体设置 ----------
plt.rcParams["font.family"] = ["STIXGeneral", "DejaVu Sans", "SimHei", "Microsoft YaHei"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 训练超参 --------------------------
FIXED_LENGTH = 1024
CHUNK_LEN = 32
NUM_CHUNKS = FIXED_LENGTH // CHUNK_LEN

BATCH_SIZE = 512
EPOCHS = 300
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 20      # 连续20轮验证震级MAE不下降则早停
EARLY_STOP_EPS = 1e-6

# CyclicLR 参数（backbone 和 head 分别设置）
BACKBONE_BASE_LR = 2e-4
BACKBONE_MAX_LR = 6e-4
HEAD_BASE_LR = 1e-3
HEAD_MAX_LR = 3e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------- 数据集（完全复用第二个脚本）--------------------------
class SeismicDataset(Dataset):
    def __init__(self, h5_file_path, max_samples=None, require_year=False,
                 exclude_200hz=False, read_processes=16):
        from training.h5_loading import load_h5_records
        self.data_list = []
        self.require_year = require_year
        records, self.load_stats = load_h5_records(
            h5_file_path, sequence_length=FIXED_LENGTH, read_processes=read_processes,
            max_samples=max_samples, exclude_200hz=exclude_200hz,
        )
        # All reader processes have already exited. Only the parent creates tensors.
        for wave, label, key in tqdm(records, desc="主进程整理训练数据"):
            wave_t = torch.from_numpy(wave)
            label_t = torch.from_numpy(label)
            if require_year:
                year_str = key[6:8] if len(key) >= 8 else "00"
                self.data_list.append((wave_t, label_t, key, year_str))
            else:
                self.data_list.append((wave_t, label_t, key))
        logging.info("加载 %s，有效样本数：%d", h5_file_path, len(self.data_list))
        if exclude_200hz:
            logging.info("采样率过滤：排除200Hz记录=%d，采样率缺失/无效记录=%d（未因采样率排除）",
                         self.load_stats['excluded_200hz'], self.load_stats['unknown_rate'])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

def collate_fn(batch):
    if len(batch[0]) == 4:
        waves, labels, keys, years = zip(*batch)
        years = list(years)
    else:
        waves, labels, keys = zip(*batch)
        years = None
    waves = torch.stack(waves, dim=0)
    labels = torch.stack(labels, dim=0)  # (B, 4)
    return waves, labels, keys, years

# -------------------------- 多任务损失函数（与第二个脚本一致）--------------------------
def multi_task_weighted_mse_loss(pred, target, log_sigma_mag, log_sigma_dis, log_sigma_azi, T=None):
    """
    pred: (B, T, 4)
    target: (B, 4)
    """
    B, T_actual, _ = pred.shape
    if T is None:
        T = T_actual
    assert T_actual == T
    if target.dim() == 3:
        target = target.squeeze(1)
    elif target.dim() > 2:
        target = target.view(B, -1)
        if target.size(1) != 4:
            target = target[:, :4]
    target_exp = target.unsqueeze(1).expand(-1, T, -1)  # (B, T, 4)

    # Four relative stages; exactly preserves the old weights for T=32.
    stage = (torch.arange(T, device=pred.device) * 4 // T).clamp_max(3)
    weights = pred.new_tensor([0.7, 0.8, 0.9, 1.0])[stage]
    weights = weights.unsqueeze(0).unsqueeze(-1)  # (1, T, 1)

    loss_mag_raw = ((pred[:, :, 0:1] - target_exp[:, :, 0:1]) ** 2 * weights).mean()
    loss_dis_raw = ((pred[:, :, 1:2] - target_exp[:, :, 1:2]) ** 2 * weights).mean()
    loss_azi_sin_raw = ((pred[:, :, 2:3] - target_exp[:, :, 2:3]) ** 2 * weights).mean()
    loss_azi_cos_raw = ((pred[:, :, 3:4] - target_exp[:, :, 3:4]) ** 2 * weights).mean()
    loss_azi_raw = (loss_azi_sin_raw + loss_azi_cos_raw) / 2.0

    sigma_mag = torch.exp(log_sigma_mag)
    sigma_dis = torch.exp(log_sigma_dis)
    sigma_azi = torch.exp(log_sigma_azi)

    loss_mag = loss_mag_raw / (2 * sigma_mag**2) + log_sigma_mag
    loss_dis = loss_dis_raw / (2 * sigma_dis**2) + log_sigma_dis
    loss_azi = loss_azi_raw / (2 * sigma_azi**2) + log_sigma_azi

    total_loss = loss_mag + loss_dis + loss_azi
    return total_loss, loss_mag, loss_dis, loss_azi, (sigma_mag, sigma_dis, sigma_azi)

# -------------------------- 评估指标 --------------------------
def compute_mae_nan_safe(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs(y_true[mask] - y_pred[mask]))

def compute_task_metrics(preds, targets):
    metrics = {}
    task_names = ['mag', 'dis', 'azi_sin', 'azi_cos']
    for i, name in enumerate(task_names):
        p = preds[:, i]
        t = targets[:, i]
        mae = compute_mae_nan_safe(t, p)
        mse = mean_squared_error(t, p) if not np.isnan(mae) else np.nan
        if i < 2:
            r2 = r2_score(t, p) if not np.isnan(mae) else np.nan
        else:
            r2 = np.nan
        metrics[name] = {'mae': mae, 'mse': mse, 'r2': r2}
    # 方位角角度误差
    azi_pred = np.arctan2(preds[:, 2], preds[:, 3]) * 180 / np.pi
    azi_true = np.arctan2(targets[:, 2], targets[:, 3]) * 180 / np.pi
    azi_pred = (azi_pred + 360) % 360
    azi_true = (azi_true + 360) % 360
    diff = np.abs(azi_pred - azi_true)
    diff = np.minimum(diff, 360 - diff)
    azi_mae = np.nanmean(diff)
    metrics['azi_deg'] = {'mae': azi_mae}
    return metrics

def safe_clamp_preds(preds):
    """对多任务预测进行安全钳制，不修改原始张量（返回新张量）"""
    return torch.stack([
        torch.clamp(preds[:, :, 0], min=0.0, max=10.0),
        torch.clamp(preds[:, :, 1], min=0.0, max=200.0),
        torch.clamp(preds[:, :, 2], min=-1.0, max=1.0),
        torch.clamp(preds[:, :, 3], min=-1.0, max=1.0),
    ], dim=2)

# -------------------------- 训练/验证/测试函数 --------------------------
def require_finite(tensor, description):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite {description}; stopping before further updates")


def train_one_epoch(model, loader, optimizer, scheduler, epoch, grad_clip=1.0,
                    diagnostic_dir=None, model_id="CompactGRU"):
    model.train()
    total_loss = 0.0
    total_loss_mag = 0.0
    total_loss_dis = 0.0
    total_loss_azi = 0.0
    all_preds, all_labels = [], []
    successful_batches = 0
    skipped_batches = 0
    diagnostic_saved = False

    for batch_index, (wave, y, keys, _) in enumerate(tqdm(loader, desc=f"训练 Epoch {epoch}")):
        wave, y = wave.to(device), y.to(device)
        optimizer.zero_grad()

        # Small BN buffers and RNG state only; full batch/model saved on failure.
        before_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state(wave.device) if wave.is_cuda else None

        context = f"epoch={epoch} batch={batch_index} keys={keys[:3]}"
        try:
            logging.debug(context)
            require_finite(wave, f"input {context}")
            require_finite(y, f"labels {context}")
            preds, _ = model(wave)  # (B, T, 4)
            require_finite(preds, f"raw predictions {context}")
            preds = safe_clamp_preds(preds)

            loss, loss_mag, loss_dis, loss_azi, sigmas = multi_task_weighted_mse_loss(
                preds, y,
                model.log_sigma_mag,
                model.log_sigma_dis,
                model.log_sigma_azi,
                T=NUM_CHUNKS
            )
            uncertainty = [model.log_sigma_mag, model.log_sigma_dis, model.log_sigma_azi]
            diagnostic = (f"{context} log_sigma={[p.detach().item() for p in uncertainty]} "
                          f"lr={[g['lr'] for g in optimizer.param_groups]} "
                          f"wave_abs_max={wave.abs().max().item()}")
            require_finite(loss, f"loss {diagnostic}")
            loss.backward()
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip,
                                               error_if_nonfinite=True)
            except RuntimeError as exc:
                if "non-finite" not in str(exc):
                    raise
                if diagnostic_dir is not None and not diagnostic_saved:
                    from training.gradient_diagnostics import save_failure
                    save_failure(model, optimizer, scheduler, wave, y, keys, epoch,
                                 batch_index, diagnostic_dir, before_buffers,
                                 rng_cpu, rng_cuda, model_id)
                    diagnostic_saved = True
                raise FloatingPointError(f"Non-finite gradient {diagnostic}") from exc
        except FloatingPointError as exc:
            # No optimizer update has happened: discard gradients and BN changes.
            with torch.no_grad():
                for name, buffer in model.named_buffers():
                    buffer.copy_(before_buffers[name])
            optimizer.zero_grad(set_to_none=True)
            skipped_batches += 1
            logging.warning("跳过异常批次：%s；累计跳过批次数=%d；原因=%s", context, skipped_batches, exc)
            continue
        optimizer.step()
        parameter_flags = torch.stack([torch.isfinite(p).all() for p in model.parameters()])
        require_finite(parameter_flags.float().masked_fill(~parameter_flags, float("nan")),
                       f"parameters after update {diagnostic}")
        scheduler.step()
        successful_batches += 1

        total_loss += loss.item()
        total_loss_mag += loss_mag.item()
        total_loss_dis += loss_dis.item()
        total_loss_azi += loss_azi.item()

        p_flat = preds.detach().cpu().numpy().reshape(-1, 4)
        l_flat = y.detach().cpu().numpy().repeat(NUM_CHUNKS, axis=0)
        all_preds.append(p_flat)
        all_labels.append(l_flat)

    if successful_batches == 0:
        raise FloatingPointError(f"Epoch {epoch}: all batches skipped; no valid training updates")
    # CyclicLR逐批次更新；记录本轮结束后的当前学习率（下一次更新使用）。
    logging.info(
        "Epoch %d: 学习率 backbone=%.6e；head=%.6e；成功批次数=%d；跳过批次数=%d",
        epoch, optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"],
        successful_batches, skipped_batches,
    )
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    metrics = compute_task_metrics(all_preds, all_labels)

    avg_loss = total_loss / successful_batches
    avg_mag = total_loss_mag / successful_batches
    avg_dis = total_loss_dis / successful_batches
    avg_azi = total_loss_azi / successful_batches
    return avg_loss, metrics, (avg_mag, avg_dis, avg_azi), sigmas

def evaluate(model, loader, mode="验证"):
    model.eval()
    total_loss = 0.0
    total_loss_mag = 0.0
    total_loss_dis = 0.0
    total_loss_azi = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for wave, y, _, _ in tqdm(loader, desc=mode):
            wave, y = wave.to(device), y.to(device)
            preds, _ = model(wave)
            require_finite(preds, "validation raw predictions")
            preds = safe_clamp_preds(preds)

            loss, loss_mag, loss_dis, loss_azi, sigmas = multi_task_weighted_mse_loss(
                preds, y,
                model.log_sigma_mag,
                model.log_sigma_dis,
                model.log_sigma_azi,
                T=NUM_CHUNKS
            )
            require_finite(loss, "validation loss")
            total_loss += loss.item()
            total_loss_mag += loss_mag.item()
            total_loss_dis += loss_dis.item()
            total_loss_azi += loss_azi.item()

            p_flat = preds.cpu().numpy().reshape(-1, 4)
            l_flat = y.cpu().numpy().repeat(NUM_CHUNKS, axis=0)
            all_preds.append(p_flat)
            all_labels.append(l_flat)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    metrics = compute_task_metrics(all_preds, all_labels)

    avg_loss = total_loss / len(loader)
    avg_mag = total_loss_mag / len(loader)
    avg_dis = total_loss_dis / len(loader)
    avg_azi = total_loss_azi / len(loader)
    return avg_loss, metrics, (avg_mag, avg_dis, avg_azi), sigmas

def test_model(model, loader):
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for wave, y, keys, _ in tqdm(loader, desc="测试中"):
            wave, y = wave.to(device), y.to(device)
            preds, _ = model(wave)
            preds = torch.nan_to_num(preds, nan=0.0, posinf=10.0, neginf=0.0)
            preds = safe_clamp_preds(preds)

            p_flat = preds.cpu().numpy().reshape(-1, 4)
            y_flat = y.cpu().numpy().repeat(NUM_CHUNKS, axis=0)
            all_preds.append(p_flat)
            all_trues.append(y_flat)

    all_preds = np.vstack(all_preds)
    all_trues = np.vstack(all_trues)
    metrics = compute_task_metrics(all_preds, all_trues)
    logging.info(f"测试结果: Mag MAE={metrics['mag']['mae']:.6f}, Dis MAE={metrics['dis']['mae']:.6f}, Azi(deg) MAE={metrics['azi_deg']['mae']:.6f}")
    return all_trues, all_preds, metrics

# -------------------------- 三任务结果图 --------------------------
def plot_task_mae_curves(report, save_dir, timestamp, model_id):
    """Only three compact curves; reuse saved metrics instead of plotting every sample."""
    blocks = report["by_chunk"]
    if not blocks:
        raise ValueError("Cannot plot an empty evaluation report")
    x = np.arange(1, len(blocks) + 1)
    # Limit annotations to about 12 positions so 93 outputs remain readable.
    marked = np.unique(np.linspace(0, len(blocks) - 1, min(12, len(blocks)), dtype=int))
    tasks = (
        ("mag", "Magnitude", "MAE (M)", "震级", "#2864A0"),
        ("dis", "Epicentral distance", "MAE (km)", "震中距_km", "#C77826"),
        ("azi_deg", "Azimuth", "MAAE (deg)", "方位角_deg", "#41845B"),
    )
    for task, title, ylabel, filename, color in tasks:
        errors = np.asarray([block["metrics"][task]["mae"] for block in blocks], dtype=float)
        overall = report["overall"][task]["mae"]
        fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
        ax.plot(x, errors, color=color, linewidth=1.8, marker="o", markersize=2.8)
        for i in marked:
            ax.annotate(f"{errors[i]:.3f}", (x[i], errors[i]), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=8, color=color)
        ax.set_xticks(x[marked])
        ax.set_xlim(0, len(blocks) + 1)
        # Explicit headroom keeps numeric annotations clear of the title.
        ax.set_ylim(0, max(float(np.max(errors)) * 1.18, 1e-6))
        ax.set_xlabel("Output block (32 new samples per output)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{model_id} — {title}\nOverall: {overall:.3f} | Final: {errors[-1]:.3f}")
        ax.grid(alpha=.25)
        path = os.path.join(save_dir, f"evaluation_{filename}_{model_id}_{timestamp}.png")
        try:
            fig.savefig(path, dpi=200)
        finally:
            plt.close(fig)
        logging.info("任务误差曲线已保存：%s", path)

# -------------------------- 主函数 --------------------------
def main(
    model_id: str,
    train_h5_path: str,
    epochs: int = 300,
    batch_size: int = 512,
    output_root: str = None,
    skip_final_test: bool = False,
    patience: int = 20,
    seed: int = 42,
    model_family: str = "cnn",
    sequence_length: int = 1024,
    grad_clip: float = 1.0,
    exclude_200hz: bool = False,
    h5_read_processes: int = 16,
):
    if not math.isfinite(grad_clip) or grad_clip <= 0:
        raise ValueError("grad_clip must be finite and positive")
    if (not isinstance(h5_read_processes, int) or isinstance(h5_read_processes, bool)
            or h5_read_processes < 1):
        raise ValueError("h5_read_processes必须为正整数；1表示串行读取")
    global BATCH_SIZE, EPOCHS, EARLY_STOP_PATIENCE, FIXED_LENGTH, NUM_CHUNKS
    if sequence_length < CHUNK_LEN or sequence_length % CHUNK_LEN:
        raise ValueError("sequence_length must be a positive multiple of 32")
    if model_family == "mlp" and sequence_length != 1024:
        raise ValueError("C-series currently requires sequence_length=1024")
    FIXED_LENGTH, NUM_CHUNKS = sequence_length, sequence_length // CHUNK_LEN
    BATCH_SIZE, EPOCHS, EARLY_STOP_PATIENCE = batch_size, epochs, patience
    if min(batch_size, epochs, patience) < 1:
        raise ValueError("batch_size, epochs and patience must be positive")
    if model_family not in ("explore", "cnn"):
        raise ValueError("This project supports ELS-EEW / CompactGRU only")
    from models.candidates import create_student_model as model_factory
    def set_deterministic(seed=42):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    set_deterministic(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_root = os.path.abspath(output_root) if output_root else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(artifact_root, exist_ok=True)
    checkpoint_dir = os.path.join(artifact_root, f"checkpoint_{model_id}_{timestamp}")
    log_dir = os.path.join(artifact_root, f"log_{model_id}_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        force=True,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"train_{model_id}.log"), encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    logging.info("=" * 60)
    logging.info(f"多任务学生模型训练启动时间：{timestamp}")
    logging.info(f"模型ID：{model_id}")
    logging.info(f"设备：{device}")
    logging.info(f"输入点数：{FIXED_LENGTH}，输出块数：{NUM_CHUNKS}")
    logging.info("H5加载进程数：%d；读取进程退出后由主进程按原顺序整理", h5_read_processes)
    logging.info(f"全局梯度L2范数裁剪阈值：{grad_clip}")
    logging.info(f"训练集/验证集来源：{train_h5_path}")
    logging.info("=" * 60)

    # ---------- 创建学生模型 ----------
    logging.info(f"创建学生模型：{model_id}")
    student = model_factory(model_id).to(device)
    logging.info(f"学生模型参数量：{sum(p.numel() for p in student.parameters()):,}")

    # ---------- 数据加载 ----------
    logging.info("加载主数据集（解析年份）...")
    logging.info("临时排除200Hz记录：%s", exclude_200hz)
    dataset_main = SeismicDataset(train_h5_path, require_year=True, exclude_200hz=exclude_200hz,
                                  read_processes=h5_read_processes)
    if len(dataset_main) == 0:
        logging.error("主数据集无有效样本")
        return

    train_samples, val_samples, test_samples = [], [], []
    for item in dataset_main.data_list:
        _, _, _, year_str = item
        if year_str in ['19', '20', '21', '22']:
            val_samples.append(item)
        elif year_str in ['23', '24']:
            test_samples.append(item)
        else:
            train_samples.append(item)

    logging.info(f"训练样本数：{len(train_samples)}")
    logging.info(f"验证样本数：{len(val_samples)}")
    logging.info(f"测试样本数：{len(test_samples)} (含23,24年)")

    class SubDataset(Dataset):
        def __init__(self, samples): self.samples = samples
        def __len__(self): return len(self.samples)
        def __getitem__(self, idx): return self.samples[idx]

    train_loader = DataLoader(SubDataset(train_samples), BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(SubDataset(val_samples), BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(SubDataset(test_samples), BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    if not train_samples or not val_samples:
        raise ValueError("Training and validation splits must be non-empty; check H5 years")

    # ---------- 优化器与学习率调度 ----------
    # 学生模型中的参数分组：特征提取器（backbone）和输出头
    backbone_params = list(student.feature_extractor.parameters())
    head_params = list(student.output_head.parameters())
    # 添加自适应噪声参数（log_sigma）到head组或单独一组，这里简单归入head
    # 如果模型有 log_sigma 参数，也放入 head 组
    log_sigma_params = []
    for name, param in student.named_parameters():
        if 'log_sigma' in name:
            log_sigma_params.append(param)
    if log_sigma_params:
        head_params.extend(log_sigma_params)

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_BASE_LR},
        {"params": head_params, "lr": HEAD_BASE_LR},
    ], weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999), eps=1e-8)

    steps_per_epoch = len(train_loader)
    step_size_up = 10 * steps_per_epoch
    step_size_down = 10 * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=[BACKBONE_BASE_LR, HEAD_BASE_LR],
        max_lr=[BACKBONE_MAX_LR, HEAD_MAX_LR],
        step_size_up=step_size_up,
        step_size_down=step_size_down,
        mode='triangular2',
        cycle_momentum=False,
    )

    # ---------- 训练循环（早停策略：监控震级MAE，与第二个脚本类似但可简化）----------
    # ---------- 训练循环（早停策略：三项任务至少两项改善）----------
    best_mag_mae = float('inf')
    best_dis_mae = float('inf')
    best_azi_mae = float('inf')
    best_epoch = 0
    early_stop_counter = 0

    # 存储曲线数据
    train_losses, val_losses = [], []
    train_loss_mag, val_loss_mag = [], []
    train_loss_dis, val_loss_dis = [], []
    train_loss_azi, val_loss_azi = [], []
    train_mae_mag, val_mae_mag = [], []
    train_mae_dis, val_mae_dis = [], []
    train_mae_azi, val_mae_azi = [], []

    for epoch in range(1, EPOCHS + 1):
        logging.info(f"\n========== Epoch {epoch}/{EPOCHS} ==========")

        try:
            train_loss, train_metrics, (avg_mag, avg_dis, avg_azi), _ = train_one_epoch(
                student, train_loader, optimizer, scheduler, epoch, grad_clip=grad_clip,
                diagnostic_dir=checkpoint_dir, model_id=model_id
            )
        except (FloatingPointError, RuntimeError):
            logging.exception("训练异常，停止当前运行；已保存的最佳checkpoint不受影响")
            raise
        val_loss, val_metrics, (val_avg_mag, val_avg_dis, val_avg_azi), _ = evaluate(
            student, val_loader, mode="验证"
        )

        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_loss_mag.append(avg_mag)
        val_loss_mag.append(val_avg_mag)
        train_loss_dis.append(avg_dis)
        val_loss_dis.append(val_avg_dis)
        train_loss_azi.append(avg_azi)
        val_loss_azi.append(val_avg_azi)

        cur_mag_mae = val_metrics['mag']['mae']
        cur_dis_mae = val_metrics['dis']['mae']
        cur_azi_mae = val_metrics['azi_deg']['mae']

        train_mae_mag.append(train_metrics['mag']['mae'])
        val_mae_mag.append(cur_mag_mae)
        train_mae_dis.append(train_metrics['dis']['mae'])
        val_mae_dis.append(cur_dis_mae)
        train_mae_azi.append(train_metrics['azi_deg']['mae'])
        val_mae_azi.append(cur_azi_mae)

        # 判断三项任务是否改善
        mag_improved = cur_mag_mae < best_mag_mae - EARLY_STOP_EPS
        dis_improved = cur_dis_mae < best_dis_mae - EARLY_STOP_EPS
        azi_improved = cur_azi_mae < best_azi_mae - EARLY_STOP_EPS
        improved_count = sum([mag_improved, dis_improved, azi_improved])

        if improved_count >= 2:
            # 更新最佳指标（只更新改善的项）
            if mag_improved:
                best_mag_mae = cur_mag_mae
            if dis_improved:
                best_dis_mae = cur_dis_mae
            if azi_improved:
                best_azi_mae = cur_azi_mae
            best_epoch = epoch
            early_stop_counter = 0

            # 保存最佳监督训练模型
            torch.save({
                'epoch': epoch,
                'model_id': model_id,
                'model_family': model_family,
                'num_parameters': sum(p.numel() for p in student.parameters()),
                'architecture_config': getattr(student, 'architecture_config', None),
                'sequence_length': FIXED_LENGTH,
                'exclude_200hz': exclude_200hz,
                'h5_read_processes': h5_read_processes,
                'grad_clip': grad_clip,
                'num_chunks': NUM_CHUNKS,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mag_mae': cur_mag_mae,
                'val_dis_mae': cur_dis_mae,
                'val_azi_mae': cur_azi_mae,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_loss_mag': train_loss_mag,
                'val_loss_mag': val_loss_mag,
                'train_loss_dis': train_loss_dis,
                'val_loss_dis': val_loss_dis,
                'train_loss_azi': train_loss_azi,
                'val_loss_azi': val_loss_azi,
                'train_mae_mag': train_mae_mag,
                'val_mae_mag': val_mae_mag,
                'train_mae_dis': train_mae_dis,
                'val_mae_dis': val_mae_dis,
                'train_mae_azi': train_mae_azi,
                'val_mae_azi': val_mae_azi,
            }, os.path.join(checkpoint_dir, f"best_{model_id}_{timestamp}.pth"))

            # 同时保存训练历史（用于复查训练过程）
            with open(os.path.join(checkpoint_dir, f"training_history_{model_id}_{timestamp}.pkl"), 'wb') as f:
                pickle.dump({
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_loss_mag': train_loss_mag,
                    'val_loss_mag': val_loss_mag,
                    'train_loss_dis': train_loss_dis,
                    'val_loss_dis': val_loss_dis,
                    'train_loss_azi': train_loss_azi,
                    'val_loss_azi': val_loss_azi,
                    'train_mae_mag': train_mae_mag,
                    'val_mae_mag': val_mae_mag,
                    'train_mae_dis': train_mae_dis,
                    'val_mae_dis': val_mae_dis,
                    'train_mae_azi': train_mae_azi,
                    'val_mae_azi': val_mae_azi,
                    'best_epoch': best_epoch,
                    'best_mag_mae': best_mag_mae,
                    'best_dis_mae': best_dis_mae,
                    'best_azi_mae': best_azi_mae,
                }, f)

            logging.info(
                f"✅ 模型改进并保存 ✓ Epoch {epoch} - Mag MAE: {cur_mag_mae:.6f}, Dis: {cur_dis_mae:.6f}, Azi: {cur_azi_mae:.6f}")
        else:
            early_stop_counter += 1
            logging.info(
                f"未达到改善条件（当前改善任务数 {improved_count}），早停计数器: {early_stop_counter}/{EARLY_STOP_PATIENCE}")

        logging.info(
            f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f} | Val MAE: Mag={cur_mag_mae:.6f}, Dis={cur_dis_mae:.6f}, Azi(deg)={cur_azi_mae:.6f}")

        if early_stop_counter >= EARLY_STOP_PATIENCE:
            logging.info(f"早停触发，最优Epoch={best_epoch}")
            break

    if epoch == EPOCHS:
        logging.info(f"达到最大Epoch数 {EPOCHS}，训练结束")

    # ---------- 测试 ----------
    logging.info("\n加载最佳学生模型进行测试...")
    ckpt_path = os.path.join(checkpoint_dir, f"best_{model_id}_{timestamp}.pth")
    if not os.path.exists(ckpt_path):
        logging.error(f"最佳模型文件不存在：{ckpt_path}")
        return
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    student.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"加载最佳模型 (Epoch {checkpoint['epoch']}), Val Mag MAE={checkpoint['val_mag_mae']:.6f}")

    # 完整历史独立保存，避免最佳checkpoint仅包含截至最佳epoch的历史。
    import json
    history = dict(train_losses=train_losses, val_losses=val_losses,
                   train_loss_mag=train_loss_mag, val_loss_mag=val_loss_mag,
                   train_loss_dis=train_loss_dis, val_loss_dis=val_loss_dis,
                   train_loss_azi=train_loss_azi, val_loss_azi=val_loss_azi,
                   train_mae_mag=train_mae_mag, val_mae_mag=val_mae_mag,
                   train_mae_dis=train_mae_dis, val_mae_dis=val_mae_dis,
                   train_mae_azi=train_mae_azi, val_mae_azi=val_mae_azi)
    with open(os.path.join(checkpoint_dir, "training_history_full.json"), "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2,
                  default=lambda value: value.item())
    metadata = dict(model_id=model_id, model_family=model_family, seed=seed,
                    num_parameters=sum(p.numel() for p in student.parameters()),
                    best_epoch=checkpoint['epoch'], completed_epochs=len(train_losses),
                    best_val_mag_mae=checkpoint['val_mag_mae'],
                    best_val_dis_mae=checkpoint['val_dis_mae'],
                    best_val_azi_maae_deg=checkpoint['val_azi_mae'],
                    sequence_length=FIXED_LENGTH, chunk_size=CHUNK_LEN, num_chunks=NUM_CHUNKS,
                    batch_size=batch_size, epochs_requested=epochs, patience=patience,
                    grad_clip=grad_clip, exclude_200hz=exclude_200hz,
                    h5_read_processes=h5_read_processes,
                    train_h5=train_h5_path,
                    architecture_config=getattr(student, 'architecture_config', None),
                    metric_source="最佳模型测试输出计算的总体及逐输出点指标",
                    parameter_count_scope="含训练任务损失权重，与checkpoint统计一致")

    if skip_final_test:
        logging.info("跳过最终测试，保留训练权重。")
        metadata['test_status'] = 'test_skipped'
    elif len(test_loader) == 0:
        logging.warning("测试集为空，跳过测试")
        metadata['test_status'] = 'empty_test_set'
    else:
        trues, preds, test_metrics = test_model(student, test_loader)
        true_seq = trues.reshape(-1, NUM_CHUNKS, 4)
        pred_seq = preds.reshape(-1, NUM_CHUNKS, 4)
        by_chunk = [dict(end_sample=(i + 1) * CHUNK_LEN,
                         metrics=compute_task_metrics(pred_seq[:, i], true_seq[:, i]))
                    for i in range(NUM_CHUNKS)]
        metadata.update(test_status='completed', test_records=len(trues) // NUM_CHUNKS,
                        prediction_rows=len(trues))
        report = {"metadata": metadata, "sequence_length": FIXED_LENGTH, "overall": test_metrics,
                  "last_chunk": by_chunk[-1], "by_chunk": by_chunk}
        with open(os.path.join(checkpoint_dir, "test_metrics_by_chunk.json"), "w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2,
                      default=lambda value: value.item())
        # 不写逐样本Excel/CSV，也不生成训练总图，只保存三张任务误差曲线。
        plot_task_mae_curves(report, checkpoint_dir, timestamp, model_id)

    logging.info("=" * 50)
    logging.info(f"多任务学生模型 {model_id} 训练、验证、测试全部完成！")
    logging.info("=" * 50)

def cli(model_family="cnn", exclude_200hz=False):
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    model_ids = MODEL_IDS
    parser = argparse.ArgumentParser(description="多任务学生模型监督训练（无蒸馏）")
    parser.add_argument("--model_id", type=str, default="CompactGRU",
                        choices=model_ids,
                        help=f"学生模型ID，可选: {model_ids}")
    parser.add_argument("--train_h5", type=str,
                        default=str(root / "dataset/dataseth5/J_K_KIKnet_3000_Pretrain.h5"),
                        help="主数据集H5文件")
    parser.add_argument("--output_root", type=str, default=None,
                        help="checkpoint和日志的统一保存根目录")
    parser.add_argument("--skip_final_test", action="store_true",
                        help="跳过最终测试，仅保存训练权重")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--h5_read_processes", type=int, default=16,
                        help="训练前H5读取进程数，默认16；1为串行读取")
    parser.add_argument("--exclude_200hz", action="store_true", default=exclude_200hz,
                        help="在读取波形前排除采样率属性为200Hz的记录")
    parser.add_argument("--grad_clip", type=float, default=0.8,
                        help="全局梯度L2范数上限，必须为正数，默认0.8")
    parser.add_argument("--sequence_length", type=int,
                        default=2976,
                        help="输入采样点数，必须为32的倍数，默认2976")
    args = parser.parse_args()
    print(f"训练配置：family={model_family} model={args.model_id} "
          f"输入={args.sequence_length}点 输出={args.sequence_length // CHUNK_LEN}块 "
          f"grad_clip={args.grad_clip} 排除200Hz={args.exclude_200hz}", flush=True)
    main(
        model_id=args.model_id,
        train_h5_path=args.train_h5,
        output_root=args.output_root,
        skip_final_test=args.skip_final_test,
        epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
        seed=args.seed, model_family=model_family,
        sequence_length=args.sequence_length,
        grad_clip=args.grad_clip,
        exclude_200hz=args.exclude_200hz,
        h5_read_processes=args.h5_read_processes,
    )


if __name__ == "__main__":
    cli()
