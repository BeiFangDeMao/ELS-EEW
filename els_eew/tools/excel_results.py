"""Portable Excel exports for the Python training workflow (no torch import).

Metrics are saved evaluation snapshots, not editable/recalculated Excel models.
Large prediction tables are streamed and split at Excel's worksheet row limit.
"""
import csv
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path

EXCEL_MAX_ROWS = 1048576
TASKS = (("mag", "震级", "M"), ("dis", "震中距", "km"),
         ("azi_deg", "方位角", "deg"))
PREDICTION_HEADERS = ["样本序号", "事件ID", "输出块(从1开始)", "累计采样点",
                      "真实震级", "预测震级", "震级绝对误差(M)",
                      "真实震中距(km)", "预测震中距(km)", "距离绝对误差(km)",
                      "真实方位角(deg)", "预测方位角(deg)", "圆周角度绝对误差(deg)"]


def require_excel_dependency():
    try:
        import openpyxl
    except ImportError as exc:
        raise RuntimeError("Excel导出需要openpyxl，请先在训练解释器安装requirements.txt中的依赖") from exc
    return openpyxl


def _value(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


def _append(sheet, values, header=False):
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.styles import Alignment, Font, PatternFill
    cells = []
    for value in values:
        cell = WriteOnlyCell(sheet, value=_value(value))
        # IDs and metadata remain literal text, even if they start with '='.
        if isinstance(cell.value, str):
            cell.data_type = "s"
        if header:
            cell.font = Font(name="Calibri", bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="24476A")
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        elif isinstance(cell.value, float):
            cell.number_format = "0.0000"
        cells.append(cell)
    sheet.append(cells)


def add_table(workbook, name, headers, rows, max_rows=EXCEL_MAX_ROWS):
    """Stream rows once; every split sheet retains its own header and filter."""
    from openpyxl.utils import get_column_letter
    if not 2 <= max_rows <= EXCEL_MAX_ROWS:
        raise ValueError("max_rows must be between 2 and Excel's row limit")
    sheet = None
    count = 0
    part = 0

    def finish():
        if sheet is not None:
            sheet.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{count}"

    for row in rows:
        if sheet is None or count >= max_rows:
            finish()
            part += 1
            sheet = workbook.create_sheet(name if part == 1 else f"{name}_{part}")
            sheet.freeze_panes = "A2"
            sheet.row_dimensions[1].height = 36
            for i, title in enumerate(headers, 1):
                sheet.column_dimensions[get_column_letter(i)].width = min(32, max(18, len(title) * 1.6))
            _append(sheet, headers, header=True)
            count = 1
        _append(sheet, row)
        count += 1
    if sheet is None:
        sheet = workbook.create_sheet(name)
        sheet.freeze_panes = "A2"
        sheet.row_dimensions[1].height = 36
        for i, title in enumerate(headers, 1):
            sheet.column_dimensions[get_column_letter(i)].width = min(32, max(18, len(title) * 1.6))
        _append(sheet, headers, header=True)
        count = 1
    finish()


def save_workbook(workbook, destination):
    """Atomic replacement; an open Excel file gets a new timestamped export."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(suffix=".xlsx", dir=destination.parent)
    os.close(descriptor)
    try:
        workbook.save(temporary)
        try:
            os.replace(temporary, destination)
        except PermissionError:
            alternate = destination.with_name(
                f"{destination.stem}_{datetime.now():%Y%m%d_%H%M%S_%f}.xlsx")
            os.replace(temporary, alternate)
            print(f"Excel文件被占用，已另存：{alternate}", flush=True)
            destination = alternate
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def metric_rows(report):
    if not report:
        return
    for scope, metrics in (("全部输出点", report["overall"]),
                           ("最后输出点", report["last_chunk"]["metrics"])):
        for task, name, unit in TASKS:
            result = metrics.get(task, {})
            yield [scope, name, unit, "MAAE" if task == "azi_deg" else "MAE",
                   result.get("mae"), result.get("mse"), result.get("r2")]


def chunk_rows(report):
    if not report:
        return
    for index, block in enumerate(report["by_chunk"], 1):
        metrics = block["metrics"]
        yield [index, block["end_sample"], metrics["mag"].get("mae"),
               metrics["dis"].get("mae"), metrics["azi_deg"].get("mae"),
               metrics["mag"].get("mse"), metrics["dis"].get("mse"),
               metrics["mag"].get("r2"), metrics["dis"].get("r2")]


CHUNK_HEADERS = ["输出块(从1开始)", "累计采样点", "震级MAE(M)", "距离MAE(km)",
                 "方位角MAAE(deg)", "震级MSE(M²)", "距离MSE(km²)", "震级R²", "距离R²"]


def prediction_rows(keys, targets, predictions, num_chunks, chunk_size=32):
    if num_chunks < 1 or chunk_size < 1:
        raise ValueError("num_chunks and chunk_size must be positive")
    if len(keys) != len(targets) or len(keys) != len(predictions) or len(keys) % num_chunks:
        raise ValueError("Prediction rows must contain complete records with matching keys/targets")
    for i, (key, target, prediction) in enumerate(zip(keys, targets, predictions)):
        true_angle = math.degrees(math.atan2(float(target[2]), float(target[3]))) % 360
        pred_angle = math.degrees(math.atan2(float(prediction[2]), float(prediction[3]))) % 360
        difference = abs(true_angle - pred_angle)
        block = i % num_chunks + 1
        yield [i // num_chunks + 1, str(key), block, block * chunk_size,
               float(target[0]), float(prediction[0]), abs(float(prediction[0]) - float(target[0])),
               float(target[1]), float(prediction[1]), abs(float(prediction[1]) - float(target[1])),
               true_angle, pred_angle, min(difference, 360 - difference)]


def csv_prediction_rows(path, num_chunks, chunk_size=32):
    """Legacy CSV fallback for re-export, retaining its saved numeric precision."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for i, row in enumerate(csv.DictReader(stream)):
            block = int(row["Chunk索引"]) + 1
            target_mag, pred_mag = float(row["真实震级"]), float(row["预测震级"])
            target_dis, pred_dis = float(row["真实震中距"]), float(row["预测震中距"])
            target_angle, pred_angle = float(row["真实方位角(deg)"]), float(row["预测方位角(deg)"])
            difference = abs((target_angle - pred_angle) % 360)
            yield [i // num_chunks + 1, row["事件ID"], block, block * chunk_size,
                   target_mag, pred_mag, abs(target_mag - pred_mag),
                   target_dis, pred_dis, abs(target_dis - pred_dis),
                   target_angle, pred_angle, min(difference, 360 - difference)]


def export_model_excel(destination, report, metadata, history, predictions=(),
                       max_rows=EXCEL_MAX_ROWS):
    workbook = require_excel_dependency().Workbook(write_only=True)
    add_table(workbook, "测试总体指标", ["统计范围", "任务", "单位", "误差指标", "误差均值", "MSE", "R²"],
              metric_rows(report), max_rows)
    add_table(workbook, "逐输出点指标", CHUNK_HEADERS, chunk_rows(report), max_rows)
    names = list(history)
    history_labels = {
        'train_losses': '训练总损失', 'val_losses': '验证总损失',
        'train_loss_mag': '训练震级损失', 'val_loss_mag': '验证震级损失',
        'train_loss_dis': '训练距离损失', 'val_loss_dis': '验证距离损失',
        'train_loss_azi': '训练方向损失(sin/cos)', 'val_loss_azi': '验证方向损失(sin/cos)',
        'train_mae_mag': '训练震级MAE(M)', 'val_mae_mag': '验证震级MAE(M)',
        'train_mae_dis': '训练距离MAE(km)', 'val_mae_dis': '验证距离MAE(km)',
        'train_mae_azi': '训练方位角MAAE(deg)', 'val_mae_azi': '验证方位角MAAE(deg)',
    }
    length = max((len(history[name]) for name in names), default=0)
    add_table(workbook, "训练历史", ["Epoch"] + [history_labels.get(name, name) for name in names],
              ([i + 1] + [history[name][i] if i < len(history[name]) else None for name in names]
               for i in range(length)), max_rows)
    info = dict(metadata)
    info["角度误差定义"] = "min(|预测角-真实角|, 360-|预测角-真实角|)，角度归一化至[0,360)"
    info["总体指标范围"] = "测试集全部记录的全部输出点；最后输出点单独统计"
    info["空白数值含义"] = "未计算、不适用或非有限值；不代表零误差"
    info["历史来源"] = metadata.get("history_source", "本次训练全部已完成epoch；测试使用最佳checkpoint")
    add_table(workbook, "实验信息", ["字段", "值"], info.items(), max_rows)
    add_table(workbook, "逐样本预测", PREDICTION_HEADERS, predictions, max_rows)
    return save_workbook(workbook, destination)
