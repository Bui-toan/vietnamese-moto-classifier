"""
Module tiện ích phục vụ huấn luyện và xuất gói báo cáo (src/train/common.py)
Tuân thủ nghiêm ngặt mục 5.2 của kế hoạch v4:
- config.json: Siêu tham số, seed, thứ tự lớp.
- env.json: Phiên bản python/torch/numpy/opencv, thiết bị, git commit.
- history.csv: epoch, lr, train_loss, train_acc, val_loss, val_acc, val_top3, val_macro_f1, grad_norm, epoch_time_s.
- learning_curve.png: Đường loss, accuracy, macro-F1.
- complexity.json: Số tham số, kích thước MB, thời gian train, bộ nhớ đỉnh.
- val_metrics.json: Top-1/2/3, macro-F1, per-class P/R/F1, NLL, ECE, T.
- model_card.json: SHA-256 trọng số, stats, config; cờ frozen.
"""

import os
import sys
import json
import hashlib
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_sha256(filepath: str) -> str:
    """Tính SHA-256 của file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    """Lấy commit git hiện tại."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def generate_env_json(filepath: str, device: str = "cpu"):
    """Xuất thông tin môi trường env.json."""
    import torch
    import cv2

    env_info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "device": device,
        "git_commit": get_git_commit()
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2, ensure_ascii=False)


def plot_learning_curves(history_df, output_path: str):
    """Vẽ biểu đồ learning_curve.png gồm loss, accuracy và macro-F1."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Loss
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", color="#EF4444")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", color="#3B82F6")
    axes[0].set_title("Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Accuracy
    axes[1].plot(history_df["epoch"], history_df["train_acc"], label="Train Acc", color="#10B981")
    axes[1].plot(history_df["epoch"], history_df["val_acc"], label="Val Acc", color="#6366F1")
    if "val_top3" in history_df.columns:
        axes[1].plot(history_df["epoch"], history_df["val_top3"], label="Val Top-3", color="#F59E0B", linestyle="--")
    axes[1].set_title("Accuracy vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Macro-F1
    axes[2].plot(history_df["epoch"], history_df["val_macro_f1"], label="Val Macro-F1", color="#14B8A6")
    axes[2].set_title("Macro-F1 vs Epoch")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_model_card(
    card_path: str,
    model_name: str,
    weights_path: str,
    stats_path: str,
    config_path: str,
    frozen: bool = True
):
    """Xuất model_card.json với SHA-256 các file."""
    card = {
        "model_name": model_name,
        "frozen": frozen,
        "weights": {
            "path": weights_path,
            "sha256": compute_sha256(weights_path) if os.path.exists(weights_path) else None
        },
        "stats": {
            "path": stats_path,
            "sha256": compute_sha256(stats_path) if os.path.exists(stats_path) else None
        },
        "config": {
            "path": config_path,
            "sha256": compute_sha256(config_path) if os.path.exists(config_path) else None
        }
    }
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, ensure_ascii=False)
