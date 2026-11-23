"""
Module chuẩn hóa dữ liệu Z-Score (src/preprocess/normalize.py)
Tính mean/std cho HOG và kênh ảnh RGB, lưu và nạp file stats .npz.
"""

from typing import Tuple, Optional
import os
import numpy as np


def compute_stats_zscore(data: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Tính mean và std theo trục axis."""
    mean = np.mean(data, axis=axis).astype(np.float32)
    std = np.std(data, axis=axis).astype(np.float32)
    std[std == 0] = 1.0  # Chống chia cho 0
    return mean, std


def apply_zscore(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-6
) -> np.ndarray:
    """Áp dụng chuẩn hóa Z-score: (x - mean) / (std + eps)."""
    return (x - mean) / (std + eps)


def save_stats(filepath: str, mean: np.ndarray, std: np.ndarray):
    """Lưu mean và std ra file .npz."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(filepath, mean=mean, std=std)


def load_stats(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Nạp mean và std từ file .npz."""
    data = np.load(filepath)
    return data["mean"], data["std"]
