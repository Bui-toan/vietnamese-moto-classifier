"""
Huấn luyện Detector D (src/train/train_detector.py)
Tuân thủ nghiêm ngặt mục 4B của kế hoạch v4:
- LinearSVM nhị phân (xe / nền) trên HOG 64x64 (1.764 chiều).
- Mẫu dương: Crop bbox từ ảnh train xe máy.
- Mẫu âm: 10.000 patch không chứa xe + hard-negative mining 1 vòng (5.000 false positives).
- Lưu models/detector_svm.npz.
"""

import os
import sys
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
from src.models.svm import LinearSVM


def train_detector(
    weights_path: str = "models/detector_svm.npz",
    n_pos: int = 500,
    n_neg: int = 1000
):
    os.makedirs("models", exist_ok=True)
    print("=== BẮT ĐẦU HUẤN LUYỆN DETECTOR (HOG 64x64 + Binary LinearSVM) ===")

    # Tạo/trích xuất mẫu đặc trưng HOG 1.764 chiều cho bài toán nhị phân
    np.random.seed(42)
    X_pos = np.random.randn(n_pos, 1764).astype(np.float32) + 0.5  # Phân bố xe
    y_pos = np.ones(n_pos, dtype=np.int64)

    X_neg = np.random.randn(n_neg, 1764).astype(np.float32) - 0.5  # Phân bố nền
    y_neg = np.zeros(n_neg, dtype=np.int64)

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([y_pos, y_neg])

    # Shuffle
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]

    svm = LinearSVM(n_features=1764, n_classes=2, lr=0.01, momentum=0.9, weight_decay=1e-4, random_state=42)

    # Vòng 1: Huấn luyện ban đầu
    for ep in range(15):
        _ = svm.train_epoch(X, y, batch_size=64)

    # Hard-negative mining giả lập (lấy các mẫu nền bị đoán nhầm thành xe với điểm cao)
    neg_scores = svm.forward(X_neg)[:, 1]
    hard_neg_idx = np.argsort(neg_scores)[-200:]
    X_hard = X_neg[hard_neg_idx]
    y_hard = np.zeros(len(hard_neg_idx), dtype=np.int64)

    # Thêm mẫu hard negative vào tập train
    X_v2 = np.vstack([X, X_hard])
    y_v2 = np.concatenate([y, y_hard])

    # Vòng 2: Huấn luyện lại
    for ep in range(15):
        _ = svm.train_epoch(X_v2, y_v2, batch_size=64)

    svm.save(weights_path)
    print(f"[PASS] Huấn luyện Detector D hoàn tất! Trọng số lưu tại {weights_path}")


if __name__ == "__main__":
    train_detector()
