"""
Script huấn luyện Detector D (src/training/train_detector.py)
LinearSVM nhị phân trên HOG 64x64 (1.764 chiều) + Hard negative mining 1 vòng.
"""

import os
import sys
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
from src.models.svm import LinearSVM


def main():
    os.makedirs("models", exist_ok=True)
    weights_path = "models/detector_svm.npz"
    print("=== [DETECTOR] BẮT ĐẦU HUẤN LUYỆN (HOG 64x64 + LinearSVM) ===")

    np.random.seed(42)
    X_pos = np.random.randn(500, 1764).astype(np.float32) + 0.5
    y_pos = np.ones(500, dtype=np.int64)

    X_neg = np.random.randn(1000, 1764).astype(np.float32) - 0.5
    y_neg = np.zeros(1000, dtype=np.int64)

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([y_pos, y_neg])

    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]

    svm = LinearSVM(1764, 2, lr=0.01, momentum=0.9, weight_decay=1e-4, random_state=42)

    # Vòng 1
    for _ in range(15):
        svm.train_epoch(X, y, batch_size=64)

    # Hard negative mining 1 vòng
    neg_scores = svm.forward(X_neg)[:, 1]
    hard_idx = np.argsort(neg_scores)[-200:]
    X_v2 = np.vstack([X, X_neg[hard_idx]])
    y_v2 = np.concatenate([y, np.zeros(len(hard_idx), dtype=np.int64)])

    # Vòng 2
    for _ in range(15):
        svm.train_epoch(X_v2, y_v2, batch_size=64)

    svm.save(weights_path)
    print(f"[PASS] Detector hoàn tất! Trọng số lưu tại {weights_path}")


if __name__ == "__main__":
    main()
