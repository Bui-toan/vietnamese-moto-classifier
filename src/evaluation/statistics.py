"""
Module phân tích thống kê suy luận (src/evaluation/statistics.py)
Tuân thủ nghiêm ngặt mục 6.4 của kế hoạch v4:
- bootstrap_ci: Bootstrap 1.000 lần (seed 42) tính khoảng tin cậy 95% (CI) cho Top-1 và Macro-F1.
- mcnemar_test_exact: Kiểm định McNemar chính xác bằng phân phối nhị thức với p=0.5.
- paired_bootstrap_delta_f1: So sánh chênh lệch Delta Macro-F1 kèm CI bootstrap cặp.
"""

from typing import Tuple, Optional
import numpy as np
import scipy.stats as stats

from src.evaluation.metrics import compute_per_class_metrics


def bootstrap_ci(
    y_true: np.ndarray,
    probs: np.ndarray,
    metric_type: str = "top1",
    n_bootstraps: int = 1000,
    seed: int = 42,
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    """
    Tính khoảng tin cậy 95% bằng Bootstrap 1.000 lần.
    Returns: (mean_val, lower_ci, upper_ci).
    """
    rng = np.random.RandomState(seed)
    N = len(y_true)
    if N == 0:
        return 0.0, 0.0, 0.0

    preds = np.argmax(probs, axis=1)
    boot_vals = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, N, size=N)
        b_true = y_true[idx]
        b_pred = preds[idx]

        if metric_type == "top1":
            val = float(np.mean(b_true == b_pred))
        elif metric_type == "macro_f1":
            res = compute_per_class_metrics(b_true, b_pred, num_classes=probs.shape[1])
            val = res["macro_f1"]
        else:
            raise ValueError(f"metric_type không hợp lệ: {metric_type}")
        boot_vals.append(val)

    lower = float(np.percentile(boot_vals, 100 * (alpha / 2.0)))
    upper = float(np.percentile(boot_vals, 100 * (1.0 - alpha / 2.0)))
    mean_val = float(np.mean(boot_vals))
    return mean_val, lower, upper


def mcnemar_test_exact(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray
) -> Tuple[int, int, float]:
    """
    Kiểm định McNemar chính xác (Exact Binomial Test) so sánh 2 model.
    Returns: (n10, n01, p_value).
    """
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    n10 = int(np.sum(correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))
    n = n10 + n01

    if n == 0:
        return 0, 0, 1.0

    k = min(n10, n01)
    p_value = float(2.0 * stats.binom.cdf(k, n, 0.5))
    p_value = min(1.0, p_value)
    return n10, n01, p_value


def paired_bootstrap_delta_f1(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    """
    Bootstrap cặp tính khoảng tin cậy cho chênh lệch Delta Macro-F1 = F1_b - F1_a.
    """
    rng = np.random.RandomState(seed)
    N = len(y_true)
    preds_a = np.argmax(probs_a, axis=1)
    preds_b = np.argmax(probs_b, axis=1)

    delta_vals = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, N, size=N)
        b_true = y_true[idx]
        f1_a = compute_per_class_metrics(b_true, preds_a[idx], num_classes=probs_a.shape[1])["macro_f1"]
        f1_b = compute_per_class_metrics(b_true, preds_b[idx], num_classes=probs_b.shape[1])["macro_f1"]
        delta_vals.append(f1_b - f1_a)

    lower = float(np.percentile(delta_vals, 100 * (alpha / 2.0)))
    upper = float(np.percentile(delta_vals, 100 * (1.0 - alpha / 2.0)))
    mean_val = float(np.mean(delta_vals))
    return mean_val, lower, upper
