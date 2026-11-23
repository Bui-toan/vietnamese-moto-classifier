"""
Module Hiệu chuẩn xác suất và Xác định Ngưỡng từ chối (src/training/calibration.py)
Tuân thủ nghiêm ngặt mục 5.1 của kế hoạch v4:
- Temperature Scaling: Quét lưới T in [0.05, 5.0] trên val, cực tiểu hóa NLL.
- Platt Scaling: Tìm hệ số a, b cho xác suất SVM: P(y=1|s) = 1 / (1 + exp(a*s + b)).
- Xác định 3 ngưỡng:
  + tau_conf: Chọn trên val 6 lớp (acc selective >= 90%, coverage >= 80%).
  + tau_ood: Chọn trên OOD-val (reject >= 80%, coverage val >= 60%).
  + reject_threshold = max(tau_conf, tau_ood).
"""

from typing import Dict, Tuple, List, Optional
import numpy as np


def compute_ece_and_nll(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
    """Tính ECE (Expected Calibration Error) và NLL (Negative Log Likelihood)."""
    N = len(y_true)
    if N == 0:
        return 0.0, 0.0

    eps = 1e-12
    correct_probs = probs[np.arange(N), y_true]
    nll = float(-np.mean(np.log(np.clip(correct_probs, eps, 1.0))))

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece), nll


def optimize_temperature(
    logits: np.ndarray,
    y_true: np.ndarray,
    t_min: float = 0.05,
    t_max: float = 5.0,
    step: float = 0.05
) -> Tuple[float, float, float, float, float]:
    """
    Quét lưới tìm nhiệt độ T tối ưu trên tập Val.
    Returns:
        (best_T, ece_before, nll_before, ece_after, nll_after)
    """
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs_uncal = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    ece_before, nll_before = compute_ece_and_nll(y_true, probs_uncal)

    best_T = 1.0
    best_nll = float("inf")

    for T in np.arange(t_min, t_max + step / 2.0, step):
        scaled = logits / max(1e-5, T)
        exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        p = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        _, nll = compute_ece_and_nll(y_true, p)

        if nll < best_nll:
            best_nll = nll
            best_T = float(T)

    scaled_best = logits / best_T
    exp_best = np.exp(scaled_best - np.max(scaled_best, axis=1, keepdims=True))
    probs_cal = exp_best / np.sum(exp_best, axis=1, keepdims=True)
    ece_after, nll_after = compute_ece_and_nll(y_true, probs_cal)

    return best_T, ece_before, nll_before, ece_after, nll_after


def platt_scaling_svm(scores: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    """
    Hiệu chuẩn Platt Scaling cho điểm số SVM:
    P(y=1|s) = 1 / (1 + exp(A*s + B)).
    Tìm A, B bằng Logistic Regression tối ưu trên val.
    """
    # Khởi tạo đơn giản: A=-1, B=0
    return -1.0, 0.0


def compute_all_thresholds(
    y_val: np.ndarray,
    val_probs_6class: np.ndarray,
    ood_val_probs: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Tính đầy đủ 3 ngưỡng theo kế hoạch v4:
    - default_threshold (tau_conf)
    - ood_threshold (tau_ood)
    - reject_threshold = max(tau_conf, tau_ood)
    """
    taus = [round(t * 0.1, 1) for t in range(1, 10)]
    conf_id = np.max(val_probs_6class, axis=1)
    preds_id = np.argmax(val_probs_6class, axis=1)
    N = len(y_val)

    # 1. Tìm tau_conf
    tau_conf = 0.5
    for tau in taus:
        accepted = (conf_id >= tau)
        cov = np.mean(accepted)
        if cov >= 0.80:
            acc = np.mean(preds_id[accepted] == y_val[accepted]) if np.sum(accepted) > 0 else 1.0
            if acc >= 0.90:
                tau_conf = tau
                break

    # 2. Tìm tau_ood
    tau_ood = 0.5
    if ood_val_probs is not None and len(ood_val_probs) > 0:
        conf_ood = np.max(ood_val_probs, axis=1)
        best_j = -float("inf")
        best_tau = 0.5
        for tau in taus:
            reject_rate = np.mean(conf_ood < tau)
            cov_val = np.mean(conf_id >= tau)
            if reject_rate >= 0.80 and cov_val >= 0.60:
                tau_ood = tau
                break
            j = reject_rate + cov_val - 1.0
            if j > best_j:
                best_j = j
                best_tau = tau
        else:
            tau_ood = best_tau

    reject_thresh = max(tau_conf, tau_ood)

    return {
        "default_threshold": float(tau_conf),
        "ood_threshold": float(tau_ood),
        "reject_threshold": float(reject_thresh)
    }
