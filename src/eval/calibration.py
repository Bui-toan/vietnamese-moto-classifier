"""
Module Temperature Scaling & Xác định Ngưỡng từ chối (src/eval/calibration.py)
Tuân thủ nghiêm ngặt mục 5.1 của kế hoạch v4:
- Tìm nhiệt độ T bằng lưới 0.05 - 5.0 trên tập Val, cực tiểu hóa NLL.
- Báo cáo ECE trước và sau calibration.
- Xác định tau_conf (default_threshold): accuracy >= 90% và coverage >= 80% trên val 6 lớp.
- Xác định tau_ood (ood_threshold): >= 80% ảnh OOD-val bị từ chối và coverage val >= 60%.
- reject_threshold = max(tau_conf, tau_ood).
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
from src.eval.metrics import compute_ece_and_nll, compute_coverage_accuracy


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
    # 1. Tính toán trước calibration (T = 1.0)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs_uncal = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    ece_before, nll_before = compute_ece_and_nll(y_true, probs_uncal)

    best_T = 1.0
    best_nll = float("inf")

    # Quét lưới T
    for T in np.arange(t_min, t_max + step / 2.0, step):
        scaled = logits / max(1e-5, T)
        exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        p = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        _, nll = compute_ece_and_nll(y_true, p)

        if nll < best_nll:
            best_nll = nll
            best_T = float(T)

    # Tính toán sau calibration
    scaled_best = logits / best_T
    exp_best = np.exp(scaled_best - np.max(scaled_best, axis=1, keepdims=True))
    probs_cal = exp_best / np.sum(exp_best, axis=1, keepdims=True)
    ece_after, nll_after = compute_ece_and_nll(y_true, probs_cal)

    return best_T, ece_before, nll_before, ece_after, nll_after


def find_tau_conf(y_val: np.ndarray, val_probs: np.ndarray) -> float:
    """
    Ngưỡng tau_conf (default_threshold):
    Giá trị nhỏ nhất trong {0.1, 0.2, ..., 0.9} thỏa:
    - accuracy trên các ảnh được chấp nhận (conf >= tau) >= 90%
    - coverage >= 80% trên val 6 lớp.
    Nếu không có giá trị nào thỏa thì dùng 0.5 và cảnh báo.
    """
    taus = [round(t * 0.1, 1) for t in range(1, 10)]
    curve = compute_coverage_accuracy(y_val, val_probs, taus=taus)

    for item in curve:
        if item["accuracy"] >= 0.90 and item["coverage"] >= 0.80:
            return float(item["tau"])

    print("[WARN] Không có ngưỡng tau_conf nào thỏa (accuracy >= 90% và coverage >= 80%). Mặc định dùng 0.5.")
    return 0.5


def find_tau_ood(
    val_probs_6class: np.ndarray,
    ood_val_probs: np.ndarray
) -> float:
    """
    Ngưỡng tau_ood (ood_threshold):
    Chọn trên OOD-val, là giá trị nhỏ nhất trong {0.1, ..., 0.9} thỏa:
    - >= 80% ảnh OOD-val bị từ chối (conf < tau).
    - coverage của val 6 lớp vẫn >= 60%.
    Nếu không có giá trị nào thỏa thì chọn tau cực đại hóa:
      J = (tỉ lệ OOD-val bị từ chối) + (coverage val 6 lớp) - 1
    """
    taus = [round(t * 0.1, 1) for t in range(1, 10)]
    conf_id = np.max(val_probs_6class, axis=1)
    conf_ood = np.max(ood_val_probs, axis=1) if len(ood_val_probs) > 0 else np.array([])

    if len(conf_ood) == 0:
        return 0.5

    best_tau = None
    best_j = -float("inf")
    best_j_tau = 0.5

    for tau in taus:
        reject_rate_ood = float(np.mean(conf_ood < tau))
        coverage_id = float(np.mean(conf_id >= tau))

        if reject_rate_ood >= 0.80 and coverage_id >= 0.60:
            best_tau = tau
            break

        j = reject_rate_ood + coverage_id - 1.0
        if j > best_j:
            best_j = j
            best_j_tau = tau

    if best_tau is not None:
        return float(best_tau)

    print(f"[WARN] Không có tau_ood nào thỏa (reject >= 80% và coverage >= 60%). Cực đại hóa Youden J -> chọn {best_j_tau}.")
    return float(best_j_tau)


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
    tau_conf = find_tau_conf(y_val, val_probs_6class)

    if ood_val_probs is not None and len(ood_val_probs) > 0:
        tau_ood = find_tau_ood(val_probs_6class, ood_val_probs)
    else:
        tau_ood = tau_conf

    reject_thresh = max(tau_conf, tau_ood)

    return {
        "default_threshold": float(tau_conf),
        "ood_threshold": float(tau_ood),
        "reject_threshold": float(reject_thresh)
    }
