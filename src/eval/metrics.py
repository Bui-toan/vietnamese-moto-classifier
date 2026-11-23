"""
Thư viện đánh giá và tính toán chỉ số tự code 100% (src/eval/metrics.py)
Tuân thủ nghiêm ngặt mục 6.4 của kế hoạch v4:
- Top-1, Top-2, Top-3 Accuracy.
- Macro-F1, Weighted-F1, Precision, Recall, Support cho từng lớp.
- Ma trận nhầm lẫn 6x6 (số đếm và chuẩn hóa) kèm cột phụ "Từ chối".
- Khoảng tin cậy 95% bằng Bootstrap 1.000 lần (seed 42) cho Top-1 và Macro-F1.
- McNemar chính xác (tự code bằng exact binomial distribution) và Delta Macro-F1 kèm CI bootstrap cặp.
- ECE và NLL (trước và sau calibration).
- Đường cong Coverage-Accuracy.
- Tỉ lệ từ chối OOD và AUROC tự code.
- Đo Latency: Cold start và Warm start (p50, p95).
"""

import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import scipy.stats as stats


CLASS_NAMES = [
    "Honda Wave",
    "Honda Vision",
    "Honda Air Blade",
    "Yamaha Exciter",
    "Piaggio Vespa",
    "Honda SH"
]


def compute_accuracy_topk(
    y_true: np.ndarray,
    probs: np.ndarray,
    ks: Tuple[int, ...] = (1, 2, 3)
) -> Dict[str, float]:
    """Tính Top-k Accuracy (k=1, 2, 3)."""
    res = {}
    N = len(y_true)
    if N == 0:
        return {f"top{k}_acc": 0.0 for k in ks}

    for k in ks:
        # Lấy top k chỉ số có xác suất cao nhất
        topk_preds = np.argsort(probs, axis=1)[:, -k:]
        correct = np.any(topk_preds == y_true[:, np.newaxis], axis=1)
        res[f"top{k}_acc"] = float(np.mean(correct))
    return res


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 6,
    rejected: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tự code ma trận nhầm lẫn 6x6 kèm cột phụ 'Từ chối'.
    
    Returns:
        cm_counts: (num_classes, num_classes + 1)
        cm_norm: (num_classes, num_classes + 1) chuẩn hóa theo hàng
    """
    # Cột 0..5: 6 lớp dự đoán; Cột 6: Bị từ chối
    cm_counts = np.zeros((num_classes, num_classes + 1), dtype=np.int64)

    for i in range(len(y_true)):
        gt = y_true[i]
        if gt < 0 or gt >= num_classes:
            continue  # Bỏ qua mẫu OOD nếu có

        if rejected is not None and rejected[i]:
            cm_counts[gt, num_classes] += 1
        else:
            p = y_pred[i]
            if 0 <= p < num_classes:
                cm_counts[gt, p] += 1

    # Chuẩn hóa theo hàng
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = (cm_counts / row_sums).astype(np.float64)

    return cm_counts, cm_norm


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 6,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Tính Precision, Recall, F1 và Support cho từng lớp; Macro-F1, Weighted-F1.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    precisions = []
    recalls = []
    f1s = []
    supports = []
    per_class = {}

    for c in range(num_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        support = int(np.sum(y_true == c))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        supports.append(support)

        c_name = class_names[c] if c < len(class_names) else f"Class_{c}"
        per_class[c_name] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "support": support
        }

    macro_f1 = float(np.mean(f1s))
    total_support = sum(supports)
    weighted_f1 = float(sum(f * s for f, s in zip(f1s, supports)) / total_support) if total_support > 0 else 0.0

    return {
        "per_class": per_class,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }


def bootstrap_ci(
    y_true: np.ndarray,
    probs: np.ndarray,
    metric_type: str = "top1",
    n_bootstraps: int = 1000,
    seed: int = 42,
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    """
    Tính khoảng tin cậy 95% bằng Bootstrap 1.000 lần cho Top-1 Accuracy hoặc Macro-F1.
    
    Returns:
        (mean_val, lower_ci, upper_ci)
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
            raise ValueError(f"Không hỗ trợ metric_type: {metric_type}")
        boot_vals.append(val)

    lower = float(np.percentile(boot_vals, 100 * (alpha / 2.0)))
    upper = float(np.percentile(boot_vals, 100 * (1.0 - alpha / 2.0)))
    mean_val = float(np.mean(boot_vals))
    return mean_val, lower, upper


def mcnemar_test_exact(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Tuple[int, int, float]:
    """
    Kiểm định McNemar chính xác (Exact Binomial Test) so sánh 2 model A và B.
    - n01: Số mẫu A sai nhưng B đúng.
    - n10: Số mẫu A đúng nhưng B sai.
    - p_value tính từ phân phối nhị thức với p = 0.5.
    
    Returns:
        (n10, n01, p_value)
    """
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    # A đúng, B sai
    n10 = int(np.sum(correct_a & ~correct_b))
    # A sai, B đúng
    n01 = int(np.sum(~correct_a & correct_b))

    n = n10 + n01
    if n == 0:
        return 0, 0, 1.0

    # Phân phối nhị thức chính xác: 2 * min(P(X <= k), P(X >= k)) với k = min(n10, n01), p = 0.5
    k = min(n10, n01)
    p_value = float(2.0 * stats.binom.cdf(k, n, 0.5))
    p_value = min(1.0, p_value)

    return n10, n01, p_value


def compute_ece_and_nll(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
    """
    Tính ECE (Expected Calibration Error) và NLL (Negative Log Likelihood).
    """
    N = len(y_true)
    if N == 0:
        return 0.0, 0.0

    # NLL
    eps = 1e-12
    correct_probs = probs[np.arange(N), y_true]
    nll = float(-np.mean(np.log(np.clip(correct_probs, eps, 1.0))))

    # ECE
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


def compute_coverage_accuracy(
    y_true: np.ndarray,
    probs: np.ndarray,
    taus: Optional[List[float]] = None
) -> List[Dict[str, float]]:
    """
    Tính Coverage và Accuracy theo từng ngưỡng tau in {0.1, 0.2, ..., 0.9}.
    """
    if taus is None:
        taus = [round(t * 0.1, 1) for t in range(1, 10)]

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    N = len(y_true)

    curve = []
    for tau in taus:
        accepted_mask = confidences >= tau
        num_accepted = int(np.sum(accepted_mask))
        coverage = num_accepted / N if N > 0 else 0.0

        if num_accepted > 0:
            acc = float(np.mean(predictions[accepted_mask] == y_true[accepted_mask]))
        else:
            acc = 1.0

        curve.append({
            "tau": tau,
            "coverage": float(coverage),
            "accuracy": float(acc),
            "accepted_count": num_accepted
        })
    return curve


def compute_auroc_ood(conf_id: np.ndarray, conf_ood: np.ndarray) -> float:
    """
    Tính AUROC tự code phân biệt ảnh 6 lớp (ID) và ảnh ngoài phân phối (OOD) dựa trên confidence.
    ID được kỳ vọng có confidence cao hơn OOD.
    """
    n_id = len(conf_id)
    n_ood = len(conf_ood)
    if n_id == 0 or n_ood == 0:
        return 0.5

    # Tính theo Wilcoxon-Mann-Whitney U-statistic
    # U = sum(I(conf_id > conf_ood) + 0.5 * I(conf_id == conf_ood))
    # AUROC = U / (n_id * n_ood)
    all_scores = np.concatenate([conf_id, conf_ood])
    labels = np.concatenate([np.ones(n_id), np.zeros(n_ood)])

    # Xếp hạng (rank) tăng dần
    order = np.argsort(all_scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(all_scores) + 1)

    # Xử lý các giá trị hòa (ties)
    unique_scores, inverse, counts = np.unique(all_scores, return_inverse=True, return_counts=True)
    for i, count in enumerate(counts):
        if count > 1:
            tie_mask = (inverse == i)
            ranks[tie_mask] = np.mean(ranks[tie_mask])

    # Tổng hạng của ID (nhãn 1)
    r_id = np.sum(ranks[:n_id])
    u_stat = r_id - (n_id * (n_id + 1)) / 2.0
    auroc = float(u_stat / (n_id * n_ood))
    return auroc


def measure_latency_cpu(predict_fn, sample_input, n_runs: int = 100) -> Dict[str, float]:
    """
    Đo độ trễ Latency trên CPU bằng time.perf_counter():
    - cold = lần chạy đầu tiên sau khi load model.
    - warm = 100 lần chạy tiếp theo, bỏ 10 lần đầu (tính trên 90 lần: mean, p50, p95).
    """
    # 1. Cold start
    t0 = time.perf_counter()
    _ = predict_fn(sample_input)
    t1 = time.perf_counter()
    cold_ms = (t1 - t0) * 1000.0

    # 2. Warm start
    durations_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = predict_fn(sample_input)
        t1 = time.perf_counter()
        durations_ms.append((t1 - t0) * 1000.0)

    # Bỏ 10 lần đầu
    warm_samples = durations_ms[10:]
    return {
        "cold_ms": float(cold_ms),
        "warm_mean_ms": float(np.mean(warm_samples)),
        "warm_p50_ms": float(np.percentile(warm_samples, 50)),
        "warm_p95_ms": float(np.percentile(warm_samples, 95))
    }
