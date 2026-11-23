"""
Module tính toán các chỉ số đánh giá (src/evaluation/metrics.py)
Tuân thủ nghiêm ngặt mục 6.4 của kế hoạch v4:
- compute_accuracy_topk: Top-1, Top-2, Top-3.
- compute_per_class_metrics: Precision, Recall, F1, Support từng lớp; Macro-F1, Weighted-F1.
- compute_confusion_matrix: Ma trận 6x6 kèm cột phụ 'Từ chối'.
- compute_ece_and_nll: Expected Calibration Error và NLL.
- compute_coverage_accuracy: Đường cong Coverage-Accuracy theo tau.
- compute_auroc_ood: AUROC phân biệt ID và OOD tự code theo Wilcoxon U-statistic.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np

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
    """Tính Top-k Accuracy (k in ks)."""
    res = {}
    N = len(y_true)
    if N == 0:
        return {f"top{k}_acc": 0.0 for k in ks}

    for k in ks:
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
    Ma trận nhầm lẫn 6x6 kèm cột phụ 'Từ chối'.
    Returns: (cm_counts, cm_norm).
    """
    cm_counts = np.zeros((num_classes, num_classes + 1), dtype=np.int64)

    for i in range(len(y_true)):
        gt = y_true[i]
        if gt < 0 or gt >= num_classes:
            continue

        if rejected is not None and rejected[i]:
            cm_counts[gt, num_classes] += 1
        else:
            p = y_pred[i]
            if 0 <= p < num_classes:
                cm_counts[gt, p] += 1

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
    """Tính P, R, F1, Support từng lớp; Macro-F1, Weighted-F1."""
    if class_names is None:
        class_names = CLASS_NAMES

    precisions, recalls, f1s, supports = [], [], [], []
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
        per_class[c_name] = {"precision": float(p), "recall": float(r), "f1": float(f1), "support": support}

    macro_f1 = float(np.mean(f1s))
    total_support = sum(supports)
    weighted_f1 = float(sum(f * s for f, s in zip(f1s, supports)) / total_support) if total_support > 0 else 0.0

    return {
        "per_class": per_class,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }


def compute_ece_and_nll(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
    """Tính ECE và NLL."""
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


def compute_coverage_accuracy(
    y_true: np.ndarray,
    probs: np.ndarray,
    taus: Optional[List[float]] = None
) -> List[Dict[str, float]]:
    """Tính Coverage và Accuracy theo từng ngưỡng tau."""
    if taus is None:
        taus = [round(t * 0.1, 1) for t in range(1, 10)]

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    N = len(y_true)

    curve = []
    for tau in taus:
        accepted = (confidences >= tau)
        num_accepted = int(np.sum(accepted))
        coverage = num_accepted / N if N > 0 else 0.0
        acc = float(np.mean(predictions[accepted] == y_true[accepted])) if num_accepted > 0 else 1.0

        curve.append({
            "tau": tau,
            "coverage": float(coverage),
            "accuracy": float(acc),
            "accepted_count": num_accepted
        })
    return curve


def compute_auroc_ood(conf_id: np.ndarray, conf_ood: np.ndarray) -> float:
    """Tính AUROC tự code phân biệt ID và OOD."""
    n_id, n_ood = len(conf_id), len(conf_ood)
    if n_id == 0 or n_ood == 0:
        return 0.5

    all_scores = np.concatenate([conf_id, conf_ood])
    order = np.argsort(all_scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(all_scores) + 1)

    unique_scores, inverse, counts = np.unique(all_scores, return_inverse=True, return_counts=True)
    for i, count in enumerate(counts):
        if count > 1:
            ranks[inverse == i] = np.mean(ranks[inverse == i])

    r_id = np.sum(ranks[:n_id])
    u_stat = r_id - (n_id * (n_id + 1)) / 2.0
    return float(u_stat / (n_id * n_ood))
