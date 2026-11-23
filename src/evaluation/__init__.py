"""
Package đánh giá và kiểm định chất lượng (src/evaluation).
"""

from src.evaluation.metrics import (
    compute_accuracy_topk,
    compute_per_class_metrics,
    compute_confusion_matrix,
    compute_ece_and_nll,
    compute_coverage_accuracy,
    compute_auroc_ood,
    CLASS_NAMES
)
from src.evaluation.statistics import (
    bootstrap_ci,
    mcnemar_test_exact,
    paired_bootstrap_delta_f1
)
from src.evaluation.preflight_check import run_preflight_checks
from src.evaluation.ledger import IOLedger
from src.evaluation.robustness import run_robustness_tests

__all__ = [
    "compute_accuracy_topk",
    "compute_per_class_metrics",
    "compute_confusion_matrix",
    "compute_ece_and_nll",
    "compute_coverage_accuracy",
    "compute_auroc_ood",
    "CLASS_NAMES",
    "bootstrap_ci",
    "mcnemar_test_exact",
    "paired_bootstrap_delta_f1",
    "run_preflight_checks",
    "IOLedger",
    "run_robustness_tests"
]
