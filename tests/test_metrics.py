"""
Unit tests cho thư viện metrics và calibration:
- TC-04: metrics trên ma trận nhầm lẫn nhỏ tính tay (Accuracy, Macro-F1, Top-k khớp tuyệt đối).
- Bootstrap 95% CI.
- McNemar test chính xác.
- ECE, NLL và Temperature Scaling.
- Xác định 3 ngưỡng: default_threshold, ood_threshold, reject_threshold.
"""

import unittest
import numpy as np

from src.eval.metrics import (
    compute_accuracy_topk,
    compute_confusion_matrix,
    compute_per_class_metrics,
    bootstrap_ci,
    mcnemar_test_exact,
    compute_ece_and_nll,
    compute_coverage_accuracy,
    compute_auroc_ood
)
from src.eval.calibration import optimize_temperature, compute_all_thresholds


class TestMetricsAndCalibration(unittest.TestCase):
    """Kiểm tra các hàm tính toán chỉ số thống kê và calibration."""

    def test_tc04_hand_calculated_metrics(self):
        """TC-04: metrics trên ma trận nhầm lẫn nhỏ tính tay: Accuracy, Macro-F1, Top-k khớp tuyệt đối."""
        # Giả sử bài toán 3 lớp với 6 mẫu:
        # Ground truth: [0, 0, 1, 1, 2, 2]
        # Predictions:  [0, 1, 1, 1, 2, 0]
        # - Lớp 0: TP=1, FP=1 (mẫu 6), FN=1 (mẫu 2) -> P = 1/2 = 0.5, R = 1/2 = 0.5, F1 = 0.5
        # - Lớp 1: TP=2, FP=1 (mẫu 2), FN=0         -> P = 2/3, R = 2/2 = 1.0, F1 = 4/5 = 0.8
        # - Lớp 2: TP=1, FP=0, FN=1 (mẫu 6)         -> P = 1/1 = 1.0, R = 1/2 = 0.5, F1 = 2/3
        # Macro-F1 = (0.5 + 0.8 + 2/3) / 3 = 1.966666... / 3 = 0.655555...
        # Accuracy = 4 / 6 = 2/3 = 0.666666...
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])

        probs = np.array([
            [0.7, 0.2, 0.1],  # 0
            [0.3, 0.6, 0.1],  # 1 (sai, gt=0)
            [0.1, 0.8, 0.1],  # 1
            [0.2, 0.5, 0.3],  # 1
            [0.1, 0.1, 0.8],  # 2
            [0.5, 0.1, 0.4]   # 0 (sai, gt=2)
        ])

        # 1. Kiểm tra Top-k
        topk = compute_accuracy_topk(y_true, probs, ks=(1, 2))
        expected_top1 = 4.0 / 6.0
        self.assertAlmostEqual(topk["top1_acc"], expected_top1, places=5)
        # Top-2:
        # mẫu 0: [0, 1] -> có 0 (đúng)
        # mẫu 1: [1, 0] -> có 0 (đúng)
        # mẫu 2: [1, ?] -> có 1 (đúng)
        # mẫu 3: [1, 2] -> có 1 (đúng)
        # mẫu 4: [2, ?] -> có 2 (đúng)
        # mẫu 5: [0, 2] -> có 2 (đúng)
        # -> Cả 6 mẫu đều đúng ở Top-2 -> Top-2 = 1.0
        self.assertAlmostEqual(topk["top2_acc"], 1.0, places=5)

        # 2. Kiểm tra Per-class metrics
        res = compute_per_class_metrics(y_true, y_pred, num_classes=3)
        expected_macro_f1 = (0.5 + 0.8 + (2.0 / 3.0)) / 3.0
        self.assertAlmostEqual(res["macro_f1"], expected_macro_f1, places=5)

        # 3. Ma trận nhầm lẫn
        cm_counts, cm_norm = compute_confusion_matrix(y_true, y_pred, num_classes=3)
        expected_cm = np.array([
            [1, 1, 0, 0],
            [0, 2, 0, 0],
            [1, 0, 1, 0]
        ])
        self.assertTrue(np.array_equal(cm_counts, expected_cm))

        print("[PASS] TC-04: Accuracy, Macro-F1, Top-k, Confusion Matrix khớp tuyệt đối với số tính tay.")

    def test_mcnemar_exact(self):
        """Kiểm tra kiểm định McNemar chính xác."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        pred_a = np.array([0, 0, 1, 1, 1, 0])  # Sai mẫu 2 và 5
        pred_b = np.array([0, 0, 0, 1, 1, 1])  # Đúng hết
        # A đúng B sai: 0 (n10 = 0)
        # A sai B đúng: 2 (n01 = 2)
        n10, n01, p_val = mcnemar_test_exact(y_true, pred_a, pred_b)
        self.assertEqual(n10, 0)
        self.assertEqual(n01, 2)
        self.assertLessEqual(p_val, 1.0)
        print(f"[PASS] McNemar exact: n10={n10}, n01={n01}, p_val={p_val:.4f}.")

    def test_bootstrap_ci(self):
        """Kiểm tra Bootstrap 95% Confidence Interval."""
        np.random.seed(42)
        y_true = np.random.randint(0, 6, size=100)
        probs = np.random.dirichlet(np.ones(6), size=100)

        mean_val, lower, upper = bootstrap_ci(y_true, probs, metric_type="top1", n_bootstraps=200, seed=42)
        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(upper, 1.0)
        self.assertLessEqual(lower, upper)
        self.assertTrue(lower <= mean_val <= upper)
        print(f"[PASS] Bootstrap 95% CI: mean={mean_val:.4f}, [{lower:.4f}, {upper:.4f}].")

    def test_calibration_and_thresholds(self):
        """Kiểm tra Temperature Scaling và tính toán 3 ngưỡng."""
        np.random.seed(42)
        logits = np.random.randn(50, 6)
        y_val = np.random.randint(0, 6, size=50)

        T, ece_b, nll_b, ece_a, nll_a = optimize_temperature(logits, y_val)
        self.assertGreater(T, 0.0)
        self.assertLessEqual(nll_a, nll_b + 1e-4)

        # Test tìm ngưỡng
        probs = np.random.dirichlet(np.ones(6), size=50)
        ood_probs = np.random.dirichlet(np.ones(6), size=30)
        threshs = compute_all_thresholds(y_val, probs, ood_probs)

        self.assertIn("default_threshold", threshs)
        self.assertIn("ood_threshold", threshs)
        self.assertIn("reject_threshold", threshs)
        self.assertGreaterEqual(threshs["reject_threshold"], threshs["default_threshold"])
        self.assertGreaterEqual(threshs["reject_threshold"], threshs["ood_threshold"])
        print(f"[PASS] Calibration T={T:.2f}, Thresholds={threshs}.")


if __name__ == "__main__":
    unittest.main()
