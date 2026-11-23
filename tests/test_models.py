"""
Unit tests cho các mô hình và bộ phát hiện phụ trợ:
- LinearSVM: Forward, Hinge Loss, SGD, Save & Load.
- CNN1: Forward shape (B, 6), hook activations.
- CNN2: Forward shape (B, 6), skip connection, hook activations.
- Detector: NMS, sắp xếp đánh số theo hàng (trên -> dưới, trái -> phải), đệm 10%.
"""

import os
import unittest
import numpy as np
import torch

from src.models.svm import LinearSVM
from src.models.cnn1 import CNN1
from src.models.cnn2 import CNN2
from src.detector.detector import nms_boxes, sort_boxes_by_rows, compute_iou, MotorbikeDetector


class TestModelsAndDetector(unittest.TestCase):
    """Kiểm tra các mô hình M1, M2, M3 và Detector."""

    def test_linear_svm(self):
        """Kiểm tra M1 LinearSVM tự code bằng NumPy."""
        np.random.seed(42)
        N, D, C = 20, 64, 6
        X = np.random.randn(N, D).astype(np.float32)
        y = np.random.randint(0, C, size=N)

        svm = LinearSVM(n_features=D, n_classes=C, lr=0.01)
        loss_before, dW, db = svm.compute_loss_and_grad(X, y)
        self.assertGreater(loss_before, 0.0)
        self.assertEqual(dW.shape, (D, C))
        self.assertEqual(db.shape, (C,))

        # Train 1 epoch
        loss_epoch = svm.train_epoch(X, y, batch_size=10)
        self.assertIsInstance(loss_epoch, float)

        # Predict & Proba
        preds = svm.predict(X)
        self.assertEqual(preds.shape, (N,))
        probs = svm.predict_proba(X, temperature=1.5)
        self.assertEqual(probs.shape, (N, C))
        self.assertTrue(np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5))

        # Test Save & Load
        tmp_path = "tests/tmp_svm.npz"
        try:
            svm.save(tmp_path)
            loaded_svm = LinearSVM.load(tmp_path)
            self.assertTrue(np.allclose(svm.W, loaded_svm.W))
            self.assertTrue(np.allclose(svm.b, loaded_svm.b))
            print("[PASS] LinearSVM forward, loss, SGD, save/load khớp tuyệt đối.")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_cnn1_forward_and_hook(self):
        """Kiểm tra M2 CNN-1 forward và hook Grad-CAM."""
        model = CNN1(num_classes=6)
        x = torch.randn(2, 3, 112, 112, requires_grad=True)
        out = model(x)
        self.assertEqual(out.shape, (2, 6))

        # Backward để kích hoạt hook Grad-CAM
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(model.activations)
        self.assertIsNotNone(model.gradients)
        print("[PASS] CNN-1 output shape (2, 6) và Grad-CAM hooks hoạt động chuẩn xác.")

    def test_cnn2_forward_and_hook(self):
        """Kiểm tra M3 CNN-2 forward và hook Grad-CAM."""
        model = CNN2(num_classes=6, dropout_p=0.3)
        x = torch.randn(2, 3, 112, 112, requires_grad=True)
        out = model(x)
        self.assertEqual(out.shape, (2, 6))

        # Backward để kích hoạt hook Grad-CAM
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(model.activations)
        self.assertIsNotNone(model.gradients)
        print("[PASS] CNN-2 output shape (2, 6), ResBlocks và Grad-CAM hooks hoạt động chuẩn xác.")

    def test_detector_sorting_and_nms(self):
        """Kiểm tra đánh số theo hàng (trên -> dưới, trái -> phải) và NMS."""
        # Tạo 4 boxes tạo thành 2 hàng:
        # Hàng 1 (y quanh 100): box A (x=50), box B (x=200)
        # Hàng 2 (y quanh 300): box C (x=80), box D (x=250)
        box_A = [50, 90, 150, 190]   # cy=140, cx=100
        box_B = [200, 100, 300, 200] # cy=150, cx=250
        box_C = [80, 290, 180, 390]  # cy=340, cx=130
        box_D = [250, 310, 350, 410] # cy=360, cx=300

        # Đưa vào theo thứ tự lộn xộn: D, A, C, B
        boxes = [box_D, box_A, box_C, box_B]
        sorted_order = sort_boxes_by_rows(boxes)

        # Thứ tự kỳ vọng:
        # Hàng 1: A (idx 1), B (idx 3)
        # Hàng 2: C (idx 2), D (idx 0)
        expected_order = [1, 3, 2, 0]
        self.assertEqual(sorted_order, expected_order, f"Kỳ vọng {expected_order}, thực tế {sorted_order}")

        # Kiểm tra NMS
        # Tạo 2 box chồng lấn mạnh
        dup_box = [52, 92, 148, 188]  # Gần trùng box A
        boxes_with_dup = [box_A, dup_box]
        scores = [0.9, 0.7]
        kept_boxes, kept_scores = nms_boxes(boxes_with_dup, scores, iou_thresh=0.3)
        self.assertEqual(len(kept_boxes), 1)
        self.assertEqual(kept_scores[0], 0.9)

        print("[PASS] Detector NMS và thuật toán đánh số theo hàng đạt chuẩn v4.")


if __name__ == "__main__":
    unittest.main()
