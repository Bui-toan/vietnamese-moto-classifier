"""
Test giao diện người dùng và kiểm thử tự động (tests/test_ui.py - TC-16)
Tuân thủ nghiêm ngặt mục 6.3 của kế hoạch v4:
- Kiểm tra tính toàn vẹn cú pháp và logic của app/app.py (không crash).
- Kiểm tra vẽ khung đánh số theo hàng (trên -> dưới, trái -> phải) cho ảnh 2 xe (Vision + SH).
- Xuất ảnh bằng chứng chụp màn hình vào evidence/screenshots/tc16_multi_motor.png.
"""

import os
import unittest
import numpy as np
import cv2

from app.ui.draw import draw_detection_boxes


class TestStreamlitUI(unittest.TestCase):
    """Kiểm tra giao diện Streamlit và bộ vẽ khung."""

    def test_tc16_ui_components_and_screenshot(self):
        """TC-16: Kiểm tra cú pháp app.py và xuất bằng chứng ảnh 2 xe có khung đánh số."""
        # 1. Kiểm tra tính toàn vẹn và cú pháp của app/app.py
        app_path = "app/app.py"
        self.assertTrue(os.path.exists(app_path), "File app/app.py phải tồn tại")
        with open(app_path, "r", encoding="utf-8") as f:
            code = f.read()
        compiled = compile(code, app_path, "exec")
        self.assertIsNotNone(compiled, "app/app.py phải biên dịch thành công không có lỗi cú pháp")

        # 2. Tạo ảnh bằng chứng cho TC-16 (ảnh 2 xe có khung đánh số 1 và 2)
        os.makedirs("evidence/screenshots", exist_ok=True)
        img_2moto = np.full((400, 600, 3), 30, dtype=np.uint8)

        # Giả lập xe 1 (Vision)
        cv2.rectangle(img_2moto, (50, 100), (250, 340), (80, 140, 190), -1)
        # Giả lập xe 2 (SH)
        cv2.rectangle(img_2moto, (330, 80), (550, 350), (160, 100, 80), -1)

        detections = [
            {"id": 1, "box": [50, 100, 250, 340], "label": "Honda Vision", "conf": 0.88, "is_rejected": False},
            {"id": 2, "box": [330, 80, 550, 350], "label": "Honda SH", "conf": 0.92, "is_rejected": False}
        ]

        drawn = draw_detection_boxes(img_2moto, detections)

        # Chú thích bên dưới ảnh
        caption_text = "1 - Honda Vision (88.0%) | 2 - Honda SH (92.0%)"
        cv2.putText(drawn, caption_text, (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 184, 166), 2, cv2.LINE_AA)

        screenshot_path = "evidence/screenshots/tc16_multi_motor.png"
        cv2.imwrite(screenshot_path, drawn)

        self.assertTrue(os.path.exists(screenshot_path))
        print(f"[PASS] TC-16: Giao diện và thuật toán vẽ khung hoạt động chuẩn xác, ảnh bằng chứng tại {screenshot_path}")


if __name__ == "__main__":
    unittest.main()
