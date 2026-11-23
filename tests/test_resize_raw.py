"""
Unit test cho quy trình Offline Resize (src/data/resize_raw.py)
Kiểm tra:
- Ảnh lớn > 800px được resize về cạnh dài = 800px với cv2.INTER_AREA.
- Ảnh <= 800px giữ nguyên kích thước.
- Tọa độ bbox trong manifest được scale chuẩn xác theo tỉ lệ ảnh.
"""

import os
import shutil
import unittest
import numpy as np
import cv2
import pandas as pd

from src.data.resize_raw import resize_single_image, process_dataset


class TestResizeRaw(unittest.TestCase):
    """Kiểm tra chức năng Offline Resize."""

    def setUp(self):
        self.tmp_raw = "tests/tmp_raw"
        self.tmp_proc = "tests/tmp_processed"
        self.tmp_manifest_in = "tests/tmp_manifest.csv"
        self.tmp_manifest_out = "tests/tmp_manifest_proc.csv"
        os.makedirs(self.tmp_raw, exist_ok=True)

    def tearDown(self):
        for p in [self.tmp_raw, self.tmp_proc]:
            if os.path.exists(p):
                shutil.rmtree(p)
        for f in [self.tmp_manifest_in, self.tmp_manifest_out]:
            if os.path.exists(f):
                os.remove(f)

    def test_resize_large_and_small_images(self):
        """Ảnh lớn được thu nhỏ về 800px; ảnh nhỏ giữ nguyên."""
        # 1. Ảnh lớn: 1200x1600 (chiều cao 1200, rộng 1600)
        large_img = np.full((1200, 1600, 3), 150, dtype=np.uint8)
        large_path = os.path.join(self.tmp_raw, "large.jpg")
        cv2.imwrite(large_path, large_img)

        out_large_path = os.path.join(self.tmp_proc, "large.jpg")
        success, scale = resize_single_image(large_path, out_large_path, max_size=800, quality=85)

        self.assertTrue(success)
        self.assertAlmostEqual(scale, 800.0 / 1600.0, places=4)  # scale = 0.5
        res_large = cv2.imread(out_large_path)
        h, w = res_large.shape[:2]
        self.assertEqual(max(h, w), 800)
        self.assertEqual(w, 800)
        self.assertEqual(h, 600)

        # 2. Ảnh nhỏ: 300x400
        small_img = np.full((300, 400, 3), 100, dtype=np.uint8)
        small_path = os.path.join(self.tmp_raw, "small.jpg")
        cv2.imwrite(small_path, small_img)

        out_small_path = os.path.join(self.tmp_proc, "small.jpg")
        success_s, scale_s = resize_single_image(small_path, out_small_path, max_size=800, quality=85)

        self.assertTrue(success_s)
        self.assertEqual(scale_s, 1.0)
        res_small = cv2.imread(out_small_path)
        self.assertEqual(res_small.shape[:2], (300, 400))
        print("[PASS] Offline resize: Ảnh lớn thu nhỏ về 800px, ảnh nhỏ giữ nguyên kích thước.")

    def test_process_dataset_with_bbox_update(self):
        """Kiểm tra xử lý dataset và cập nhật tọa độ bbox theo scale."""
        # Tạo ảnh 1000x2000
        img = np.full((1000, 2000, 3), 120, dtype=np.uint8)
        p = os.path.join(self.tmp_raw, "motor.jpg")
        cv2.imwrite(p, img)

        # Bbox trên ảnh gốc: [100, 200, 500, 800]
        # max_dim = 2000 -> resize về 800 -> scale = 800/2000 = 0.4
        # Bbox mới kỳ vọng: [40, 80, 200, 320]
        df = pd.DataFrame([{
            "image_path": p,
            "label": "Honda Wave",
            "group_id": "g1",
            "bbox": "[100, 200, 500, 800]"
        }])
        df.to_csv(self.tmp_manifest_in, index=False)

        process_dataset(
            raw_dir=self.tmp_raw,
            processed_dir=self.tmp_proc,
            manifest_in=self.tmp_manifest_in,
            manifest_out=self.tmp_manifest_out,
            max_size=800,
            quality=85
        )

        self.assertTrue(os.path.exists(self.tmp_manifest_out))
        df_out = pd.read_csv(self.tmp_manifest_out)
        self.assertEqual(len(df_out), 1)

        new_bbox_str = df_out.iloc[0]["bbox"]
        new_bbox = eval(new_bbox_str)
        expected_bbox = [40, 80, 200, 320]
        self.assertEqual(new_bbox, expected_bbox)
        print(f"[PASS] Bbox được scale chính xác: {expected_bbox} -> khớp với tỉ lệ co giãn 0.4.")


if __name__ == "__main__":
    unittest.main()
