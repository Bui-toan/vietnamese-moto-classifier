"""
Unit tests cho Pipeline dữ liệu, Chống rò rỉ và Khóa dữ liệu:
- Kiểm tra thuật toán split_groups (deficit).
- TC-05: Kiểm tra Average-hash 8x8 tự code bằng NumPy, phát hiện near-duplicate.
- TC-06: Khóa dữ liệu, kiểm định toàn vẹn, và thử nghiệm sửa 1 byte trên bản sao test_frozen_copy phải báo FAIL.
"""

import os
import shutil
import unittest
import numpy as np
import cv2

from src.data.split import split_groups
from src.data.check_leak import compute_average_hash_8x8, hamming_distance
from src.data.lock import lock_dataset
from src.data.verify_lock import verify_lock


class TestDataPipeline(unittest.TestCase):
    """Kiểm tra pipeline dữ liệu, chống rò rỉ và khóa dữ liệu."""

    def test_split_groups_algorithm(self):
        """Kiểm tra thuật toán chia group deficit theo mục 3.2."""
        # 12 group với số ảnh khác nhau
        items = {f"group_{i}": 10 + i * 5 for i in range(12)}
        (tr, val, te), got = split_groups(items, ratios=(0.70, 0.15, 0.15), seed=42)

        # 1. Không trùng lặp
        set_tr, set_val, set_te = set(tr), set(val), set(te)
        self.assertEqual(len(set_tr & set_val), 0)
        self.assertEqual(len(set_tr & set_te), 0)
        self.assertEqual(len(set_val & set_te), 0)

        # 2. Toàn bộ group được phân bổ
        self.assertEqual(len(set_tr | set_val | set_te), 12)
        print(f"[PASS] split_groups phân bổ chính xác: train={len(tr)}, val={len(val)}, test={len(te)}.")

    def test_tc05_average_hash_near_duplicate(self):
        """TC-05: Average-hash 8x8 tự code bằng NumPy, Hamming distance <= 5."""
        np.random.seed(42)
        # Tạo ảnh gốc 64x64
        base_img = np.random.randint(50, 200, (64, 64), dtype=np.uint8)

        # Ảnh gần trùng: chỉ thay đổi nhẹ độ sáng (+3 đơn vị) và thêm nhiễu rất nhỏ
        near_img = np.clip(base_img.astype(float) + 3.0, 0, 255).astype(np.uint8)
        near_img[:5, :5] = 120

        # Ảnh khác biệt hoàn toàn (đảo ngược màu và hoa văn hoàn toàn khác)
        diff_img = (255 - base_img)[::-1, ::-1]

        h_base = compute_average_hash_8x8(base_img)
        h_near = compute_average_hash_8x8(near_img)
        h_diff = compute_average_hash_8x8(diff_img)

        # 1. Hash 64 bit
        self.assertEqual(len(h_base), 64)

        # 2. Khoảng cách Hamming của ảnh gần trùng phải <= 5
        dist_near = hamming_distance(h_base, h_near)
        self.assertLessEqual(dist_near, 5, f"Ảnh gần trùng phải có Hamming <= 5, thực tế={dist_near}")

        # 3. Khoảng cách Hamming của ảnh khác biệt phải lớn
        dist_diff = hamming_distance(h_base, h_diff)
        self.assertGreater(dist_diff, 10, f"Ảnh khác biệt phải có Hamming lớn, thực tế={dist_diff}")

        print(f"[PASS] TC-05: aHash 8x8 tự code phát hiện near-duplicate chuẩn xác (dist_near={dist_near} <= 5, dist_diff={dist_diff} > 10).")

    def test_tc06_lock_and_tamper_detection(self):
        """TC-06: Khóa dữ liệu, verify PASS; sửa 1 byte trên bản sao test_frozen_copy phải FAIL."""
        test_tmp_dir = "tests/tmp_test_data"
        frozen_dir = "tests/tmp_test_frozen"
        copy_dir = "tests/tmp_test_frozen_copy"
        lock_json = "tests/tmp_lock.json"

        os.makedirs(test_tmp_dir, exist_ok=True)
        try:
            # Tạo 3 ảnh mẫu
            file_paths = []
            for i in range(3):
                p = os.path.join(test_tmp_dir, f"sample_{i}.png")
                img = np.full((32, 32, 3), i * 50 + 20, dtype=np.uint8)
                cv2.imwrite(p, img)
                file_paths.append(p)

            # Tạo split CSV giả lập
            splits_tmp = "tests/tmp_splits"
            os.makedirs(splits_tmp, exist_ok=True)
            import pandas as pd
            df_te = pd.DataFrame({"image_path": file_paths, "label": ["Wave", "Vision", "SH"], "group_id": ["g1", "g2", "g3"]})
            df_te.to_csv(os.path.join(splits_tmp, "test.csv"), index=False)

            # 1. Khóa dữ liệu
            lock_dataset(splits_dir=splits_tmp, frozen_dir=frozen_dir, out_lock_json=lock_json)

            # 2. Verify trên thư mục gốc -> Phải PASS
            pass_result = verify_lock(lock_json_path=lock_json, data_dir=frozen_dir)
            self.assertTrue(pass_result, "Xác minh trên thư mục gốc phải PASS")

            # 3. Tạo bản sao test_frozen_copy
            if os.path.exists(copy_dir):
                shutil.rmtree(copy_dir)
            shutil.copytree(frozen_dir, copy_dir)

            # 4. Sửa đúng 1 byte trong file đầu tiên của bản sao
            tampered_file = os.path.join(copy_dir, "sample_0.png")
            with open(tampered_file, "r+b") as f:
                f.seek(10)
                orig_byte = f.read(1)
                # Đổi byte
                new_byte = bytes([(orig_byte[0] ^ 0xFF)])
                f.seek(10)
                f.write(new_byte)

            # 5. Verify trên thư mục bản sao bị sửa -> Phải FAIL
            fail_result = verify_lock(lock_json_path=lock_json, data_dir=copy_dir)
            self.assertFalse(fail_result, "Xác minh trên thư mục bị sửa 1 byte phải FAIL")
            print("[PASS] TC-06: Sửa 1 byte trong test_frozen_copy bị bắt lỗi lập tức (FAIL). Bảo vệ toàn vẹn tuyệt đối.")

        finally:
            # Dọn dẹp thư mục tạm
            for d in [test_tmp_dir, frozen_dir, copy_dir, "tests/tmp_splits"]:
                if os.path.exists(d):
                    shutil.rmtree(d)
            if os.path.exists(lock_json):
                os.remove(lock_json)


if __name__ == "__main__":
    unittest.main()
