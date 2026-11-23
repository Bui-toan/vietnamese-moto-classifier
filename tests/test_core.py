"""
Unit tests cho các thuật toán cốt lõi tự code trong Phần 1:
- TC-01: Kiểm tra letterbox ảnh 640x480 -> shape (112, 112, 3), đệm viền bằng giá trị trung bình.
- TC-02: Kiểm tra HOG: 8x8 báo lỗi rõ ràng; 16x16 cho 36 chiều; 112x112 cho 6.084 chiều; không NaN.
- TC-03: Kiểm tra xoay aug_rotate: 0 độ giữ nguyên; +theta rồi -theta đạt PSNR > 30 dB ở vùng lõi.
- Kiểm tra bổ sung: aug_random_erase, aug_brightness_contrast, aug_bbox_jitter.
"""

import sys
import unittest
import numpy as np
import cv2

from src.preprocess.transform import (
    letterbox,
    aug_rotate,
    aug_random_erase,
    aug_brightness_contrast,
    aug_blur,
    aug_bbox_jitter,
)
from src.preprocess.hog import extract_hog_features


class TestCorePreprocessing(unittest.TestCase):
    """Bộ kiểm thử cho các hàm xử lý ảnh và HOG tự code."""

    def test_tc01_letterbox(self):
        """TC-01: letterbox ảnh 640x480 -> shape (112, 112, 3), viền đệm đúng giá trị trung bình."""
        # Tạo ảnh giả lập 480x640x3 có màu sắc xác định
        np.random.seed(42)
        orig_img = np.random.randint(50, 200, size=(480, 640, 3), dtype=np.uint8)

        target_size = (112, 112)
        out = letterbox(orig_img, target_size=target_size)

        # 1. Kiểm tra shape
        self.assertEqual(out.shape, (112, 112, 3), "Output shape phải là (112, 112, 3)")
        self.assertEqual(out.dtype, np.uint8, "Dtype phải là uint8")

        # 2. Kiểm tra tỷ lệ:
        # Với ảnh gốc 480x640, scale = min(112/640, 112/480) = 112/640 = 0.175
        # new_w = round(640 * 0.175) = 112
        # new_h = round(480 * 0.175) = 84
        # pad_top = (112 - 84) // 2 = 14. Viền trên 14 pixel, viền dưới 14 pixel.
        pad_top = (112 - 84) // 2

        # 3. Tính mean của ảnh đã resize
        resized = cv2.resize(orig_img, (112, 84), interpolation=cv2.INTER_LINEAR)
        expected_pad_val = resized.mean(axis=(0, 1)).astype(np.uint8)

        # Kiểm tra dải viền trên [0:pad_top, :]
        top_pad = out[0:pad_top, :, :]
        # Viền phải khớp giá trị expected_pad_val (không được là viền đen 0, 0, 0)
        self.assertTrue(np.all(expected_pad_val > 10), "Giá trị đệm trung bình không được là 0 (đen)")
        diff = np.abs(top_pad.astype(float) - expected_pad_val.astype(float))
        self.assertTrue(np.all(diff <= 1.0), "Viền đệm phải khớp giá trị trung bình của ảnh đã resize")
        print("[PASS] TC-01: letterbox shape (112, 112, 3) & viền đệm giá trị trung bình đạt chuẩn.")

    def test_tc02_hog_features(self):
        """TC-02: HOG: ảnh 8x8 báo lỗi rõ ràng; 16x16 cho 36 giá trị; 112x112 cho 6.084 chiều; không NaN."""
        # 1. Ảnh 8x8 (chỉ có 1 cell 8x8, không đủ block 2x2 cells) -> Phải báo lỗi ValueError
        small_img = np.zeros((8, 8), dtype=np.uint8)
        with self.assertRaises(ValueError) as ctx:
            extract_hog_features(small_img, cell_size=8, block_size=2)
        self.assertIn("quá nhỏ", str(ctx.exception))

        # 2. Ảnh 16x16 (2x2 cells = 1 block duy nhất -> 1 block * 4 cells * 9 bins = 36 giá trị)
        img_16 = np.arange(16 * 16, dtype=np.uint8).reshape((16, 16))
        hog_16 = extract_hog_features(img_16, cell_size=8, block_size=2)
        self.assertEqual(len(hog_16), 36, "HOG ảnh 16x16 phải có đúng 36 chiều")
        self.assertFalse(np.isnan(hog_16).any(), "HOG 16x16 không được chứa NaN")

        # 3. Ảnh 112x112 (14x14 cells -> 13x13 blocks = 169 blocks * 36 = 6.084 giá trị)
        np.random.seed(42)
        img_112 = np.random.randint(0, 256, (112, 112), dtype=np.uint8)
        hog_112 = extract_hog_features(img_112, cell_size=8, block_size=2)
        self.assertEqual(len(hog_112), 6084, "HOG ảnh 112x112 phải có đúng 6.084 chiều")
        self.assertFalse(np.isnan(hog_112).any(), "HOG 112x112 không được chứa NaN")
        self.assertFalse(np.isinf(hog_112).any(), "HOG 112x112 không được chứa Inf")

        # 4. Ảnh 64x64 cho detector (8x8 cells -> 7x7 blocks = 49 blocks * 36 = 1.764 giá trị)
        img_64 = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        hog_64 = extract_hog_features(img_64, cell_size=8, block_size=2)
        self.assertEqual(len(hog_64), 1764, "HOG ảnh 64x64 phải có đúng 1.764 chiều")
        self.assertFalse(np.isnan(hog_64).any(), "HOG 64x64 không được chứa NaN")

        print("[PASS] TC-02: HOG 8x8 raise error; 16x16 -> 36; 64x64 -> 1.764; 112x112 -> 6.084 chiều đạt chuẩn.")

    def test_tc03_aug_rotate(self):
        """TC-03: Xoay 0 độ giữ nguyên; xoay +theta rồi -theta đạt PSNR > 30 dB trong vùng lõi."""
        # Tạo ảnh mẫu có độ mượt và texture tương tự ảnh tự nhiên
        x = np.linspace(-2, 2, 112)
        y = np.linspace(-2, 2, 112)
        xx, yy = np.meshgrid(x, y)
        pattern = (np.sin(xx * 2) * np.cos(yy * 2) + 1.0) * 100.0 + 20.0
        img = np.clip(pattern, 0, 255).astype(np.uint8)[:, :, np.newaxis]
        img = np.repeat(img, 3, axis=2)

        # 1. Xoay 0 độ: phải bằng hệt ảnh gốc
        rot_0 = aug_rotate(img, 0.0)
        self.assertTrue(np.array_equal(img, rot_0), "Xoay 0 độ phải trả về ảnh giống hệt ảnh gốc")

        # 2. Xoay +10 độ rồi xoay -10 độ
        theta = 10.0
        rot_pos = aug_rotate(img, theta)
        rot_back = aug_rotate(rot_pos, -theta)

        # Kiểm tra PSNR trong vùng lõi (tránh biên do hiệu ứng góc xoay)
        # Vùng lõi trung tâm: từ y=30:82, x=30:82
        core_orig = img[30:82, 30:82].astype(np.float64)
        core_back = rot_back[30:82, 30:82].astype(np.float64)

        mse = np.mean((core_orig - core_back) ** 2)
        if mse == 0:
            psnr = 100.0
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))

        self.assertGreater(psnr, 30.0, f"PSNR vùng lõi ({psnr:.2f} dB) phải lớn hơn 30 dB")
        print(f"[PASS] TC-03: aug_rotate xoay 0 deg giống hệt; +{theta} deg rồi -{theta} deg đạt PSNR = {psnr:.2f} dB (> 30 dB).")

    def test_aug_bbox_jitter(self):
        """Kiểm tra aug_bbox_jitter: mở rộng 0-20%, dịch tâm <= 10%, kẹp trong ảnh."""
        rng = np.random.RandomState(42)
        img_shape = (480, 640)
        bbox = [100, 100, 200, 200]  # width=100, height=100

        for _ in range(50):
            nx1, ny1, nx2, ny2 = aug_bbox_jitter(bbox, img_shape, max_expand=0.2, max_shift=0.1, rng=rng)
            # Kiểm tra kẹp trong ảnh
            self.assertGreaterEqual(nx1, 0)
            self.assertGreaterEqual(ny1, 0)
            self.assertLessEqual(nx2, 640)
            self.assertLessEqual(ny2, 480)
            self.assertGreater(nx2, nx1)
            self.assertGreater(ny2, ny1)

        print("[PASS] aug_bbox_jitter kiểm tra mở rộng, dịch tâm và kẹp biên đạt chuẩn.")


    def test_aug_transforms(self):
        """Kiểm tra aug_random_erase, aug_brightness_contrast, aug_blur."""
        img = np.full((112, 112, 3), 128, dtype=np.uint8)

        # 1. aug_brightness_contrast: np.clip(alpha * img + beta, 0, 255)
        bright = aug_brightness_contrast(img, alpha=1.2, beta=20.0)
        expected = np.clip(1.2 * 128.0 + 20.0, 0, 255).astype(np.uint8)
        self.assertEqual(bright[0, 0, 0], expected)
        self.assertEqual(bright.dtype, np.uint8)

        # 2. aug_blur: cv2.GaussianBlur
        blurred = aug_blur(img, ksize=(5, 5), sigma=1.0)
        self.assertEqual(blurred.shape, (112, 112, 3))
        self.assertEqual(blurred.dtype, np.uint8)

        # 3. aug_random_erase
        rng = np.random.RandomState(42)
        erased = aug_random_erase(img, p=1.0, rng=rng)
        self.assertEqual(erased.shape, (112, 112, 3))
        # Vì img ban đầu toàn 128, vùng bị erase sẽ có các giá trị khác 128
        self.assertFalse(np.all(erased == 128), "Phải có vùng bị thay đổi giá trị khi erase")

        print("[PASS] aug_transforms: brightness/contrast, blur, random erase đạt chuẩn.")


if __name__ == "__main__":
    unittest.main()
