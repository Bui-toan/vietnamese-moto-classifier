"""
Module nạp và tiền xử lý dữ liệu cho quá trình huấn luyện (src/train/dataset.py)
Tuân thủ nghiêm ngặt mục 4 của kế hoạch v4:
- Thứ tự xử lý: Đọc ảnh -> crop bbox -> augment (train) -> letterbox 112x112 -> normalize.
- Jitter bbox (aug_bbox_jitter) với xác suất 0.7 trên tập train.
- Toàn bộ phép xoay, che, sáng/tối tự code bằng NumPy.
- Letterbox đệm giá trị trung bình (không dùng viền đen).
"""

import os
from typing import Optional, Tuple, List, Dict, Any, Union
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd

from src.preprocess.transform import (
    letterbox,
    aug_rotate,
    aug_random_erase,
    aug_brightness_contrast,
    aug_blur,
    aug_bbox_jitter
)
from src.preprocess.hog import extract_hog_features


CLASS_TO_IDX = {
    "Honda Wave": 0,
    "Honda Vision": 1,
    "Honda Air Blade": 2,
    "Yamaha Exciter": 3,
    "Piaggio Vespa": 4,
    "Honda SH": 5
}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


class MotorbikeDataset(Dataset):
    """
    PyTorch Dataset cho bài toán phân loại dòng xe máy phổ biến Việt Nam.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        is_train: bool = True,
        img_size: Tuple[int, int] = (112, 112),
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        use_hog: bool = False,
        seed: int = 42
    ):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.img_size = img_size
        self.use_hog = use_hog
        self.rng = np.random.RandomState(seed)

        # Chuẩn hóa Z-score
        self.mean = mean if mean is not None else np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = std if std is not None else np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Union[torch.Tensor, np.ndarray], int]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label_str = row["label"]
        label = CLASS_TO_IDX.get(label_str, -1)  # -1 nếu OOD

        # 1. Đọc ảnh
        if not os.path.exists(img_path):
            # Ảnh giả lập nếu không tìm thấy trên đĩa (phục vụ test)
            img = np.full((112, 112, 3), 128, dtype=np.uint8)
        else:
            img = cv2.imread(img_path)
            if img is None:
                img = np.full((112, 112, 3), 128, dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        H, W = img.shape[:2]

        # 2. Xử lý bbox nếu có
        bbox = None
        if "bbox" in row and pd.notna(row["bbox"]):
            try:
                # Format: "x1,y1,x2,y2" hoặc list
                b_str = str(row["bbox"]).replace("[", "").replace("]", "")
                parts = [int(float(p.strip())) for p in b_str.split(",")]
                if len(parts) == 4:
                    bbox = parts
            except Exception:
                bbox = None

        if bbox is not None:
            if self.is_train and self.rng.rand() < 0.7:
                # Jitter bbox xác suất 0.7 theo v4
                bbox = aug_bbox_jitter(bbox, (H, W), max_expand=0.2, max_shift=0.1, rng=self.rng)
            x1, y1, x2, y2 = bbox
            img = img[y1:y2, x1:x2]
            if img.size == 0:
                img = np.full((112, 112, 3), 128, dtype=np.uint8)

        # 3. Augmentation nếu là tập train
        if self.is_train:
            # Xoay tự code [-15, 15] độ
            if self.rng.rand() < 0.5:
                angle = self.rng.uniform(-15.0, 15.0)
                img = aug_rotate(img, angle)

            # Sáng tối tự code
            if self.rng.rand() < 0.5:
                alpha = self.rng.uniform(0.7, 1.3)
                beta = self.rng.uniform(-25.0, 25.0)
                img = aug_brightness_contrast(img, alpha=alpha, beta=beta)

            # Blur
            if self.rng.rand() < 0.3:
                img = aug_blur(img, ksize=(5, 5), sigma=self.rng.uniform(0.5, 1.5))

            # Che vùng chữ nhật
            if self.rng.rand() < 0.4:
                img = aug_random_erase(img, p=1.0, rng=self.rng)

        # 4. Letterbox 112x112 với đệm trung bình
        img_lb = letterbox(img, target_size=self.img_size)

        # 5. Output
        if self.use_hog:
            # Trích xuất vector HOG 6.084 chiều
            hog_feat = extract_hog_features(img_lb, cell_size=8, block_size=2)
            # Z-score normalize HOG
            hog_feat = (hog_feat - self.mean) / (self.std + 1e-6)
            return hog_feat.astype(np.float32), label
        else:
            # Chuyển về Tensor CHW float32
            img_f = img_lb.astype(np.float32) / 255.0
            # Z-score normalize theo kênh
            img_f = (img_f - self.mean) / (self.std + 1e-6)
            tensor_chw = torch.from_numpy(img_f.transpose(2, 0, 1)).float()
            return tensor_chw, label
