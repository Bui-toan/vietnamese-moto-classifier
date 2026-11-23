"""
Module Letterbox (src/preprocess/letterbox.py)
Tuân thủ nghiêm ngặt ràng buộc của đề tài (TC-01):
- Resize ảnh giữ tỷ lệ về target_size (H, W).
- Phần đệm (padding) sử dụng giá trị trung bình (mean) của ảnh đã resize.
- Tuyệt đối không dùng viền đen để tránh tạo cạnh gradient mạnh gây nhiễu cho HOG.
"""

from typing import Tuple
import numpy as np
import cv2


def letterbox(img: np.ndarray, target_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    """
    Resize giữ tỉ lệ ảnh và đệm bằng giá trị trung bình.
    
    Args:
        img: np.ndarray (H, W, C) hoặc (H, W).
        target_size: (target_h, target_w), mặc định (112, 112).
        
    Returns:
        np.ndarray kích thước target_size cùng dtype với img.
    """
    target_h, target_w = target_size
    orig_h, orig_w = img.shape[:2]

    # Tính scale giữ tỷ lệ
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    # Resize cơ bản
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Tính giá trị trung bình của ảnh đã resize để làm viền đệm
    if resized.ndim == 3:
        pad_val = resized.mean(axis=(0, 1)).astype(resized.dtype)
        out = np.empty((target_h, target_w, resized.shape[2]), dtype=resized.dtype)
        out[:] = pad_val
    else:
        pad_val = int(round(float(resized.mean())))
        out = np.full((target_h, target_w), pad_val, dtype=resized.dtype)

    # Đặt ảnh vào giữa
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2

    out[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return out
