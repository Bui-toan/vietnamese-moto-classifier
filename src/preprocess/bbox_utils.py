"""
Module tiện ích Bounding Box (src/preprocess/bbox_utils.py)
Cung cấp các hàm tính toán hình học dùng chung cho Detector, Classifier và Jitter:
- compute_iou: Tính IoU giữa 2 bounding box.
- clip_bbox: Kẹp tọa độ box trong giới hạn ảnh.
- crop_bbox: Cắt vùng ảnh theo bbox có tùy chọn đệm (padding).
"""

from typing import List, Tuple, Union
import numpy as np


def clip_bbox(
    bbox: Union[List[int], Tuple[int, int, int, int]],
    img_shape: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """Kẹp tọa độ bounding box [x1, y1, x2, y2] trong kích thước ảnh (H, W)."""
    H, W = img_shape[:2]
    x1, y1, x2, y2 = bbox
    nx1 = max(0, min(W - 1, int(round(x1))))
    ny1 = max(0, min(H - 1, int(round(y1))))
    nx2 = max(nx1 + 1, min(W, int(round(x2))))
    ny2 = max(ny1 + 1, min(H, int(round(y2))))
    return nx1, ny1, nx2, ny2


def compute_iou(
    box1: Union[List[int], Tuple[int, int, int, int]],
    box2: Union[List[int], Tuple[int, int, int, int]]
) -> float:
    """Tính Intersection over Union (IoU) giữa 2 bounding box [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def crop_bbox(
    img: np.ndarray,
    bbox: Union[List[int], Tuple[int, int, int, int]],
    pad_ratio: float = 0.0
) -> np.ndarray:
    """
    Cắt vùng ảnh theo bounding box kèm mở rộng pad_ratio (ví dụ 0.1 = 10% mỗi cạnh).
    """
    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    if pad_ratio > 0:
        pad_w = int(round(bw * pad_ratio))
        pad_h = int(round(bh * pad_ratio))
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(W, x2 + pad_w)
        y2 = min(H, y2 + pad_h)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return img.copy()
    return crop
