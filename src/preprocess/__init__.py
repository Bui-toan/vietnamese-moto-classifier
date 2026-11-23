"""
Package tiền xử lý và đặc trưng hình ảnh cho đề tài.
Tách nhỏ theo cấu trúc Clean Architecture:
- bbox_utils: Tiện ích bounding box (IoU, crop, clip).
- letterbox: Resize giữ tỷ lệ, đệm giá trị trung bình.
- transforms: Biến đổi hình học & màu sắc (xoay song tuyến, che, sáng/tối, blur).
- jitter: Jitter bbox cho tập train classifier.
- normalize: Chuẩn hóa Z-score mean/std.
- hog: Trích xuất đặc trưng HOG tự code 6.084 chiều.
"""

from src.preprocess.bbox_utils import compute_iou, crop_bbox, clip_bbox
from src.preprocess.letterbox import letterbox
from src.preprocess.transforms import aug_rotate, aug_random_erase, aug_brightness_contrast, aug_blur
from src.preprocess.jitter import aug_bbox_jitter
from src.preprocess.normalize import compute_stats_zscore, apply_zscore, save_stats, load_stats
from src.preprocess.hog import extract_hog_features, compute_gradients, compute_cell_histograms

__all__ = [
    "compute_iou",
    "crop_bbox",
    "clip_bbox",
    "letterbox",
    "aug_rotate",
    "aug_random_erase",
    "aug_brightness_contrast",
    "aug_blur",
    "aug_bbox_jitter",
    "compute_stats_zscore",
    "apply_zscore",
    "save_stats",
    "load_stats",
    "extract_hog_features",
    "compute_gradients",
    "compute_cell_histograms",
]
