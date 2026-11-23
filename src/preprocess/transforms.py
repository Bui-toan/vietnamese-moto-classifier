"""
Module biến đổi hình học và tăng cường dữ liệu (src/preprocess/transforms.py)
Tuân thủ nghiêm ngặt ràng buộc của đề tài (TC-03):
- aug_rotate: tự code bằng ánh xạ ngược (inverse mapping) + nội suy song tuyến (bilinear interpolation) với viền reflect. KHÔNG dùng cv2.warpAffine.
- aug_random_erase: tự code che vùng chữ nhật bằng NumPy.
- aug_brightness_contrast: np.clip(alpha * img + beta, 0, 255). KHÔNG dùng cv2.convertScaleAbs.
- aug_blur: dùng cv2.GaussianBlur (thuộc nhóm blur cơ bản được phép).
"""

from typing import Tuple, Optional
import numpy as np
import cv2


def aug_rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Xoay ảnh quanh tâm bằng góc angle_deg (độ) ngược chiều kim đồng hồ.
    Tự code 100% bằng NumPy:
    - Ánh xạ ngược (inverse mapping) từ tọa độ đích về tọa độ nguồn.
    - Nội suy song tuyến (bilinear interpolation).
    - Biên phản xạ (reflect mode).
    KHÔNG dùng cv2.warpAffine.
    """
    if abs(angle_deg) < 1e-6:
        return img.copy()

    orig_shape = img.shape
    is_2d = (img.ndim == 2)
    img_3d = img[:, :, np.newaxis] if is_2d else img

    H, W, C = img_3d.shape
    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    # Lưới tọa độ ảnh đích
    grid_y, grid_x = np.indices((H, W), dtype=np.float64)

    # Dịch tâm về (0,0)
    dx = grid_x - cx
    dy = grid_y - cy

    # Ánh xạ ngược R(-theta)
    src_x = cos_t * dx + sin_t * dy + cx
    src_y = -sin_t * dx + cos_t * dy + cy

    # Hàm ánh xạ biên reflect
    def reflect_coord(coord: np.ndarray, max_val: int) -> np.ndarray:
        c = coord.copy()
        period = 2 * (max_val - 1)
        if period <= 0:
            return np.zeros_like(c)
        c = np.abs(c)
        c = c % period
        mask = c >= max_val
        c[mask] = period - c[mask]
        return c

    # Lấy 4 điểm lân cận
    x0 = np.floor(src_x).astype(np.int64)
    x1 = x0 + 1
    y0 = np.floor(src_y).astype(np.int64)
    y1 = y0 + 1

    wx = (src_x - x0)[:, :, np.newaxis]
    wy = (src_y - y0)[:, :, np.newaxis]

    rx0 = reflect_coord(x0, W).astype(np.int64)
    rx1 = reflect_coord(x1, W).astype(np.int64)
    ry0 = reflect_coord(y0, H).astype(np.int64)
    ry1 = reflect_coord(y1, H).astype(np.int64)

    I00 = img_3d[ry0, rx0, :].astype(np.float64)
    I10 = img_3d[ry0, rx1, :].astype(np.float64)
    I01 = img_3d[ry1, rx0, :].astype(np.float64)
    I11 = img_3d[ry1, rx1, :].astype(np.float64)

    top = I00 * (1.0 - wx) + I10 * wx
    bottom = I01 * (1.0 - wx) + I11 * wx
    out = top * (1.0 - wy) + bottom * wy

    out = np.clip(np.round(out), 0, 255).astype(img.dtype)
    if is_2d:
        return out[:, :, 0]
    return out


def aug_random_erase(
    img: np.ndarray,
    p: float = 0.5,
    sl: float = 0.02,
    sh: float = 0.33,
    r1: float = 0.3,
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """Che ngẫu nhiên một vùng chữ nhật bằng NumPy."""
    if rng is None:
        rng = np.random.RandomState()

    if rng.rand() > p:
        return img.copy()

    out = img.copy()
    H, W = out.shape[:2]
    area = H * W

    for _ in range(10):
        target_area = rng.uniform(sl, sh) * area
        aspect_ratio = rng.uniform(r1, 1.0 / r1)

        h = int(round(np.sqrt(target_area * aspect_ratio)))
        w = int(round(np.sqrt(target_area / aspect_ratio)))

        if h < H and w < W:
            y1 = rng.randint(0, H - h)
            x1 = rng.randint(0, W - w)
            if out.ndim == 3:
                val = rng.randint(0, 256, (h, w, out.shape[2])).astype(out.dtype)
            else:
                val = rng.randint(0, 256, (h, w)).astype(out.dtype)
            out[y1:y1 + h, x1:x1 + w] = val
            break

    return out


def aug_brightness_contrast(
    img: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0
) -> np.ndarray:
    """Điều chỉnh độ sáng/tương phản bằng np.clip(alpha * img + beta, 0, 255)."""
    res = alpha * img.astype(np.float32) + beta
    return np.clip(res, 0, 255).astype(np.uint8)


def aug_blur(
    img: np.ndarray,
    ksize: Tuple[int, int] = (5, 5),
    sigma: float = 1.0
) -> np.ndarray:
    """Làm mờ ảnh bằng cv2.GaussianBlur (được phép dùng)."""
    return cv2.GaussianBlur(img, ksize, sigma)
