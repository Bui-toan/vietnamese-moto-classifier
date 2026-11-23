"""
Module tiền xử lý và tăng cường dữ liệu (Augmentation) tự code 100% bằng NumPy.
Tuân thủ nghiêm ngặt ràng buộc của đề tài:
- letterbox: đệm bằng giá trị trung bình của ảnh đã resize (không dùng viền đen).
- aug_rotate: tự code bằng ánh xạ ngược (inverse mapping) + nội suy song tuyến (bilinear interpolation) với viền reflect. Không dùng cv2.warpAffine.
- aug_random_erase: tự code che vùng chữ nhật bằng NumPy.
- aug_brightness_contrast: np.clip(alpha * img + beta, 0, 255). Không dùng cv2.convertScaleAbs.
- aug_blur: dùng cv2.GaussianBlur (thuộc nhóm blur cơ bản được phép).
- aug_bbox_jitter: mở rộng 0-20% mỗi cạnh, dịch tâm tối đa 10% kích thước bbox, kẹp trong ảnh.
"""

from typing import Tuple, Optional, Union
import numpy as np
import cv2


def letterbox(img: np.ndarray, target_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    """
    Resize ảnh giữ tỷ lệ về target_size (H, W).
    Phần đệm (padding) sử dụng giá trị trung bình của ảnh đã resize,
    không dùng viền đen để tránh tạo cạnh gradient mạnh gây nhiễu cho HOG.
    
    Args:
        img: np.ndarray (H, W, C) hoặc (H, W) uint8.
        target_size: (target_h, target_w), mặc định (112, 112).
        
    Returns:
        np.ndarray kích thước (target_h, target_w, C) hoặc (target_h, target_w).
    """
    target_h, target_w = target_size
    orig_h, orig_w = img.shape[:2]
    
    # Tính scale giữ tỷ lệ
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    
    # Resize ảnh gốc theo scale
    # cv2.resize cơ bản được phép dùng cho phép co giãn hình học cơ bản
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Tính giá trị trung bình của ảnh đã resize để làm viền đệm
    if resized.ndim == 3:
        pad_val = resized.mean(axis=(0, 1)).astype(resized.dtype)
        out = np.empty((target_h, target_w, resized.shape[2]), dtype=resized.dtype)
        out[:] = pad_val
    else:
        pad_val = int(round(float(resized.mean())))
        out = np.full((target_h, target_w), pad_val, dtype=resized.dtype)
        
    # Tính tọa độ đặt ảnh vào giữa
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    
    out[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return out


def aug_rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Xoay ảnh quanh tâm bằng góc angle_deg (độ) ngược chiều kim đồng hồ.
    Tự code 100% bằng NumPy:
    - Sử dụng ánh xạ ngược (inverse mapping) từ tọa độ đích về tọa độ nguồn.
    - Nội suy song tuyến (bilinear interpolation).
    - Biên phản xạ (reflect mode): x < 0 hoặc x >= W được phản xạ đối xứng.
    KHÔNG dùng cv2.warpAffine.
    
    Args:
        img: np.ndarray (H, W, C) hoặc (H, W) uint8.
        angle_deg: Góc xoay theo độ.
        
    Returns:
        np.ndarray cùng kích thước và dtype với img.
    """
    if abs(angle_deg) < 1e-6:
        return img.copy()
        
    orig_shape = img.shape
    is_2d = (img.ndim == 2)
    if is_2d:
        img_3d = img[:, :, np.newaxis]
    else:
        img_3d = img

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

    # Ánh xạ ngược: quay góc -theta để tìm tọa độ nguồn (src_x, src_y)
    # R(-theta) = [[cos(theta), sin(theta)], [-sin(theta), cos(theta)]]
    src_x = cos_t * dx + sin_t * dy + cx
    src_y = -sin_t * dx + cos_t * dy + cy

    # Hàm ánh xạ biên reflect
    def reflect_coord(coord: np.ndarray, max_val: int) -> np.ndarray:
        # Đối xứng qua biên: ví dụ x in [-1, -2] -> [0, 1]
        # Xử lý tổng quát cho mảng numpy
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

    # Trọng số song tuyến
    wx = (src_x - x0)[:, :, np.newaxis]
    wy = (src_y - y0)[:, :, np.newaxis]

    # Phản xạ biên cho 4 điểm
    rx0 = reflect_coord(x0, W).astype(np.int64)
    rx1 = reflect_coord(x1, W).astype(np.int64)
    ry0 = reflect_coord(y0, H).astype(np.int64)
    ry1 = reflect_coord(y1, H).astype(np.int64)

    # Lấy giá trị pixel tại 4 đỉnh
    # img_3d shape (H, W, C)
    I00 = img_3d[ry0, rx0, :].astype(np.float64)
    I10 = img_3d[ry0, rx1, :].astype(np.float64)
    I01 = img_3d[ry1, rx0, :].astype(np.float64)
    I11 = img_3d[ry1, rx1, :].astype(np.float64)

    # Nội suy theo trục x rồi theo trục y
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
    """
    Augmentation che ngẫu nhiên một vùng chữ nhật (Random Erasing) tự code bằng NumPy.
    
    Args:
        img: np.ndarray (H, W, C) hoặc (H, W).
        p: Xác suất áp dụng.
        sl, sh: Tỉ lệ diện tích vùng che tối thiểu và tối đa so với ảnh.
        r1: Tỉ lệ khung hình (aspect ratio) tối thiểu (tối đa là 1/r1).
        rng: Bộ sinh số ngẫu nhiên numpy.
    """
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
            # Điền giá trị trung bình hoặc ngẫu nhiên
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
    """
    Điều chỉnh độ sáng và độ tương phản tự code bằng NumPy:
    np.clip(alpha * img + beta, 0, 255).
    KHÔNG dùng cv2.convertScaleAbs.
    
    Args:
        img: np.ndarray uint8.
        alpha: Hệ số tương phản (1.0 = giữ nguyên).
        beta: Hệ số độ sáng (0.0 = giữ nguyên).
    """
    res = alpha * img.astype(np.float32) + beta
    return np.clip(res, 0, 255).astype(np.uint8)


def aug_blur(
    img: np.ndarray,
    ksize: Tuple[int, int] = (5, 5),
    sigma: float = 1.0
) -> np.ndarray:
    """
    Làm mờ ảnh bằng cv2.GaussianBlur (hàm được phép dùng trong đề tài).
    """
    return cv2.GaussianBlur(img, ksize, sigma)


def aug_bbox_jitter(
    bbox: Union[Tuple[int, int, int, int], list],
    img_shape: Tuple[int, int],
    max_expand: float = 0.2,
    max_shift: float = 0.1,
    rng: Optional[np.random.RandomState] = None
) -> Tuple[int, int, int, int]:
    """
    Jitter bounding box cho tập train (xác suất 0.7 theo yêu cầu v4):
    Mở rộng ngẫu nhiên mỗi cạnh 0-20% và dịch tâm tối đa 10% kích thước bbox,
    kẹp trong kích thước ảnh gốc.
    
    Args:
        bbox: [x1, y1, x2, y2]
        img_shape: (H, W) của ảnh gốc
        max_expand: Tỉ lệ mở rộng tối đa (0.2 = 20%)
        max_shift: Tỉ lệ dịch tâm tối đa (0.1 = 10%)
        rng: np.random.RandomState
    
    Returns:
        (new_x1, new_y1, new_x2, new_y2) kiểu int đã kẹp trong ảnh.
    """
    if rng is None:
        rng = np.random.RandomState()

    H, W = img_shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # Dịch tâm tối đa max_shift * bw/bh
    shift_x = rng.uniform(-max_shift, max_shift) * bw
    shift_y = rng.uniform(-max_shift, max_shift) * bh
    new_cx = cx + shift_x
    new_cy = cy + shift_y

    # Mở rộng ngẫu nhiên mỗi chiều (0 đến max_expand)
    expand_w = rng.uniform(0.0, max_expand) * bw
    expand_h = rng.uniform(0.0, max_expand) * bh
    new_w = bw + expand_w
    new_h = bh + expand_h

    # Tính lại tọa độ và kẹp trong [0, W], [0, H]
    nx1 = max(0, int(round(new_cx - new_w / 2.0)))
    ny1 = max(0, int(round(new_cy - new_h / 2.0)))
    nx2 = min(W, int(round(new_cx + new_w / 2.0)))
    ny2 = min(H, int(round(new_cy + new_h / 2.0)))

    # Đảm bảo box hợp lệ
    if nx2 <= nx1:
        nx2 = min(W, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(H, ny1 + 1)

    return nx1, ny1, nx2, ny2
