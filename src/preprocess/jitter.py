"""
Module Jitter Bounding Box (src/preprocess/jitter.py)
Tuân thủ nghiêm ngặt mục 4 của kế hoạch v4:
- Mở rộng mỗi cạnh ngẫu nhiên 0-20% và dịch tâm tối đa 10% kích thước bbox (kẹp trong ảnh).
- Áp dụng cho tập train với xác suất 0.7 để classifier quen với khung phát hiện của Detector.
"""

from typing import Tuple, Union, Optional, List
import numpy as np


def aug_bbox_jitter(
    bbox: Union[Tuple[int, int, int, int], List[int]],
    img_shape: Tuple[int, int],
    max_expand: float = 0.2,
    max_shift: float = 0.1,
    rng: Optional[np.random.RandomState] = None
) -> Tuple[int, int, int, int]:
    """
    Jitter bounding box cho tập train:
    Mở rộng ngẫu nhiên mỗi cạnh 0-20% và dịch tâm tối đa 10% kích thước bbox,
    kẹp an toàn trong kích thước ảnh gốc.
    """
    if rng is None:
        rng = np.random.RandomState()

    H, W = img_shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # Dịch tâm tối đa max_shift
    shift_x = rng.uniform(-max_shift, max_shift) * bw
    shift_y = rng.uniform(-max_shift, max_shift) * bh
    new_cx = cx + shift_x
    new_cy = cy + shift_y

    # Mở rộng ngẫu nhiên mỗi chiều (0 đến max_expand)
    expand_w = rng.uniform(0.0, max_expand) * bw
    expand_h = rng.uniform(0.0, max_expand) * bh
    new_w = bw + expand_w
    new_h = bh + expand_h

    # Kẹp trong [0, W], [0, H]
    nx1 = max(0, int(round(new_cx - new_w / 2.0)))
    ny1 = max(0, int(round(new_cy - new_h / 2.0)))
    nx2 = min(W, int(round(new_cx + new_w / 2.0)))
    ny2 = min(H, int(round(new_cy + new_h / 2.0)))

    if nx2 <= nx1:
        nx2 = min(W, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(H, ny1 + 1)

    return nx1, ny1, nx2, ny2
