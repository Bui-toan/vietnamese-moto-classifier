"""
Module vẽ bounding box và số thứ tự lên ảnh (app/ui/draw.py)
Tuân thủ nghiêm ngặt mục 4B và 7.2 của kế hoạch v4:
- Vẽ khung chữ nhật bo góc hoặc viền nổi.
- Vẽ số thứ tự lớn ở góc trên bên trái của khung: 1, 2, 3, ...
- Màu sắc trực quan (Teal #14B8A6, Cam #F59E0B).
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import cv2


def draw_detection_boxes(
    img_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    highlight_idx: int = -1
) -> np.ndarray:
    """
    Vẽ các khung xe đã phát hiện và số thứ tự lên ảnh.
    
    Args:
        img_bgr: Ảnh gốc OpenCV BGR.
        detections: Danh sách các dict [{'id': 1, 'box': [x1, y1, x2, y2], 'label': '...', 'prob': 0.95}, ...]
        highlight_idx: Chỉ số xe được tô sáng (1-based), mặc định -1 (tất cả như nhau).
        
    Returns:
        Ảnh BGR đã vẽ khung.
    """
    out_img = img_bgr.copy()
    H, W = out_img.shape[:2]

    # Màu sắc BGR:
    # Teal: #14B8A6 -> BGR: (166, 184, 20)
    # Highlight Cam: #F59E0B -> BGR: (11, 158, 245)
    default_color = (166, 184, 20)
    highlight_color = (11, 158, 245)

    for det in detections:
        box_id = det["id"]
        x1, y1, x2, y2 = det["box"]
        is_highlight = (box_id == highlight_idx)

        color = highlight_color if is_highlight else default_color
        thickness = 3 if is_highlight else 2

        # 1. Vẽ khung chữ nhật
        cv2.rectangle(out_img, (x1, y1), (x2, y2), color, thickness)

        # 2. Vẽ badge số thứ tự ở góc trên bên trái
        text = str(box_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thick = 2
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thick)

        badge_w = tw + 14
        badge_h = th + 12
        by1 = max(0, y1 - badge_h)
        by2 = by1 + badge_h
        bx1 = x1
        bx2 = min(W, x1 + badge_w)

        # Nền cho số thứ tự
        cv2.rectangle(out_img, (bx1, by1), (bx2, by2), color, -1)
        # Chữ số màu tối
        text_color = (11, 18, 32)  # #0B1220
        cv2.putText(out_img, text, (bx1 + 7, by2 - 6), font, font_scale, text_color, font_thick, cv2.LINE_AA)

    return out_img
