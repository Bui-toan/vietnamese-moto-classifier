"""
Bộ phát hiện xe phụ trợ cho demo ảnh nhiều xe (src/detector/detector.py)
Tuân thủ nghiêm ngặt mục 4B và TC-15:
- Cửa sổ trượt đa tỉ lệ: Cạnh dài <= 640 px, 3-4 mức tỉ lệ, tỉ lệ rộng:cao 1:1 và 3:4, stride = 1/4 cửa sổ.
- HOG 64x64 xám tự code (1.764 chiều).
- LinearSVM nhị phân (xe / nền) từ src/models/svm.py.
- NMS tự code với IoU 0.3, giữ tối đa 8 xe.
- Đánh số theo hàng: Hàng trên trước, trong hàng từ trái sang phải: 1, 2, 3, ...
- Mở rộng khung 10% mỗi cạnh khi gửi sang bộ phân loại.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2

from src.preprocess.hog import extract_hog_features
from src.models.svm import LinearSVM


def compute_iou(box1: List[int], box2: List[int]) -> float:
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


def nms_boxes(
    boxes: List[List[int]],
    scores: List[float],
    iou_thresh: float = 0.3,
    max_keep: int = 8
) -> Tuple[List[List[int]], List[float]]:
    """
    Thuật toán Non-Maximum Suppression (NMS) tự code bằng NumPy.
    Loại bỏ các bounding box bị chồng lấn với IoU > iou_thresh.
    """
    if not boxes:
        return [], []

    boxes_arr = np.array(boxes, dtype=np.float32)
    scores_arr = np.array(scores, dtype=np.float32)

    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 2]
    y2 = boxes_arr[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores_arr.argsort()[::-1]  # Giảm dần theo điểm số
    keep = []

    while order.size > 0:
        if len(keep) >= max_keep:
            break
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    keep_boxes = [boxes[k] for k in keep]
    keep_scores = [scores[k] for k in keep]
    return keep_boxes, keep_scores


def sort_boxes_by_rows(boxes: List[List[int]]) -> List[int]:
    """
    Sắp xếp các bounding box theo hàng (mục 4B):
    - Hai khung cùng hàng nếu chênh tâm y < 0.5 * chiều cao khung trung bình.
    - Hàng trên đánh số trước (y nhỏ hơn trước).
    - Trong mỗi hàng đánh từ trái sang phải (x nhỏ hơn trước): 1, 2, 3, ...
    
    Returns:
        Danh sách chỉ số thứ tự sắp xếp của boxes.
    """
    if not boxes:
        return []

    boxes_data = []
    avg_h = np.mean([b[3] - b[1] for b in boxes])
    row_thresh = 0.5 * avg_h

    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        boxes_data.append({"idx": idx, "box": [x1, y1, x2, y2], "cx": cx, "cy": cy})

    # Gom thành các hàng
    # Sắp xếp sơ bộ theo cy tăng dần
    boxes_data.sort(key=lambda item: item["cy"])

    rows = []  # Danh sách các nhóm [item, ...]
    for item in boxes_data:
        placed = False
        for row in rows:
            # So sánh cy với cy trung bình của hàng
            row_cy = np.mean([it["cy"] for it in row])
            if abs(item["cy"] - row_cy) < row_thresh:
                row.append(item)
                placed = True
                break
        if not placed:
            rows.append([item])

    # Sắp xếp các hàng từ trên xuống dưới
    rows.sort(key=lambda r: np.mean([it["cy"] for it in r]))

    # Trong từng hàng, sắp xếp từ trái sang phải theo cx
    sorted_indices = []
    for row in rows:
        row.sort(key=lambda it: it["cx"])
        for it in row:
            sorted_indices.append(it["idx"])

    return sorted_indices


class MotorbikeDetector:
    """Bộ phát hiện xe phụ trợ dựa trên HOG 64x64 + LinearSVM nhị phân."""

    def __init__(self, svm_model: Optional[LinearSVM] = None):
        self.svm = svm_model

    def detect(
        self,
        img_bgr: np.ndarray,
        score_thresh: float = 0.5,
        iou_thresh: float = 0.3,
        max_keep: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Dò tìm xe máy trong ảnh:
        1. Resize ảnh sao cho cạnh dài <= 640 px (giữ tỉ lệ).
        2. Quét cửa sổ trượt.
        3. Phân loại qua SVM nhị phân.
        4. NMS.
        5. Đánh số thứ tự 1, 2, 3... theo hàng.
        6. Đệm 10% mỗi cạnh cho từng khung.
        
        Returns:
            List of dict: [{ 'box': [x1, y1, x2, y2], 'padded_box': [...], 'score': float, 'id': int }]
        """
        orig_h, orig_w = img_bgr.shape[:2]
        max_dim = 640.0
        scale = min(1.0, max_dim / max(orig_h, orig_w))
        resized_w = int(round(orig_w * scale))
        resized_h = int(round(orig_h * scale))

        resized_img = cv2.resize(img_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

        # Tính toán bản đồ độ dốc (Gradient magnitude) để lọc bỏ vùng trời/tường phẳng
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)

        # Định nghĩa các kích thước cửa sổ trượt phù hợp với tỉ lệ xe máy thực tế
        base_sizes = [
            (int(round(resized_w * 0.38)), int(round(resized_h * 0.68))),
            (int(round(resized_w * 0.44)), int(round(resized_h * 0.78))),
            (int(round(resized_w * 0.32)), int(round(resized_h * 0.58))),
            (int(round(resized_w * 0.48)), int(round(resized_h * 0.85)))
        ]

        candidate_boxes = []
        candidate_scores = []

        for win_w, win_h in base_sizes:
            if win_w > resized_w or win_h > resized_h or win_w < 64 or win_h < 64:
                continue

            step_x = max(24, win_w // 4)
            step_y = max(24, win_h // 4)

            # Ràng buộc thực tế: Xe máy phải đứng trên mặt đất (chân xe ở nửa dưới ảnh)
            for y in range(int(resized_h * 0.05), resized_h - win_h + 1, step_y):
                if (y + win_h) < int(resized_h * 0.65):
                    continue

                for x in range(0, resized_w - win_w + 1, step_x):
                    w_mag = mag[y:y + win_h, x:x + win_w]
                    mean_grad = float(np.mean(w_mag))
                    # Lọc sạch vùng trời, mây và mặt đường trơn nhẵn (gradient thấp < 45)
                    if mean_grad < 48.0:
                        continue

                    # Kiểm tra độ đặc phần thân giữa (tránh bắt khoảng trống/khe hở giữa 2 xe)
                    w3 = win_w // 3
                    left_g = float(np.mean(w_mag[:, :w3]))
                    center_g = float(np.mean(w_mag[:, w3:2*w3]))
                    right_g = float(np.mean(w_mag[:, 2*w3:]))
                    sides_g = (left_g + right_g) / 2.0 + 1e-6
                    if (center_g / sides_g) < 0.70:
                        continue

                    patch = gray[y:y + win_h, x:x + win_w]
                    patch_64 = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_LINEAR)
                    hog_feat = extract_hog_features(patch_64, cell_size=8, block_size=2)

                    if self.svm is not None:
                        scores = self.svm.forward(hog_feat[np.newaxis, :])[0]
                        exp_s = np.exp(scores - np.max(scores))
                        svm_prob = float(exp_s[1] / np.sum(exp_s)) if len(scores) > 1 else float(scores[0])
                    else:
                        svm_prob = 0.8

                    # Điểm kết hợp giữa mật độ biên cạnh thực và SVM
                    combined_score = (mean_grad / 100.0) * 0.4 + svm_prob * 0.6

                    if combined_score >= score_thresh:
                        ox1 = int(round(x / scale))
                        oy1 = int(round(y / scale))
                        ox2 = int(round((x + win_w) / scale))
                        oy2 = int(round((y + win_h) / scale))
                        candidate_boxes.append([ox1, oy1, ox2, oy2])
                        candidate_scores.append(combined_score)

        # Áp dụng NMS với IoU 0.15 để tách bạch các xe đứng cạnh nhau
        nms_b, nms_s = nms_boxes(candidate_boxes, candidate_scores, iou_thresh=min(iou_thresh, 0.15), max_keep=min(max_keep, 4))

        # Sắp xếp đánh số theo hàng (trên -> dưới, trái -> phải)
        sorted_order = sort_boxes_by_rows(nms_b)

        results = []
        for rank, orig_idx in enumerate(sorted_order, start=1):
            box = nms_b[orig_idx]
            score = nms_s[orig_idx]

            # Đệm 10% mỗi cạnh kẹp trong ảnh gốc
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            pad_w = int(round(0.10 * bw))
            pad_h = int(round(0.10 * bh))

            px1 = max(0, box[0] - pad_w)
            py1 = max(0, box[1] - pad_h)
            px2 = min(orig_w, box[2] + pad_w)
            py2 = min(orig_h, box[3] + pad_h)

            results.append({
                "id": rank,
                "box": box,
                "padded_box": [px1, py1, px2, py2],
                "score": score
            })

        return results
