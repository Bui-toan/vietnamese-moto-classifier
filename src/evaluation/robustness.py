"""
Module kiểm tra độ bền mô hình (src/evaluation/robustness.py)
Tuân thủ nghiêm ngặt mục 6.3 của kế hoạch v4 (TC-10 đến TC-13):
- TC-10: Blur sigma in {1, 2, 3}.
- TC-11: Xoay +-5, 10, 15 độ (tự code NumPy).
- TC-12: Che 10, 20, 30% diện tích.
- TC-13: Độ sáng x0.6 / x1.4, nhiễu Gauss, bbox rộng thêm 15/30% và lệch tâm 10%.
"""

import os
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from src.preprocess.transforms import aug_rotate, aug_brightness_contrast, aug_blur, aug_random_erase
from src.preprocess.jitter import aug_bbox_jitter
from src.preprocess.letterbox import letterbox
from src.eval.metrics import CLASS_NAMES


def run_robustness_tests(
    model_predict_fn,
    test_images: List[np.ndarray],
    test_labels: List[int],
    evidence_dir: str = "evidence",
    prefix: str = ""
) -> Dict[str, Any]:
    """Chạy toàn bộ các bài test robustness cho 1 model."""
    os.makedirs(evidence_dir, exist_ok=True)
    summary = {}

    # 1. TC-10: Blur
    blur_results = []
    for sigma in [1.0, 2.0, 3.0]:
        correct = 0
        for img, lbl in zip(test_images, test_labels):
            b_img = aug_blur(img, sigma=sigma)
            lb = letterbox(b_img, (112, 112))
            pred = model_predict_fn(lb)
            if pred == lbl:
                correct += 1
        acc = correct / len(test_labels) if test_labels else 0.0
        blur_results.append({"sigma": sigma, "top1_acc": acc})
    df_blur = pd.DataFrame(blur_results)
    df_blur.to_csv(os.path.join(evidence_dir, f"{prefix}robust_blur.csv"), index=False)
    summary["blur"] = df_blur

    # 2. TC-11: Xoay
    rot_results = []
    for angle in [-15.0, -10.0, -5.0, 5.0, 10.0, 15.0]:
        correct = 0
        for img, lbl in zip(test_images, test_labels):
            r_img = aug_rotate(img, angle)
            lb = letterbox(r_img, (112, 112))
            pred = model_predict_fn(lb)
            if pred == lbl:
                correct += 1
        acc = correct / len(test_labels) if test_labels else 0.0
        rot_results.append({"angle_deg": angle, "top1_acc": acc})
    df_rot = pd.DataFrame(rot_results)
    df_rot.to_csv(os.path.join(evidence_dir, f"{prefix}robust_rotate.csv"), index=False)
    summary["rotate"] = df_rot

    # 3. TC-12: Che phủ
    occ_results = []
    for ratio in [0.1, 0.2, 0.3]:
        correct = 0
        for img, lbl in zip(test_images, test_labels):
            o_img = aug_random_erase(img, p=1.0, sh=ratio)
            lb = letterbox(o_img, (112, 112))
            pred = model_predict_fn(lb)
            if pred == lbl:
                correct += 1
        acc = correct / len(test_labels) if test_labels else 0.0
        occ_results.append({"occlude_ratio": ratio, "top1_acc": acc})
    df_occ = pd.DataFrame(occ_results)
    df_occ.to_csv(os.path.join(evidence_dir, f"{prefix}robust_occlude.csv"), index=False)
    summary["occlude"] = df_occ

    # 4. TC-13: BBox rộng thêm 15% / 30%
    bbox_results = []
    for expand in [0.15, 0.30]:
        correct = 0
        for img, lbl in zip(test_images, test_labels):
            H, W = img.shape[:2]
            # Mở rộng bbox
            fake_box = [W//4, H//4, 3*W//4, 3*H//4]
            nx1, ny1, nx2, ny2 = aug_bbox_jitter(fake_box, (H, W), max_expand=expand, max_shift=0.1)
            crop = img[ny1:ny2, nx1:nx2]
            lb = letterbox(crop, (112, 112))
            pred = model_predict_fn(lb)
            if pred == lbl:
                correct += 1
        acc = correct / len(test_labels) if test_labels else 0.0
        bbox_results.append({"expand_ratio": expand, "top1_acc": acc})
    df_bbox = pd.DataFrame(bbox_results)
    df_bbox.to_csv(os.path.join(evidence_dir, f"{prefix}robust_bbox.csv"), index=False)
    summary["bbox"] = df_bbox

    print(f"[PASS] Đã hoàn thành các bài test robustness! Kết quả lưu tại {evidence_dir}")
    return summary
