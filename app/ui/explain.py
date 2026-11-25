"""
Module trực quan hóa và giải thích mô hình (app/ui/explain.py)
Tuân thủ nghiêm ngặt mục 7.2 của kế hoạch v4:
- Grad-CAM cho M2 (CNN-1) và M3 (CNN-2).
- Bản đồ đặc trưng HOG cho M1.
"""

from typing import Tuple, Optional
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from src.preprocess.hog import compute_gradients, compute_cell_histograms


def generate_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None
) -> np.ndarray:
    """
    Tạo bản đồ nhiệt Grad-CAM cho mạng CNN-1 / CNN-2.
    
    Args:
        model: CNN1 hoặc CNN2 có tích hợp hook activations & gradients.
        input_tensor: Tensor shape (1, 3, 112, 112) requires_grad=True.
        target_class: Chỉ số lớp cần giải thích (mặc định lớp có logit cao nhất).
        
    Returns:
        np.ndarray (112, 112, 3) uint8 RGB bản đồ nhiệt.
    """
    model.eval()
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    logits = model(input_tensor)

    if target_class is None:
        target_class = int(logits.argmax(dim=1).item())

    # Backward theo logit của target class
    model.zero_grad()
    score = logits[0, target_class]
    score.backward()

    # Lấy gradients và activations từ hook
    gradients = model.gradients  # (1, C, H, W)
    activations = model.activations  # (1, C, H, W)

    if gradients is None or activations is None:
        # Fallback nếu hook không kích hoạt
        return np.full((112, 112, 3), 128, dtype=np.uint8)

    # GAP trên gradients để lấy trọng số alpha_k
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

    # Tích chập trọng số với feature maps
    cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H, W)
    cam = F.relu(cam)  # Chỉ giữ các đóng góp dương

    cam_np = cam.squeeze().detach().cpu().numpy()
    if np.max(cam_np) > 0:
        cam_np = cam_np / np.max(cam_np)
    else:
        cam_np = np.zeros_like(cam_np)

    # Resize về kích thước của input_tensor
    in_h, in_w = input_tensor.shape[2], input_tensor.shape[3]
    cam_resized = cv2.resize((cam_np * 255).astype(np.uint8), (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return heatmap_rgb


def visualize_hog_features(img_112_rgb: np.ndarray) -> np.ndarray:
    """
    Trực quan hóa bản đồ hướng HOG cho mô hình M1.
    Vẽ các đoạn thẳng đại diện cho hướng gradient mạnh nhất tại mỗi cell.
    """
    gray = cv2.cvtColor(img_112_rgb, cv2.COLOR_RGB2GRAY).astype(float)
    mag, angle = compute_gradients(gray)
    cell_hists = compute_cell_histograms(mag, angle, cell_size=8, num_bins=9)  # (14, 14, 9)

    out_viz = np.zeros((112, 112, 3), dtype=np.uint8)
    bin_centers = np.array([10, 30, 50, 70, 90, 110, 130, 150, 170])

    for cy in range(14):
        for cx in range(14):
            hist = cell_hists[cy, cx]
            max_bin = np.argmax(hist)
            val = hist[max_bin]

            if val > 1e-3:
                theta = np.deg2rad(bin_centers[max_bin])
                center_x = cx * 8 + 4
                center_y = cy * 8 + 4

                # Độ dài đường vẽ tỉ lệ với độ lớn gradient
                length = min(3.5, 3.5 * (val / (np.max(cell_hists) + 1e-5)))
                dx = int(round(length * np.cos(theta)))
                dy = int(round(length * np.sin(theta)))

                cv2.line(
                    out_viz,
                    (center_x - dx, center_y - dy),
                    (center_x + dx, center_y + dy),
                    (20, 184, 166),  # Teal
                    1,
                    cv2.LINE_AA
                )

    return out_viz
