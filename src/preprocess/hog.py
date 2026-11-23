"""
Module trích xuất đặc trưng HOG (Histogram of Oriented Gradients) tự code 100% bằng NumPy.
Tuân thủ nghiêm ngặt ràng buộc của đề tài:
- Không dùng cv2.HOGDescriptor hay skimage.feature.hog.
- Cell 8x8, 9 bin hướng (0 - 180 độ).
- Block 2x2 cell (stride 1 cell), chuẩn hóa L2-Hys.
- Ảnh 112x112 -> 6.084 chiều.
- Ảnh 64x64 (detector) -> 1.764 chiều.
- Báo lỗi rõ ràng với ảnh không đủ kích thước block (ví dụ 8x8).
"""

from typing import Tuple, Union
import numpy as np


def compute_gradients(gray_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính gradient Gx, Gy bằng bộ lọc [-1, 0, 1].
    Tự code bằng phép toán chỉ số NumPy.
    
    Args:
        gray_img: np.ndarray float64 (H, W).
        
    Returns:
        (magnitude, orientation_deg)
        orientation_deg trong khoảng [0, 180).
    """
    H, W = gray_img.shape
    gx = np.zeros_like(gray_img)
    gy = np.zeros_like(gray_img)

    # Gradient theo trục X: [-1, 0, 1]
    if W > 2:
        gx[:, 1:-1] = gray_img[:, 2:] - gray_img[:, :-2]
        gx[:, 0] = gray_img[:, 1] - gray_img[:, 0]
        gx[:, -1] = gray_img[:, -1] - gray_img[:, -2]
    elif W == 2:
        gx[:, 0] = gray_img[:, 1] - gray_img[:, 0]
        gx[:, 1] = gx[:, 0]

    # Gradient theo trục Y: [-1, 0, 1]^T
    if H > 2:
        gy[1:-1, :] = gray_img[2:, :] - gray_img[:-2, :]
        gy[0, :] = gray_img[1, :] - gray_img[0, :]
        gy[-1, :] = gray_img[-1, :] - gray_img[-2, :]
    elif H == 2:
        gy[0, :] = gray_img[1, :] - gray_img[0, :]
        gy[1, :] = gy[0, :]

    mag = np.sqrt(gx ** 2 + gy ** 2)
    
    # Góc định hướng theo độ: arctan2(gy, gx) trả về [-pi, pi]
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180.0  # Chuyển về [0, 180)
    
    return mag, angle


def compute_cell_histograms(
    mag: np.ndarray,
    angle: np.ndarray,
    cell_size: int = 8,
    num_bins: int = 9
) -> np.ndarray:
    """
    Tích lũy gradient magnitude vào các bin góc của từng cell.
    Sử dụng phân bổ tuyến tính giữa 2 bin góc lân cận (soft voting).
    
    Args:
        mag: Độ lớn gradient (H, W).
        angle: Góc pha [0, 180) (H, W).
        cell_size: Kích thước cell (mặc định 8x8).
        num_bins: Số bin hướng (mặc định 9 bin, 20 độ/bin).
        
    Returns:
        cell_histograms: (n_cells_y, n_cells_x, num_bins)
    """
    H, W = mag.shape
    n_cells_y = H // cell_size
    n_cells_x = W // cell_size
    
    bin_width = 180.0 / num_bins  # 20 độ
    
    # Chỉ số bin thực: bin_idx_float = (angle / bin_width) - 0.5
    # Trung tâm các bin là 10, 30, 50, ..., 170
    bin_center_shifted = (angle / bin_width) - 0.5
    b0 = np.floor(bin_center_shifted).astype(int)
    b1 = b0 + 1
    
    # Trọng số nội suy giữa 2 bin
    wb1 = bin_center_shifted - b0
    wb0 = 1.0 - wb1
    
    # Xử lý vòng tròn (wrap around) cho góc
    b0 = b0 % num_bins
    b1 = b1 % num_bins

    # Tạo mảng lưu histogram của các cell
    histograms = np.zeros((n_cells_y, n_cells_x, num_bins), dtype=np.float64)

    # Tích lũy cho từng cell
    for cy in range(n_cells_y):
        y_start = cy * cell_size
        y_end = y_start + cell_size
        for cx in range(n_cells_x):
            x_start = cx * cell_size
            x_end = x_start + cell_size

            c_mag = mag[y_start:y_end, x_start:x_end]
            c_b0 = b0[y_start:y_end, x_start:x_end]
            c_b1 = b1[y_start:y_end, x_start:x_end]
            c_wb0 = wb0[y_start:y_end, x_start:x_end]
            c_wb1 = wb1[y_start:y_end, x_start:x_end]

            # Bỏ phiếu vào bin b0 và b1
            vote0 = c_mag * c_wb0
            vote1 = c_mag * c_wb1

            # Tích lũy nhanh bằng np.bincount
            hist = np.bincount(c_b0.ravel(), weights=vote0.ravel(), minlength=num_bins)
            hist += np.bincount(c_b1.ravel(), weights=vote1.ravel(), minlength=num_bins)
            
            histograms[cy, cx, :] = hist

    return histograms


def extract_hog_features(
    img: np.ndarray,
    cell_size: int = 8,
    block_size: int = 2,
    num_bins: int = 9,
    clip_val: float = 0.2,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Trích xuất vector đặc trưng HOG tự code 100% bằng NumPy.
    
    Quy tắc kiểm tra:
    - Ảnh phải có kích thước tối thiểu bằng block_size * cell_size.
      Với cell=8, block=2 thì kích thước tối thiểu là 16x16.
      Nếu ảnh < 16x16 (ví dụ 8x8), raise ValueError với thông báo rõ ràng (phục vụ TC-02).
      
    Args:
        img: Ảnh đầu vào (H, W) hoặc (H, W, C) uint8 hoặc float.
        cell_size: Số pixel mỗi cell (mặc định 8).
        block_size: Số cell mỗi block (mặc định 2).
        num_bins: Số bin hướng (mặc định 9).
        clip_val: Ngưỡng kẹp L2-Hys (mặc định 0.2).
        eps: Epsilon chống chia cho 0.
        
    Returns:
        np.ndarray vector 1D chứa các đặc trưng HOG đã chuẩn hóa L2-Hys.
        Ví dụ:
        - img 112x112: ra (6084,)
        - img 64x64: ra (1764,)
        - img 16x16: ra (36,)
    """
    # Chuyển sang ảnh xám nếu là 3 kênh
    if img.ndim == 3:
        # Tự tính grayscale bằng công thức tiêu chuẩn ITU-R 601-2 luma:
        # Y = 0.299 R + 0.587 G + 0.114 B
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img.astype(np.float64)

    H, W = gray.shape
    min_size = cell_size * block_size

    # Kiểm tra kích thước tối thiểu bắt buộc
    if H < min_size or W < min_size:
        raise ValueError(
            f"Kích thước ảnh ({H}x{W}) quá nhỏ, không đủ để tạo một block "
            f"({block_size}x{block_size} cells = {min_size}x{min_size} pixels). "
            f"Yêu cầu tối thiểu mỗi chiều >= {min_size}."
        )

    # 1. Tính gradient và góc
    mag, angle = compute_gradients(gray)

    # 2. Tính histogram của từng cell
    cell_hists = compute_cell_histograms(mag, angle, cell_size=cell_size, num_bins=num_bins)
    n_cells_y, n_cells_x, _ = cell_hists.shape

    # 3. Chuẩn hóa block theo stride 1 cell
    n_blocks_y = n_cells_y - block_size + 1
    n_blocks_x = n_cells_x - block_size + 1

    block_feats = []
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            # Lấy các cell trong block (block_size x block_size)
            block = cell_hists[by:by + block_size, bx:bx + block_size, :]
            vec = block.ravel()

            # Chuẩn hóa L2-Hys:
            # Bước 1: L2 norm ban đầu
            l2_norm = np.sqrt(np.sum(vec ** 2) + eps ** 2)
            vec = vec / l2_norm

            # Bước 2: Kẹp giá trị tại clip_val
            vec = np.minimum(vec, clip_val)

            # Bước 3: L2 norm lại lần nữa
            l2_norm2 = np.sqrt(np.sum(vec ** 2) + eps ** 2)
            vec = vec / l2_norm2

            block_feats.append(vec)

    hog_vector = np.concatenate(block_feats).astype(np.float32)

    # Đảm bảo không chứa NaN hoặc Inf
    if np.isnan(hog_vector).any() or np.isinf(hog_vector).any():
        raise FloatingPointError("Đặc trưng HOG chứa giá trị NaN hoặc Inf!")

    return hog_vector
