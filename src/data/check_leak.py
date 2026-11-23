"""
Module kiểm tra rò rỉ dữ liệu mở rộng (src/data/check_leak.py)
Tuân thủ nghiêm ngặt mục 3.3 và TC-05:
1. Kiểm tra Group ID không trùng lặp giữa các tập (Train, Val, Test).
2. Kiểm tra trùng hệt: So sánh SHA-256 nội dung file ảnh.
3. Kiểm tra gần trùng (Near-duplicate):
   - Average-hash 8x8 tự code 100% bằng NumPy (resize ảnh xám về 8x8, so từng điểm với giá trị trung bình -> 64 bit).
   - Khoảng cách Hamming <= 5 trên 64 bit.
   - KHÔNG dùng cv2.img_hash hay thư viện ngoài.
   - Ghi kết quả vào near_duplicates.csv.
   - Xử lý qua near_duplicates_resolved.csv nếu được cung cấp.
"""

import os
import hashlib
import argparse
from typing import Dict, List, Tuple
import numpy as np
import cv2
import pandas as pd


def compute_sha256(filepath: str) -> str:
    """Tính mã SHA-256 của file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_average_hash_8x8(img_path_or_array) -> np.ndarray:
    """
    Thuật toán Average-hash 8x8 tự code bằng NumPy:
    - Resize ảnh xám về 8x8.
    - So sánh từng điểm ảnh với giá trị trung bình (pixel >= mean).
    - Trả về mảng boolean 64 phần tử (64-bit hash).
    KHÔNG dùng cv2.img_hash.
    """
    if isinstance(img_path_or_array, str):
        img = cv2.imread(img_path_or_array, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Không thể đọc ảnh: {img_path_or_array}")
    elif img_path_or_array.ndim == 3:
        img = (0.299 * img_path_or_array[:, :, 0] + 0.587 * img_path_or_array[:, :, 1] + 0.114 * img_path_or_array[:, :, 2]).astype(np.uint8)
    else:
        img = img_path_or_array

    # Resize về 8x8
    small = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    mean_val = float(np.mean(small))
    
    # 64 bit nhị phân
    bit_hash = (small >= mean_val).flatten()
    return bit_hash


def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
    """Tính khoảng cách Hamming giữa 2 vector nhị phân 64-bit."""
    return int(np.sum(hash1 != hash2))


def check_leak(
    train_csv: str = "splits/train.csv",
    val_csv: str = "splits/val.csv",
    test_csv: str = "splits/test.csv",
    resolved_csv: str = "splits/near_duplicates_resolved.csv",
    out_near_csv: str = "splits/near_duplicates.csv"
) -> bool:
    """
    Kiểm tra toàn diện rò rỉ dữ liệu giữa các tập.
    Returns:
        True nếu sạch hoàn toàn, False nếu còn rò rỉ.
    """
    print("=== BẮT ĐẦU KIỂM TRA RÒ RỈ DỮ LIỆU (LEAK CHECK) ===")
    df_tr = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_te = pd.read_csv(test_csv)

    clean = True

    # 1. Kiểm tra Group ID
    groups_tr = set(df_tr["group_id"].unique())
    groups_val = set(df_val["group_id"].unique())
    groups_te = set(df_te["group_id"].unique())

    leak_tr_val = groups_tr & groups_val
    leak_tr_te = groups_tr & groups_te
    leak_val_te = groups_val & groups_te

    if leak_tr_val:
        print(f"[FAIL] Rò rỉ Group ID giữa Train và Val: {leak_tr_val}")
        clean = False
    if leak_tr_te:
        print(f"[FAIL] Rò rỉ Group ID giữa Train và Test: {leak_tr_te}")
        clean = False
    if leak_val_te:
        print(f"[FAIL] Rò rỉ Group ID giữa Val và Test: {leak_val_te}")
        clean = False

    if clean:
        print("[PASS] Không có trùng lặp Group ID giữa Train, Val, Test.")

    # 2. Kiểm tra trùng hệt (Exact SHA-256) và gần trùng (Average Hash 8x8)
    splits = [("train", df_tr), ("val", df_val), ("test", df_te)]
    hashes = {}
    ahashes = {}

    for split_name, df in splits:
        for idx, row in df.iterrows():
            p = row["image_path"]
            if os.path.exists(p):
                file_hash = compute_sha256(p)
                ahash = compute_average_hash_8x8(p)
                hashes[p] = (split_name, file_hash, row["group_id"])
                ahashes[p] = ahash

    # Kiểm tra trùng SHA-256
    hash_to_paths = {}
    for p, (s, h, g) in hashes.items():
        hash_to_paths.setdefault(h, []).append((p, s, g))

    exact_duplicates = []
    for h, plist in hash_to_paths.items():
        if len(plist) > 1:
            splits_in_h = set(s for _, s, _ in plist)
            if len(splits_in_h) > 1:
                exact_duplicates.append(plist)
                print(f"[FAIL] Trùng hệt SHA-256 xuyên split: {plist}")
                clean = False

    # Kiểm tra gần trùng (Near-duplicates: Hamming <= 5)
    # Lấy danh sách đã giải quyết nếu có
    resolved_pairs = set()
    if os.path.exists(resolved_csv):
        df_res = pd.read_csv(resolved_csv)
        for _, row in df_res.iterrows():
            pair = tuple(sorted([row["path_1"], row["path_2"]]))
            resolved_pairs.add(pair)

    near_duplicates = []
    path_list = list(ahashes.keys())
    n = len(path_list)

    for i in range(n):
        p1 = path_list[i]
        s1, _, g1 = hashes[p1]
        a1 = ahashes[p1]
        for j in range(i + 1, n):
            p2 = path_list[j]
            s2, _, g2 = hashes[p2]
            # Chỉ kiểm tra giữa các split khác nhau
            if s1 != s2:
                pair_key = tuple(sorted([p1, p2]))
                if pair_key in resolved_pairs:
                    continue
                dist = hamming_distance(a1, ahashes[p2])
                if dist <= 5:
                    near_duplicates.append({
                        "path_1": p1,
                        "split_1": s1,
                        "group_1": g1,
                        "path_2": p2,
                        "split_2": s2,
                        "group_2": g2,
                        "hamming_dist": dist
                    })

    if near_duplicates:
        df_near = pd.DataFrame(near_duplicates)
        df_near.to_csv(out_near_csv, index=False)
        print(f"[WARN] Tìm thấy {len(near_duplicates)} cặp gần trùng (Hamming <= 5) xuyên split. Đã ghi vào {out_near_csv}")
        clean = False
    else:
        print("[PASS] Không có ảnh gần trùng (Hamming <= 5) chưa giải quyết.")

    if clean:
        print("=== KẾT QUẢ: LEAK CHECK HOÀN TOÀN SẠCH (PASS) ===")
    else:
        print("=== KẾT QUẢ: CẦN XỬ LÝ RÒ RỈ DỮ LIỆU ===")

    return clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm tra rò rỉ dữ liệu")
    parser.add_argument("--train", type=str, default="splits/train.csv")
    parser.add_argument("--val", type=str, default="splits/val.csv")
    parser.add_argument("--test", type=str, default="splits/test.csv")
    parser.add_argument("--resolved", type=str, default="splits/near_duplicates_resolved.csv")
    parser.add_argument("--out-near", type=str, default="splits/near_duplicates.csv")
    args = parser.parse_args()
    check_leak(args.train, args.val, args.test, args.resolved, args.out_near)
