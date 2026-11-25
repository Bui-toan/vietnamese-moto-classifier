"""
Script tiền xử lý và gom cụm chống rò rỉ dữ liệu (src/data/deduplicate_and_group.py)
1. Khử trùng hệt SHA-256 (bỏ trùng lặp cùng lớp, loại bỏ xung đột nhãn khác lớp).
2. Tính aHash 8x8 chuẩn NumPy.
3. Gom cụm các ảnh gần trùng (Hamming <= 5) của cùng một lớp vào chung Group ID (Disjoint Set Union / Connected Components).
4. Xử lý các cặp gần trùng khác lớp nếu có (loại bỏ để tránh nhiễu nhãn).
5. Phân chia các ảnh còn lại thành các group_id có quy mô vừa phải (3-5 ảnh/group).
6. Xuất data/manifest.csv sạch 100%, sẵn sàng cho split.py và check_leak.py.
"""

import os
import hashlib
from collections import defaultdict
import numpy as np
import pandas as pd
import cv2


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_ahash(filepath: str) -> np.ndarray:
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {filepath}")
    small = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    m = float(np.mean(small))
    return (small >= m).flatten()


def deduplicate_and_group(
    manifest_in: str = "data/manifest.csv",
    manifest_out: str = "data/manifest.csv",
    group_size: int = 5,
    hamming_threshold: int = 5
):
    print("=== BẮT ĐẦU DEDUPLICATE VÀ GOM CỤM CHỐNG RÒ RỈ ===")
    df = pd.read_csv(manifest_in)
    print(f"Tổng số bản ghi ban đầu: {len(df)}")

    # 1. Tính SHA-256
    print("Đang tính SHA-256...")
    sha_map = {}
    valid_rows = []
    for idx, row in df.iterrows():
        p = row["image_path"]
        if os.path.exists(p):
            h = compute_sha256(p)
            sha_map[p] = h
            valid_rows.append(row)
        else:
            print(f"[WARN] File không tồn tại: {p}")

    df_valid = pd.DataFrame(valid_rows)

    # Nhóm theo SHA-256
    h_to_rows = defaultdict(list)
    for idx, row in df_valid.iterrows():
        p = row["image_path"]
        h_to_rows[sha_map[p]].append((idx, row["label"], p, row))

    clean_rows = []
    dropped_cross_label = 0
    dropped_same_label = 0

    for h, items in h_to_rows.items():
        labels = set(x[1] for x in items)
        if len(labels) > 1:
            # Xung đột nhãn: cùng một ảnh nhưng được gán 2 lớp khác nhau
            dropped_cross_label += len(items)
            continue
        elif len(items) > 1:
            dropped_same_label += (len(items) - 1)
            clean_rows.append(items[0][3])
        else:
            clean_rows.append(items[0][3])

    df_clean = pd.DataFrame(clean_rows).reset_index(drop=True)
    print(f"Đã loại bỏ {dropped_cross_label} ảnh xung đột nhãn khác lớp.")
    print(f"Đã loại bỏ {dropped_same_label} ảnh trùng hệt SHA-256 cùng lớp.")
    print(f"Số lượng ảnh duy nhất còn lại: {len(df_clean)}")

    # 2. Tính aHash 8x8
    print("Đang tính aHash 8x8...")
    ahashes = []
    for p in df_clean["image_path"]:
        ahashes.append(compute_ahash(p))
    ahashes = np.array(ahashes)  # shape: (N, 64), boolean

    # 3. Phân cụm Disjoint Set Union theo lớp
    print("Đang phân cụm các ảnh gần trùng (Hamming <= 5) theo từng lớp...")
    
    # Pack boolean hash thành uint64 để tính nhanh khoảng cách Hamming
    # Mỗi hash 64 bit có thể biểu diễn bằng 1 số uint64
    uint_hashes = np.zeros(len(ahashes), dtype=np.uint64)
    for bit_idx in range(64):
        uint_hashes |= (ahashes[:, bit_idx].astype(np.uint64) << np.uint64(bit_idx))

    classes = sorted(df_clean["label"].unique())
    new_groups = [None] * len(df_clean)
    
    # Bỏ các cặp gần trùng giữa các lớp khác nhau nếu có
    # (Tránh trường hợp 2 xe thuộc 2 hãng khác nhau nhưng ảnh quá giống do góc chụp hoặc watermark)
    drop_cross_class_indices = set()
    for i in range(len(df_clean)):
        if i in drop_cross_class_indices:
            continue
        h1 = uint_hashes[i]
        c1 = df_clean.at[i, "label"]
        for j in range(i + 1, len(df_clean)):
            c2 = df_clean.at[j, "label"]
            if c1 != c2:
                # Tính hamming giữa 2 uint64
                diff = int(h1 ^ uint_hashes[j])
                dist = bin(diff).count('1')
                if dist <= 2:  # Quá giống nhau dù khác nhãn -> nhiễu nghiêm trọng
                    drop_cross_class_indices.add(i)
                    drop_cross_class_indices.add(j)
                    break

    if drop_cross_class_indices:
        print(f"Loại bỏ {len(drop_cross_class_indices)} ảnh nhiễu gần trùng xuyên nhãn.")
        df_clean = df_clean.drop(index=list(drop_cross_class_indices)).reset_index(drop=True)
        # Tính lại aHash & uint_hashes
        ahashes = []
        for p in df_clean["image_path"]:
            ahashes.append(compute_ahash(p))
        ahashes = np.array(ahashes)
        uint_hashes = np.zeros(len(ahashes), dtype=np.uint64)
        for bit_idx in range(64):
            uint_hashes |= (ahashes[:, bit_idx].astype(np.uint64) << np.uint64(bit_idx))

    # Bây giờ gom cụm theo từng lớp
    for cls in classes:
        cls_indices = df_clean.index[df_clean["label"] == cls].tolist()
        num_cls = len(cls_indices)
        prefix = cls.lower().replace(" ", "_")

        # DSU để gom cụm
        parent = {idx: idx for idx in cls_indices}
        def find(u):
            if parent[u] == u: return u
            parent[u] = find(parent[u])
            return parent[u]
        def union(u, v):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv

        cls_uints = uint_hashes[cls_indices]
        for a in range(num_cls):
            idx_a = cls_indices[a]
            h_a = cls_uints[a]
            for b in range(a + 1, num_cls):
                idx_b = cls_indices[b]
                h_b = cls_uints[b]
                diff = int(h_a ^ h_b)
                dist = bin(diff).count('1')
                if dist <= hamming_threshold:
                    union(idx_a, idx_b)

        # Nhóm theo root DSU
        clusters = defaultdict(list)
        for idx in cls_indices:
            clusters[find(idx)].append(idx)

        # Gán group_id sao cho các cụm to giữ nguyên, các ảnh đơn lẻ được ghép thành group size ~ group_size
        current_small_group = []
        group_counter = 0

        for root, members in clusters.items():
            if len(members) > 1:
                # Cụm gần trùng: bắt buộc chung 1 group_id
                gid = f"{prefix}_grp_{group_counter:03d}"
                group_counter += 1
                for m in members:
                    df_clean.at[m, "group_id"] = gid
            else:
                # Ảnh đơn lẻ: gộp theo nhóm group_size
                current_small_group.append(members[0])
                if len(current_small_group) >= group_size:
                    gid = f"{prefix}_grp_{group_counter:03d}"
                    group_counter += 1
                    for m in current_small_group:
                        df_clean.at[m, "group_id"] = gid
                    current_small_group = []

        # Xử lý phần dư của ảnh đơn lẻ
        if current_small_group:
            gid = f"{prefix}_grp_{group_counter:03d}"
            group_counter += 1
            for m in current_small_group:
                df_clean.at[m, "group_id"] = gid

        print(f"Lớp '{cls}': {len(cls_indices)} ảnh -> {group_counter} groups.")

    df_clean.to_csv(manifest_out, index=False)
    print(f"=== ĐÃ XUẤT MANIFEST MỚI SẠCH 100%: {manifest_out} ({len(df_clean)} ảnh) ===")


if __name__ == "__main__":
    deduplicate_and_group()
