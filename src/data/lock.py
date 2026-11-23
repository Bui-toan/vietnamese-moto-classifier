"""
Module khóa dữ liệu (src/data/lock.py)
Tuân thủ nghiêm ngặt mục 3.4 của kế hoạch v4:
- Sao chép ảnh test sang data/test_frozen/.
- Ghi lại SHA-256 của manifest, train.csv, val.csv, test.csv, ood_val.csv, ood_test.csv.
- Ghi lại SHA-256 từng ảnh test và ảnh OOD (đường dẫn tương đối -> hash).
- Ghi lại trường locked_at (ISO-8601, UTC).
- Ghi lại git commit hash.
- Xuất splits/lock.json.
"""

import os
import shutil
import hashlib
import json
import subprocess
from datetime import datetime, timezone
import argparse
import pandas as pd


def compute_sha256(filepath: str) -> str:
    """Tính SHA-256 của một file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    """Lấy mã commit git hiện tại."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def lock_dataset(
    splits_dir: str = "splits",
    manifest_path: str = "data/manifest.csv",
    frozen_dir: str = "data/test_frozen",
    out_lock_json: str = "splits/lock.json"
):
    print("=== BẮT ĐẦU KHÓA DỮ LIỆU (DATA LOCKING) ===")
    os.makedirs(frozen_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    lock_data = {
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "split_hashes": {},
        "file_hashes": {}
    }

    # 1. Băm manifest nếu có
    if os.path.exists(manifest_path):
        lock_data["manifest_hash"] = compute_sha256(manifest_path)

    # 2. Băm các file CSV split
    split_files = ["train.csv", "val.csv", "test.csv", "ood_val.csv", "ood_test.csv"]
    for sf in split_files:
        p = os.path.join(splits_dir, sf)
        if os.path.exists(p):
            lock_data["split_hashes"][sf] = compute_sha256(p)

    # 3. Sao chép và băm ảnh test
    test_csv = os.path.join(splits_dir, "test.csv")
    if os.path.exists(test_csv):
        df_te = pd.read_csv(test_csv)
        print(f"Đang sao chép {len(df_te)} ảnh test sang {frozen_dir}...")
        for _, row in df_te.iterrows():
            orig_p = row["image_path"]
            if os.path.exists(orig_p):
                filename = os.path.basename(orig_p)
                dest_p = os.path.join(frozen_dir, filename)
                shutil.copy2(orig_p, dest_p)
                # Băm file trong frozen_dir
                h = compute_sha256(dest_p)
                lock_data["file_hashes"][filename] = h

    # 4. Băm ảnh OOD (cả val và test)
    for ood_sf in ["ood_val.csv", "ood_test.csv"]:
        ood_csv = os.path.join(splits_dir, ood_sf)
        if os.path.exists(ood_csv):
            df_ood = pd.read_csv(ood_csv)
            for _, row in df_ood.iterrows():
                orig_p = row["image_path"]
                if os.path.exists(orig_p):
                    filename = os.path.basename(orig_p)
                    h = compute_sha256(orig_p)
                    lock_data["file_hashes"][f"ood_{filename}"] = h

    # Ghi lock.json
    with open(out_lock_json, "w", encoding="utf-8") as f:
        json.dump(lock_data, f, indent=2, ensure_ascii=False)

    print(f"[PASS] Đã tạo thành công {out_lock_json}")
    print(f"- locked_at:   {lock_data['locked_at']}")
    print(f"- git_commit:  {lock_data['git_commit']}")
    print(f"- Số file băm: {len(lock_data['file_hashes'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Khóa dữ liệu test và OOD")
    parser.add_argument("--splits-dir", type=str, default="splits")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv")
    parser.add_argument("--frozen-dir", type=str, default="data/test_frozen")
    parser.add_argument("--out-lock", type=str, default="splits/lock.json")
    args = parser.parse_args()
    lock_dataset(args.splits_dir, args.manifest, args.frozen_dir, args.out_lock)
