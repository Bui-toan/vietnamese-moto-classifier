"""
Script quét ảnh từ data/raw/, lấy trực tiếp tên thư mục làm label và sinh manifest chuẩn (src/data/build_manifest_from_raw.py)
"""

import os
import argparse
import pandas as pd

VALID_CLASSES = {
    "Honda Air Blade",
    "Honda Vision",
    "Honda Wave",
    "Yamaha Exciter",
    "Piaggio Vespa",
    "Honda SH",
    "ood"
}


def build_manifest(
    raw_dir: str = "data/raw",
    out_manifest: str = "data/manifest.csv",
    group_size: int = 5
) -> pd.DataFrame:
    print(f"=== BẮT ĐẦU QUÉT THƯ MỤC ẢNH GỐC: {raw_dir} ===")
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    rows = []

    subdirs = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])

    for folder in subdirs:
        if folder not in VALID_CLASSES and folder.lower() != "ood":
            print(f"[WARN] Bỏ qua thư mục '{folder}' không thuộc 6 lớp quy định.")
            continue

        label = folder
        folder_path = os.path.join(raw_dir, folder)
        files = sorted(os.listdir(folder_path))
        img_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_exts]

        print(f"-> Thư mục '{folder}': {len(img_files)} ảnh")

        # Sinh group_id chuẩn
        folder_prefix = folder.lower().replace(" ", "_")
        for idx, filename in enumerate(img_files):
            group_idx = idx // group_size
            group_id = f"{folder_prefix}_grp_{group_idx:03d}"
            img_rel_path = os.path.join(raw_dir, folder, filename).replace("\\", "/")

            rows.append({
                "image_path": img_rel_path,
                "label": label,
                "group_id": group_id,
                "bbox": ""
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_manifest), exist_ok=True)
    df.to_csv(out_manifest, index=False)

    print(f"\n[PASS] Đã tạo thành công {out_manifest} với tổng cộng {len(df)} ảnh!")
    print(df["label"].value_counts().to_string())
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--out-manifest", type=str, default="data/manifest.csv")
    parser.add_argument("--group-size", type=int, default=5)
    args = parser.parse_args()

    build_manifest(args.raw_dir, args.out_manifest, args.group_size)
