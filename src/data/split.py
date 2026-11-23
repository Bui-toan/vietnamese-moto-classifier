"""
Module phân chia dữ liệu theo Group ID (src/data/split.py)
Tuân thủ nghiêm ngặt mục 3.2 của kế hoạch v4:
- Thuật toán deficit: Group lớn xếp trước, gán vào tập đang thiếu nhiều nhất để hướng tới tỉ lệ 70/15/15.
- Assert cứng: Mỗi lớp có ít nhất 10 group; không group_id nào xuất hiện ở hai tập.
- Cảnh báo [WARN]: Nếu tỉ lệ mỗi lớp lệch quá +-3% so với 70/15/15, ghi cảnh báo vào splits/split_warnings.txt (không cắt group).
"""

import os
import random
import argparse
from typing import Dict, List, Tuple
import pandas as pd


def split_groups(
    items: Dict[str, int],
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42
) -> Tuple[Tuple[List[str], List[str], List[str]], List[int]]:
    """
    Thuật toán chia group theo deficit theo đúng đặc tả mục 3.2:
    def split_groups(items, ratios=(0.70, 0.15, 0.15), seed=42):
        rng = random.Random(seed)
        gids = sorted(items); rng.shuffle(gids)
        gids.sort(key=lambda g: -items[g]) # group lớn xếp trước
        total = sum(items.values()); got = [0, 0, 0]; out = ([], [], [])
        for g in gids:
            deficit = [ratios[i] * total - got[i] for i in range(3)]
            k = deficit.index(max(deficit)) # tập đang thiếu nhiều nhất
            out[k].append(g); got[k] += items[g]
        return out, got
    """
    rng = random.Random(seed)
    gids = sorted(items.keys())
    rng.shuffle(gids)
    gids.sort(key=lambda g: -items[g])  # group lớn xếp trước, hòa thì giữ thứ tự ngẫu nhiên

    total = sum(items.values())
    got = [0, 0, 0]
    out = ([], [], [])

    for g in gids:
        deficit = [ratios[i] * total - got[i] for i in range(3)]
        k = deficit.index(max(deficit))  # tập đang thiếu nhiều nhất
        out[k].append(g)
        got[k] += items[g]

    return out, got


def run_split(manifest_path: str, output_dir: str = "splits", seed: int = 42):
    """
    Chia manifest thành train.csv, val.csv, test.csv, ood_val.csv, ood_test.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(manifest_path)

    required_cols = {"image_path", "label", "group_id"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Manifest thiếu các cột bắt buộc: {required_cols - set(df.columns)}")

    # Tách dữ liệu OOD nếu có
    is_ood = df["label"].str.lower().isin(["ood", "unknown", "other", "khac"])
    df_ood = df[is_ood].copy()
    df_main = df[~is_ood].copy()

    classes = sorted(df_main["label"].unique())
    print(f"Tổng số lớp chính: {len(classes)} -> {classes}")

    warnings = []
    train_dfs, val_dfs, test_dfs = [], [], []

    for cls in classes:
        df_cls = df_main[df_main["label"] == cls]
        group_counts = df_cls["group_id"].value_counts().to_dict()

        # Assert cứng: mỗi lớp có ít nhất 10 group
        if len(group_counts) < 10:
            raise AssertionError(
                f"Lớp '{cls}' chỉ có {len(group_counts)} group. "
                f"Yêu cầu tối thiểu mỗi lớp phải có >= 10 group!"
            )

        (train_g, val_g, test_g), got = split_groups(group_counts, ratios=(0.70, 0.15, 0.15), seed=seed)

        # Assert cứng: không group_id nào xuất hiện ở hai tập
        set_tr, set_val, set_te = set(train_g), set(val_g), set(test_g)
        assert len(set_tr & set_val) == 0, f"Rò rỉ group giữa train và val ở lớp {cls}"
        assert len(set_tr & set_te) == 0, f"Rò rỉ group giữa train và test ở lớp {cls}"
        assert len(set_val & set_te) == 0, f"Rò rỉ group giữa val và test ở lớp {cls}"

        total_cls = sum(got)
        ratios_actual = [g / total_cls for g in got]
        target_ratios = [0.70, 0.15, 0.15]

        # Kiểm tra độ lệch +-3%
        names = ["train", "val", "test"]
        for i in range(3):
            diff = ratios_actual[i] - target_ratios[i]
            if abs(diff) > 0.03:
                warn_msg = (
                    f"[WARN] Lớp '{cls}' tập {names[i]} lệch {diff*100:+.2f}% "
                    f"(thực tế: {ratios_actual[i]*100:.1f}%, mục tiêu: {target_ratios[i]*100:.1f}%)"
                )
                print(warn_msg)
                warnings.append(warn_msg)

        train_dfs.append(df_cls[df_cls["group_id"].isin(set_tr)].copy())
        val_dfs.append(df_cls[df_cls["group_id"].isin(set_val)].copy())
        test_dfs.append(df_cls[df_cls["group_id"].isin(set_te)].copy())

    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    # Xử lý OOD (chia thành OOD-val và OOD-test theo group_id)
    if not df_ood.empty:
        ood_groups = df_ood["group_id"].value_counts().to_dict()
        (ood_tr, ood_val_g, ood_te_g), ood_got = split_groups(ood_groups, ratios=(0.0, 0.33, 0.67), seed=seed)
        ood_val_set = set(ood_val_g) | set(ood_tr)
        ood_te_set = set(ood_te_g)
        ood_val_df = df_ood[df_ood["group_id"].isin(ood_val_set)].copy()
        ood_test_df = df_ood[df_ood["group_id"].isin(ood_te_set)].copy()
        ood_val_df["split"] = "ood_val"
        ood_test_df["split"] = "ood_test"
        ood_val_df.to_csv(os.path.join(output_dir, "ood_val.csv"), index=False)
        ood_test_df.to_csv(os.path.join(output_dir, "ood_test.csv"), index=False)
        print(f"OOD chia xong: OOD-val={len(ood_val_df)}, OOD-test={len(ood_test_df)}")

    # Ghi file cảnh báo nếu có
    warn_path = os.path.join(output_dir, "split_warnings.txt")
    with open(warn_path, "w", encoding="utf-8") as f:
        for w in warnings:
            f.write(w + "\n")

    print(f"Đã lưu các file phân chia vào {output_dir}:")
    print(f"- Train: {len(train_df)} mẫu")
    print(f"- Val:   {len(val_df)} mẫu")
    print(f"- Test:  {len(test_df)} mẫu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chia dữ liệu theo group_id")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv", help="Đường dẫn manifest.csv")
    parser.add_argument("--output-dir", type=str, default="splits", help="Thư mục xuất CSV")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên")
    args = parser.parse_args()
    run_split(args.manifest, args.output_dir, args.seed)
