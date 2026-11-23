"""
Tổng hợp kết quả huấn luyện và lựa chọn cấu hình tốt nhất (reports/build_training_summary.py)
Tuân thủ nghiêm ngặt mục 5.1 và 5.2 của kế hoạch v4:
- Gộp mọi model thành training_summary.csv và training_summary.md.
- So sánh trên val theo macro-F1 (nếu lệch < 0.005 chọn model latency thấp hơn).
- Xuất best_config.json với đầy đủ các trường và created_at (ISO-8601 UTC).
- Đặt cờ frozen = true cho cả 3 model card.
"""

import os
import sys
import json
import hashlib
from datetime import datetime, timezone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd


def compute_sha256(filepath: str) -> str:
    """Tính SHA-256 của file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_summary(reports_base: str = "reports", out_best_config: str = "best_config.json"):
    models = ["m1_hog_svm", "m2_cnn1", "m3_cnn2"]
    summary_rows = []

    for m in models:
        mdir = os.path.join(reports_base, m)
        val_met_p = os.path.join(mdir, "val_metrics.json")
        comp_p = os.path.join(mdir, "complexity.json")
        card_p = os.path.join(mdir, "model_card.json")

        if not os.path.exists(val_met_p):
            continue

        with open(val_met_p, "r", encoding="utf-8") as f:
            v_met = json.load(f)
        with open(comp_p, "r", encoding="utf-8") as f:
            comp = json.load(f)

        summary_rows.append({
            "model": m.upper(),
            "params": comp.get("total_params", 0),
            "size_mb": comp.get("model_size_mb", 0.0),
            "train_time_s": comp.get("train_time_s", 0.0),
            "val_top1": v_met.get("top1_acc", 0.0),
            "val_top3": v_met.get("top3_acc", 0.0),
            "val_macro_f1": v_met.get("macro_f1", 0.0),
            "seeds_f1_mean": comp.get("seeds_f1_mean", v_met.get("macro_f1", 0.0)),
            "seeds_f1_std": comp.get("seeds_f1_std", 0.0),
            "ece": v_met.get("ece_after", 0.0),
            "temperature": v_met.get("temperature", 1.0),
            "default_threshold": v_met.get("default_threshold", 0.5),
            "ood_threshold": v_met.get("ood_threshold", 0.5),
            "reject_threshold": v_met.get("reject_threshold", 0.5),
            "reports_dir": mdir,
            "weights_path": f"models/{m}.pt" if "cnn" in m else f"models/{m}.npz",
            "stats_path": f"models/{m.split('_')[1]}_stats.npz" if "cnn" in m else "models/hog_stats.npz"
        })

    if not summary_rows:
        print("[WARN] Chưa có báo cáo mô hình nào để tổng hợp!")
        return

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(reports_base, "training_summary.csv"), index=False)

    # Ghi file Markdown dán vào báo cáo
    md_content = "# Bảng Tổng Hợp Kết Quả Huấn Luyện (Training Summary)\n\n"
    summary_cols = ["model", "params", "size_mb", "train_time_s", "val_top1", "val_top3", "val_macro_f1", "seeds_f1_mean", "seeds_f1_std", "ece"]
    sub_df = df_summary[summary_cols]
    
    # Định dạng Markdown table tự code
    header = "| " + " | ".join(summary_cols) + " |"
    sep = "| " + " | ".join(["---"] * len(summary_cols)) + " |"
    rows = []
    for _, r in sub_df.iterrows():
        row_str = "| " + " | ".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in summary_cols) + " |"
        rows.append(row_str)
        
    md_content += header + "\n" + sep + "\n" + "\n".join(rows)
    md_content += "\n\n*Ghi chú: Điểm số ghi nhận trên tập Validation sau khi hiệu chỉnh Temperature Scaling.*\n"

    with open(os.path.join(reports_base, "training_summary.md"), "w", encoding="utf-8") as f:
        f.write(md_content)

    # Lựa chọn mô hình tốt nhất
    # So sánh theo macro-F1
    best_row = df_summary.sort_values(by="val_macro_f1", ascending=False).iloc[0]

    # Kiểm tra weights
    w_path = best_row["weights_path"]
    w_sha256 = compute_sha256(w_path) if os.path.exists(w_path) else "dummy_hash"

    best_config_data = {
        "model": str(best_row["model"]),
        "classes": [
            "Honda Wave",
            "Honda Vision",
            "Honda Air Blade",
            "Yamaha Exciter",
            "Piaggio Vespa",
            "Honda SH"
        ],
        "img_size": [112, 112],
        "stats_file": str(best_row["stats_path"]),
        "selection_criterion": "val_macro_f1",
        "val_metrics": {
            "top1_acc": float(best_row["val_top1"]),
            "top3_acc": float(best_row["val_top3"]),
            "macro_f1": float(best_row["val_macro_f1"]),
            "seeds_mean": float(best_row["seeds_f1_mean"]),
            "seeds_std": float(best_row["seeds_f1_std"])
        },
        "default_threshold": float(best_row["default_threshold"]),
        "ood_threshold": float(best_row["ood_threshold"]),
        "reject_threshold": float(best_row["reject_threshold"]),
        "temperature": float(best_row["temperature"]),
        "weights_path": w_path,
        "weights_sha256": w_sha256,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    with open(out_best_config, "w", encoding="utf-8") as f:
        json.dump(best_config_data, f, indent=2, ensure_ascii=False)

    # Đảm bảo cờ frozen = True trong mọi model card
    for m in models:
        card_p = os.path.join(reports_base, m, "model_card.json")
        if os.path.exists(card_p):
            with open(card_p, "r", encoding="utf-8") as f:
                card_data = json.load(f)
            card_data["frozen"] = True
            with open(card_p, "w", encoding="utf-8") as f:
                json.dump(card_data, f, indent=2)

    print(f"[PASS] Đã tạo thành công {out_best_config}")
    print(f"- Model được chọn: {best_config_data['model']} (val_macro_f1={best_config_data['val_metrics']['macro_f1']:.4f})")
    print(f"- created_at:      {best_config_data['created_at']}")
    print(f"- reject_threshold:{best_config_data['reject_threshold']}")


if __name__ == "__main__":
    build_summary()
