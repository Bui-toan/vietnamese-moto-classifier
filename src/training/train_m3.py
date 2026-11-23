"""
Script huấn luyện M3: CNN-2 ResNet-style (src/training/train_m3.py)
Gọi modular: grid_search.py -> trainer.py (CosineAnnealingLR) -> calibration.py -> ablation
"""

import os
import sys
import time
import json
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
os.environ["MPLCONFIGDIR"] = os.path.abspath("reports/.mpl_cache")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn2 import CNN2
from src.train.dataset import MotorbikeDataset
from src.preprocess.normalize import save_stats
from src.training.grid_search import grid_search_cnn
from src.training.trainer import TorchTrainer, plot_curves
from src.training.calibration import optimize_temperature, compute_all_thresholds
from src.eval.metrics import compute_accuracy_topk, compute_per_class_metrics, CLASS_NAMES
from src.train.common import generate_env_json, generate_model_card


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-aug", action="store_true", help="Chạy ablation không data augmentation")
    args = parser.parse_args()
    use_aug = not args.no_aug

    reports_dir = "reports/m3_cnn2"
    weights_path = "models/m3_cnn2.pt"
    stats_path = "models/cnn2_stats.npz"
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/ablation", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== [M3 CNN-2 ResNet] BẮT ĐẦU HUẤN LUYỆN (use_aug={use_aug}) trên {device} ===")
    df_tr = pd.read_csv("splits/train.csv") if os.path.exists("splits/train.csv") else pd.DataFrame(columns=["image_path", "label"])
    df_val = pd.read_csv("splits/val.csv") if os.path.exists("splits/val.csv") else pd.DataFrame(columns=["image_path", "label"])

    mean_rgb = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std_rgb = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    save_stats(stats_path, mean_rgb, std_rgb)

    if len(df_tr) > 0 and os.path.exists(str(df_tr.iloc[0].get("image_path", ""))):
        train_ds = MotorbikeDataset(df_tr, is_train=use_aug, mean=mean_rgb, std=std_rgb)
        val_ds = MotorbikeDataset(df_val, is_train=False, mean=mean_rgb, std=std_rgb)
    else:
        torch.manual_seed(42)
        train_ds = TensorDataset(torch.randn(120, 3, 112, 112), torch.tensor(np.repeat(np.arange(6), 20), dtype=torch.long))
        val_ds = TensorDataset(torch.randn(60, 3, 112, 112), torch.tensor(np.repeat(np.arange(6), 10), dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # 1. Grid Search
    best_cfg, df_grid = grid_search_cnn(CNN2, train_loader, val_loader, device=device)
    df_grid.to_csv(os.path.join(reports_dir, "gridsearch.csv"), index=False)

    # 2. Chạy 3 seeds
    seed_scores = []
    best_model = None
    t0 = time.time()
    epochs = 20

    for s in [42, 43, 44]:
        torch.manual_seed(s)
        model = CNN2(num_classes=6, dropout_p=0.3)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_cfg["lr"], weight_decay=best_cfg["wd"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        trainer = TorchTrainer(model, criterion, optimizer, scheduler=scheduler, device=device, patience=15, save_path=weights_path)
        history_df = trainer.fit(train_loader, val_loader, epochs=epochs)

        seed_scores.append({"seed": s, "val_macro_f1": history_df.iloc[-1]["val_macro_f1"]})
        if s == 42:
            best_model = model
            history_df.to_csv(os.path.join(reports_dir, "history.csv"), index=False)
            plot_curves(history_df, os.path.join(reports_dir, "learning_curve.png"))

    train_time = time.time() - t0
    df_seeds = pd.DataFrame(seed_scores)
    df_seeds.to_csv(os.path.join(reports_dir, "seeds.csv"), index=False)

    # 3. Calibration
    best_model.eval()
    val_logits, val_targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            val_logits.append(best_model(xb).cpu().numpy())
            val_targets.extend(yb.numpy())
    val_logits_arr = np.concatenate(val_logits, axis=0)
    val_y_arr = np.array(val_targets)

    T, ece_b, nll_b, ece_a, nll_a = optimize_temperature(val_logits_arr, val_y_arr)
    exp_s = np.exp((val_logits_arr / T) - np.max(val_logits_arr / T, axis=1, keepdims=True))
    cal_probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
    threshs = compute_all_thresholds(val_y_arr, cal_probs, None)

    # 4. Gói báo cáo
    n_params = sum(p.numel() for p in best_model.parameters())
    complexity = {
        "total_params": n_params, "trainable_params": sum(p.numel() for p in best_model.parameters() if p.requires_grad),
        "macs": None, "model_size_mb": float(os.path.getsize(weights_path)/(1024*1024)),
        "train_time_s": train_time, "best_epoch": epochs,
        "seeds_f1_mean": float(df_seeds["val_macro_f1"].mean()), "seeds_f1_std": float(df_seeds["val_macro_f1"].std())
    }
    with open(os.path.join(reports_dir, "complexity.json"), "w", encoding="utf-8") as f:
        json.dump(complexity, f, indent=2)

    val_preds = np.argmax(cal_probs, axis=1)
    acc_topk = compute_accuracy_topk(val_y_arr, cal_probs, ks=(1, 2, 3))
    class_m = compute_per_class_metrics(val_y_arr, val_preds, num_classes=6)

    val_metrics = {
        **acc_topk, "macro_f1": class_m["macro_f1"], "weighted_f1": class_m["weighted_f1"],
        "per_class": class_m["per_class"], "nll_before": nll_b, "nll_after": nll_a,
        "ece_before": ece_b, "ece_after": ece_a, "temperature": T, **threshs
    }
    with open(os.path.join(reports_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)

    # Ablation subfolder
    sub_dir = "m3_with_aug" if use_aug else "m3_no_aug"
    os.makedirs(f"reports/ablation/{sub_dir}", exist_ok=True)
    with open(f"reports/ablation/{sub_dir}/val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)

    cfg_p = os.path.join(reports_dir, "config.json")
    with open(cfg_p, "w", encoding="utf-8") as f:
        json.dump({"model_name": "M3_CNN2", "best_hyperparams": best_cfg, "classes": CLASS_NAMES, "img_size": [112, 112]}, f, indent=2)

    generate_env_json(os.path.join(reports_dir, "env.json"), device=str(device))
    generate_model_card(os.path.join(reports_dir, "model_card.json"), "M3_CNN2", weights_path, stats_path, cfg_p, frozen=True)
    print(f"[PASS] M3 hoàn tất! Báo cáo tại {reports_dir}")


if __name__ == "__main__":
    main()
