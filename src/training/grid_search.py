"""
Module Grid Search Siêu tham số (src/training/grid_search.py)
Tuân thủ nghiêm ngặt mục 5.1 của kế hoạch v4:
- Grid search chỉ chạy 1 seed (42) để chọn cấu hình tốt nhất.
- M1 có 8 tổ hợp: lr in {0.01, 0.05} x epoch in {30, 50} x wd in {0, 1e-4}.
- CNN có 4 tổ hợp: lr in {1e-3, 5e-4} x wd in {1e-4, 1e-3}.
- Tiêu chí lựa chọn: Val Macro-F1.
"""

from typing import Dict, List, Tuple, Any, Optional
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.svm import LinearSVM
from src.eval.metrics import compute_per_class_metrics


def grid_search_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    grid_lr: List[float] = [0.01, 0.05],
    grid_epochs: List[int] = [30, 50],
    grid_wd: List[float] = [0.0, 1e-4]
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Chạy Grid Search cho M1 LinearSVM trên seed 42."""
    results = []
    best_f1 = -1.0
    best_cfg = {}

    for lr in grid_lr:
        for ep in grid_epochs:
            for wd in grid_wd:
                svm = LinearSVM(
                    n_features=X_train.shape[1],
                    n_classes=6,
                    lr=lr,
                    weight_decay=wd,
                    random_state=42
                )
                for _ in range(ep):
                    _ = svm.train_epoch(X_train, y_train, batch_size=32)

                probs = svm.predict_proba(X_val)
                preds = np.argmax(probs, axis=1)
                f1 = compute_per_class_metrics(y_val, preds, num_classes=6)["macro_f1"]

                results.append({
                    "lr": lr,
                    "epochs": ep,
                    "weight_decay": wd,
                    "val_macro_f1": f1
                })
                if f1 > best_f1:
                    best_f1 = f1
                    best_cfg = {"lr": lr, "epochs": ep, "weight_decay": wd}

    df_grid = pd.DataFrame(results)
    return best_cfg, df_grid


def grid_search_cnn(
    model_class: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    grid_lr: List[float] = [1e-3, 5e-4],
    grid_wd: List[float] = [1e-4, 1e-3],
    max_epochs: int = 5,
    device: Optional[torch.device] = None,
    csv_save_path: Optional[str] = None
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Chạy Grid Search cho CNN trên seed 42 với checkpointing."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    
    # Nạp các tổ hợp đã hoàn tất trước đó nếu có file checkpoint
    if csv_save_path and os.path.exists(csv_save_path):
        try:
            df_existing = pd.read_csv(csv_save_path)
            for _, r in df_existing.iterrows():
                results.append({
                    "lr": float(r["lr"]),
                    "weight_decay": float(r["weight_decay"]),
                    "val_macro_f1": float(r["val_macro_f1"])
                })
        except Exception:
            results = []

    total_combos = len(grid_lr) * len(grid_wd)
    print(f"\n[Grid Search M2] Khảo sát {total_combos} tổ hợp siêu tham số ({max_epochs} epochs/tổ hợp)...", flush=True)
    combo_idx = 0

    for lr in grid_lr:
        for wd in grid_wd:
            combo_idx += 1
            # Kiểm tra xem tổ hợp này đã tính chưa
            existing = [r for r in results if abs(r["lr"] - lr) < 1e-7 and abs(r["weight_decay"] - wd) < 1e-7]
            if existing:
                print(f"  [Tổ hợp {combo_idx}/{total_combos}] Đã có kết quả lưu trước đó: lr={lr}, wd={wd} -> Val Macro-F1 = {existing[0]['val_macro_f1']:.4f} (Tiếp tục)", flush=True)
                continue

            print(f"  [Tổ hợp {combo_idx}/{total_combos}] Đang chạy: lr={lr}, weight_decay={wd}...", flush=True)
            torch.manual_seed(42)
            model = model_class(num_classes=6).to(dev)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

            for _ in range(max_epochs):
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(dev), yb.to(dev)
                    optimizer.zero_grad()
                    out = model(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()

            # Val
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(dev)
                    preds = model(xb).argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_targets.extend(yb.numpy())

            f1 = compute_per_class_metrics(np.array(all_targets), np.array(all_preds), num_classes=6)["macro_f1"]
            print(f"  --> Hoàn tất tổ hợp {combo_idx}: Val Macro-F1 = {f1:.4f}", flush=True)
            results.append({"lr": lr, "weight_decay": wd, "val_macro_f1": f1})
            if csv_save_path:
                pd.DataFrame(results).to_csv(csv_save_path, index=False)

    df_grid = pd.DataFrame(results)
    best_row = df_grid.loc[df_grid["val_macro_f1"].idxmax()]
    best_cfg = {"lr": float(best_row["lr"]), "wd": float(best_row["weight_decay"])}
    return best_cfg, df_grid
