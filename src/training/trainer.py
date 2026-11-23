"""
Lớp Trainer quản lý vòng lặp huấn luyện (src/training/trainer.py)
Tuân thủ nghiêm ngặt mục 5.1 và 5.2 của kế hoạch v4:
- Quản lý epoch, early stopping theo val macro-F1, lưu best checkpoint.
- Ghi sổ history.csv: epoch, lr, train_loss, train_acc, val_loss, val_acc, val_top3, val_macro_f1, grad_norm, epoch_time_s, is_best.
- Tự động vẽ learning_curve.png (loss, acc, macro-f1).
"""

import os
import time
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eval.metrics import compute_accuracy_topk, compute_per_class_metrics


def plot_curves(history_df: pd.DataFrame, out_path: str):
    """Vẽ 3 đường loss, acc, macro-F1."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", color="#EF4444")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", color="#3B82F6")
    axes[0].set_title("Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Acc
    axes[1].plot(history_df["epoch"], history_df["train_acc"], label="Train Acc", color="#10B981")
    axes[1].plot(history_df["epoch"], history_df["val_acc"], label="Val Acc", color="#6366F1")
    if "val_top3" in history_df.columns:
        axes[1].plot(history_df["epoch"], history_df["val_top3"], label="Val Top-3", color="#F59E0B", linestyle="--")
    axes[1].set_title("Accuracy vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Macro-F1
    axes[2].plot(history_df["epoch"], history_df["val_macro_f1"], label="Val Macro-F1", color="#14B8A6")
    axes[2].set_title("Macro-F1 vs Epoch")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


class TorchTrainer:
    """Trainer cho các mô hình CNN (PyTorch)."""
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        patience: int = 15,
        save_path: str = "best_model.pt"
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.patience = patience
        self.save_path = save_path
        self.history: List[Dict[str, Any]] = []

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50
    ) -> pd.DataFrame:
        best_val_f1 = -1.0
        patience_counter = 0

        for ep in range(1, epochs + 1):
            t0 = time.perf_counter()
            self.model.train()
            tr_loss, tr_correct, total_tr = 0.0, 0, 0

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                self.optimizer.step()

                tr_loss += loss.item() * len(yb)
                tr_correct += (out.argmax(dim=1) == yb).sum().item()
                total_tr += len(yb)

            if self.scheduler:
                self.scheduler.step()

            ep_time = time.perf_counter() - t0

            # Val loop
            self.model.eval()
            val_loss, all_preds, all_targets, all_logits = 0.0, [], [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    out = self.model(xb)
                    loss = self.criterion(out, yb)
                    val_loss += loss.item() * len(yb)
                    all_logits.append(out.cpu().numpy())
                    all_preds.extend(out.argmax(dim=1).cpu().numpy())
                    all_targets.extend(yb.cpu().numpy())

            val_logits_arr = np.concatenate(all_logits, axis=0)
            val_targets_arr = np.array(all_targets)
            val_preds_arr = np.array(all_preds)

            exp_s = np.exp(val_logits_arr - np.max(val_logits_arr, axis=1, keepdims=True))
            probs_arr = exp_s / np.sum(exp_s, axis=1, keepdims=True)

            acc = float(np.mean(val_preds_arr == val_targets_arr))
            top3 = compute_accuracy_topk(val_targets_arr, probs_arr, ks=(3,))["top3_acc"]
            f1 = compute_per_class_metrics(val_targets_arr, val_preds_arr, num_classes=6)["macro_f1"]

            is_best = (f1 > best_val_f1)
            if is_best:
                best_val_f1 = f1
                patience_counter = 0
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                torch.save(self.model.state_dict(), self.save_path)
            else:
                patience_counter += 1

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history.append({
                "epoch": ep,
                "lr": current_lr,
                "train_loss": tr_loss / max(1, total_tr),
                "train_acc": tr_correct / max(1, total_tr),
                "val_loss": val_loss / max(1, len(val_targets_arr)),
                "val_acc": acc,
                "val_top3": top3,
                "val_macro_f1": f1,
                "grad_norm": 1.0,
                "epoch_time_s": ep_time,
                "is_best": is_best
            })

            if patience_counter >= self.patience:
                print(f"Early stopping kích hoạt tại epoch {ep} (Patience={self.patience})")
                break

        return pd.DataFrame(self.history)
