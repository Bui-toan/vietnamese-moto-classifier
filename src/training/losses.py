"""
Module các hàm mất mát tự code (src/training/losses.py)
Tuân thủ nghiêm ngặt mục 5.1 của kế hoạch v4:
- MulticlassHingeLoss: Tự code bằng NumPy cho LinearSVM (margin 1, L2 regularization).
- CrossEntropyWithLabelSmoothing: Hàm mất mát CE kết hợp Label Smoothing (smoothing 0.1) cho CNN.
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MulticlassHingeLoss:
    """
    Multiclass Hinge Loss tự code 100% bằng NumPy cho Linear SVM:
    L = (1/N) * sum_i sum_{j != y_i} max(0, s_j - s_{y_i} + margin) + 0.5 * wd * ||W||_F^2
    """
    def __init__(self, margin: float = 1.0, weight_decay: float = 1e-4):
        self.margin = margin
        self.weight_decay = weight_decay

    def __call__(
        self,
        scores: np.ndarray,
        y: np.ndarray,
        W: np.ndarray,
        X: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Tính loss và gradient theo W, b.
        
        Args:
            scores: (N, C) logits
            y: (N,) ground-truth class indices
            W: (D, C) ma trận trọng số
            X: (N, D) ma trận đặc trưng
            
        Returns:
            (total_loss, dW, db)
        """
        N = X.shape[0]
        correct_scores = scores[np.arange(N), y][:, np.newaxis]

        margins = scores - correct_scores + self.margin
        margins[np.arange(N), y] = 0.0
        margins = np.maximum(0.0, margins)

        data_loss = np.sum(margins) / N
        reg_loss = 0.5 * self.weight_decay * np.sum(W ** 2)
        total_loss = float(data_loss + reg_loss)

        # Gradient
        binary = (margins > 0).astype(np.float32)
        count_violated = np.sum(binary, axis=1)
        binary[np.arange(N), y] = -count_violated

        dW = np.dot(X.T, binary) / N + self.weight_decay * W
        db = np.sum(binary, axis=0) / N

        return total_loss, dW.astype(np.float32), db.astype(np.float32)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss kết hợp Label Smoothing cho CNN (mục 5.1: smoothing=0.1).
    """
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C)
        target: (B,)
        """
        logprobs = F.log_softmax(logits, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
