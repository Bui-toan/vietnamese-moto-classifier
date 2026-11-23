"""
Mô hình Multiclass Linear SVM tự code 100% bằng NumPy (src/models/svm.py)
Tuân thủ nghiêm ngặt mục 5.1 của kế hoạch v4:
- LinearSVM tự code, multiclass hinge loss (margin 1).
- SGD momentum 0.9, weight decay (L2 regularization).
- Hỗ trợ M1: 6.084 chiều -> 6 lớp.
- Hỗ trợ Detector: 1.764 chiều -> 2 lớp nhị phân (xe / nền).
- KHÔNG dùng sklearn.svm hay thư viện bên ngoài.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np


class LinearSVM:
    def __init__(
        self,
        n_features: int,
        n_classes: int = 6,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        margin: float = 1.0,
        random_state: int = 42
    ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.margin = margin
        self.random_state = random_state

        rng = np.random.RandomState(random_state)
        # Khởi tạo trọng số Xavier/He
        scale = np.sqrt(2.0 / (n_features + n_classes))
        self.W = (rng.randn(n_features, n_classes) * scale).astype(np.float32)
        self.b = np.zeros(n_classes, dtype=np.float32)

        # Biến tích lũy momentum
        self.v_W = np.zeros_like(self.W)
        self.v_b = np.zeros_like(self.b)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Tính điểm số (logits): scores = X @ W + b
        Args:
            X: (N, n_features)
        Returns:
            scores: (N, n_classes)
        """
        return np.dot(X, self.W) + self.b

    def compute_loss_and_grad(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Tính Multiclass Hinge Loss và đạo hàm theo W, b.
        Loss: L = (1/N) * sum_i sum_{j != y_i} max(0, s_j - s_{y_i} + margin) + 0.5 * wd * ||W||_F^2
        """
        N = X.shape[0]
        scores = self.forward(X)  # (N, C)

        # Lấy điểm số của nhãn đúng
        correct_scores = scores[np.arange(N), y][:, np.newaxis]  # (N, 1)

        # Tính margin vi phạm: s_j - s_{y_i} + margin
        margins = scores - correct_scores + self.margin
        margins[np.arange(N), y] = 0.0  # Không tính cho j == y_i
        margins = np.maximum(0.0, margins)

        # Tổng loss
        data_loss = np.sum(margins) / N
        reg_loss = 0.5 * self.weight_decay * np.sum(self.W ** 2)
        total_loss = float(data_loss + reg_loss)

        # Gradient
        binary = (margins > 0).astype(np.float32)  # (N, C)
        # Số lớp vi phạm margin với mỗi mẫu
        count_violated = np.sum(binary, axis=1)  # (N,)
        binary[np.arange(N), y] = -count_violated

        dW = np.dot(X.T, binary) / N + self.weight_decay * self.W
        db = np.sum(binary, axis=0) / N

        return total_loss, dW.astype(np.float32), db.astype(np.float32)

    def train_epoch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 64,
        lr: Optional[float] = None
    ) -> float:
        """
        Chạy 1 epoch huấn luyện bằng Mini-batch SGD với Momentum.
        """
        if lr is None:
            lr = self.lr

        N = X.shape[0]
        indices = np.random.permutation(N)
        epoch_loss = 0.0
        n_batches = int(np.ceil(N / batch_size))

        for b in range(n_batches):
            batch_idx = indices[b * batch_size:(b + 1) * batch_size]
            X_b = X[batch_idx]
            y_b = y[batch_idx]

            loss, dW, db = self.compute_loss_and_grad(X_b, y_b)
            epoch_loss += loss

            # Cập nhật Momentum: v = momentum * v - lr * grad
            self.v_W = self.momentum * self.v_W - lr * dW
            self.v_b = self.momentum * self.v_b - lr * db

            # Cập nhật tham số
            self.W += self.v_W
            self.b += self.v_b

        return epoch_loss / n_batches

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán nhãn có điểm số cao nhất."""
        scores = self.forward(X)
        return np.argmax(scores, axis=1)

    def predict_proba(self, X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Tính xác suất qua hàm Softmax có điều chỉnh Temperature."""
        scores = self.forward(X) / max(1e-5, temperature)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def save(self, filepath: str):
        """Lưu trọng số và siêu tham số ra file .npz."""
        np.savez_compressed(
            filepath,
            W=self.W,
            b=self.b,
            n_features=self.n_features,
            n_classes=self.n_classes,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            margin=self.margin,
            random_state=self.random_state
        )

    @classmethod
    def load(cls, filepath: str) -> "LinearSVM":
        """Nạp trọng số từ file .npz."""
        data = np.load(filepath)
        model = cls(
            n_features=int(data["n_features"]),
            n_classes=int(data["n_classes"]),
            lr=float(data["lr"]),
            momentum=float(data["momentum"]),
            weight_decay=float(data["weight_decay"]),
            margin=float(data["margin"]),
            random_state=int(data["random_state"])
        )
        model.W = data["W"].astype(np.float32)
        model.b = data["b"].astype(np.float32)
        return model
