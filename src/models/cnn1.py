"""
Mô hình M2: CNN-1 Baseline (src/models/cnn1.py)
Tuân thủ nghiêm ngặt mục 5.1 của kế hoạch v4:
- 3 conv + GAP + FC(128 -> 6).
- Batch 32, AdamW, CE + label smoothing 0.1.
- Hỗ trợ hook trích xuất activation map phục vụ Grad-CAM.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1(nn.Module):
    """
    M2 Baseline (Nâng cấp dung lượng & Receptive Field):
    3 tầng Convolution (64 -> 128 -> 256) + BatchNorm + ReLU + MaxPool -> GAP -> Dropout(0.2) -> FC(256 -> 6).
    Tầng 1 dùng kernel 5x5 để tăng Receptive Field bao quát dáng xe.
    """
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.num_classes = num_classes

        # Conv Block 1: 3 -> 64, kernel 5x5 để tăng Receptive Field sớm, MaxPool -> (64, 56, 56)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Conv Block 2: 64 -> 128, kernel 3x3, MaxPool -> (128, 28, 28)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        # Conv Block 3: 128 -> 256, kernel 3x3, MaxPool -> (256, 14, 14)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(256, num_classes)

        # Biến lưu trữ feature map và gradient cho Grad-CAM
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Trích xuất feature maps qua 3 khối conv."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        # Hook cho Grad-CAM tại conv3
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        self.activations = x

        x = self.pool(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Đầu vào: (B, 3, H, W)
        Đầu ra: Logits (B, 6)
        """
        feats = self.forward_features(x)
        gap = self.gap(feats).flatten(1)  # (B, 256)
        drop = self.dropout(gap)
        logits = self.fc(drop)            # (B, 6)
        return logits
