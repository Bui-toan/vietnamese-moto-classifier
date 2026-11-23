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
    M2 Baseline: 3 tầng Convolution + BatchNorm + ReLU + MaxPool -> GAP -> FC(128 -> 6).
    """
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.num_classes = num_classes

        # Conv Block 1: 3 -> 32, MaxPool -> (32, 56, 56)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Conv Block 2: 32 -> 64, MaxPool -> (64, 28, 28)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # Conv Block 3: 64 -> 128, MaxPool -> (128, 14, 14)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

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
        Đầu vào: (B, 3, 112, 112)
        Đầu ra: Logits (B, 6)
        """
        feats = self.forward_features(x)
        gap = self.gap(feats).flatten(1)  # (B, 128)
        logits = self.fc(gap)            # (B, 6)
        return logits
