"""
Mô hình M3: CNN-2 ResNet-style (src/models/cnn2.py)
Tuân thủ nghiêm ngặt mục 5.1 của kế hoạch v4:
- Stem + ResBlock, GAP + Dropout 0.3 + FC(128 -> 6).
- AdamW, cosine schedule, batch 32.
- Dùng trong ablation "có/không residual" đối chiếu với M2.
- Hỗ trợ hook trích xuất activation map phục vụ Grad-CAM.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Khối Residual cơ bản: 2 lớp Conv 3x3 kèm BatchNorm và Skip Connection x + F(x).
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class CNN2(nn.Module):
    """
    M3 ResNet: Stem -> ResBlocks (32 -> 64 -> 128) -> GAP -> Dropout 0.3 -> FC(128 -> 6).
    """
    def __init__(self, num_classes: int = 6, dropout_p: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        # Stem: 3 -> 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # (32, 56, 56)
        )

        # Stage 1: 32 -> 32
        self.layer1 = ResBlock(32, 32, stride=1)  # (32, 56, 56)

        # Stage 2: 32 -> 64
        self.layer2 = ResBlock(32, 64, stride=2)  # (64, 28, 28)

        # Stage 3: 64 -> 128
        self.layer3 = ResBlock(64, 128, stride=2) # (128, 14, 14)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(128, num_classes)

        # Biến lưu trữ feature map và gradient cho Grad-CAM
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if x.requires_grad:
            x.register_hook(self.activations_hook)
        self.activations = x

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Đầu vào: (B, 3, 112, 112)
        Đầu ra: Logits (B, 6)
        """
        feats = self.forward_features(x)
        gap = self.gap(feats).flatten(1)       # (B, 128)
        drop = self.dropout(gap)
        logits = self.fc(drop)                 # (B, 6)
        return logits
