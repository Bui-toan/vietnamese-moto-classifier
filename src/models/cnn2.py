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


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Channel Attention:
    Tự động tái định lượng trọng số các kênh đặc trưng để tập trung vào các đặc trưng quan trọng
    (như hình dạng đầu đèn, yếm xe, tem xe) thay vì phông nền.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class ResBlock(nn.Module):
    """
    Khối Residual cơ bản: 2 lớp Conv 3x3 kèm BatchNorm, SE Channel Attention và Skip Connection x + F(x).
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

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
        out = self.se(out)
        out += residual
        return F.relu(out)


class CNN2(nn.Module):
    """
    M3 ResNet-14 Nâng Cấp (Deep Residual + SE-Attention):
    Stem (kernel 5x5) -> 3 Stages (mỗi stage 2 ResBlocks: 32 -> 64 -> 128) -> GAP -> Dropout 0.3 -> FC(128 -> 6).
    Tổng cộng 6 ResBlocks (12 conv layers) tích hợp Squeeze-and-Excitation channel attention.
    Dung lượng ~702k tham số (gấp đôi M2, giải quyết triệt để vấn đề underfitting).
    """
    def __init__(self, num_classes: int = 6, dropout_p: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        # Stem: 3 -> 32 với kernel 5x5 mở rộng Receptive Field sớm bao quát form dáng xe
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # (32, 56, 56)
        )

        # Stage 1: 32 -> 32 (2 ResBlocks, resolution 56x56)
        self.layer1 = nn.Sequential(
            ResBlock(32, 32, stride=1, use_se=True),
            ResBlock(32, 32, stride=1, use_se=True)
        )

        # Stage 2: 32 -> 64 (2 ResBlocks, downsample -> 28x28)
        self.layer2 = nn.Sequential(
            ResBlock(32, 64, stride=2, use_se=True),
            ResBlock(64, 64, stride=1, use_se=True)
        )

        # Stage 3: 64 -> 128 (2 ResBlocks, downsample -> 14x14)
        self.layer3 = nn.Sequential(
            ResBlock(64, 128, stride=2, use_se=True),
            ResBlock(128, 128, stride=1, use_se=True)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(128, num_classes)

        # Biến lưu trữ feature map và gradient cho Grad-CAM tại layer cuối
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Hook cho Grad-CAM tại đầu ra layer3 (trước GAP)
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        self.activations = x

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Đầu vào: (B, 3, H, W)
        Đầu ra: Logits (B, 6)
        """
        feats = self.forward_features(x)
        gap = self.gap(feats).flatten(1)       # (B, 128)
        drop = self.dropout(gap)
        logits = self.fc(drop)                 # (B, 6)
        return logits
