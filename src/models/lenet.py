"""
LeNet-style CNN for ANN-to-SNN conversion experiments.

Uses BatchNorm after every conv (required for threshold normalization)
and AvgPool instead of MaxPool (SpikingJelly conversion compatible).

No adaptive pooling — uses a fixed AvgPool sized to each dataset's spatial output
so there are no MPS divisibility issues and torch.fx tracing works cleanly.
"""

import torch
import torch.nn as nn


class LeNet(nn.Module):
    """
    LeNet variant compatible with ANN-to-SNN conversion.
    Supports 1-channel (MNIST/FashionMNIST) and 3-channel (CIFAR/Tiny-ImageNet) inputs.

    After two AvgPool2d(2,2), spatial sizes are:
      28×28  → 7×7   (MNIST, FashionMNIST)
      32×32  → 8×8   (CIFAR-10, CIFAR-100)
      64×64  → 16×16 (Tiny-ImageNet)

    A final AvgPool2d(kernel=spatial, stride=spatial) collapses to 1×1 using
    integer strides, which is fully compatible with MPS and torch.fx tracing.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1, input_size: int = 28):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # Spatial size after two AvgPool2d(2,2)
        spatial = input_size // 4
        self.pool = nn.AvgPool2d(kernel_size=spatial, stride=spatial)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
