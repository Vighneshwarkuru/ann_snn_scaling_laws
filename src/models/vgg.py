"""
VGG Family (VGG-9, VGG-11, VGG-16) ANN for ANN-to-SNN conversion experiments.

Key modifications vs. standard VGG:
- AvgPool instead of MaxPool (required for SpikingJelly threshold normalization)
- BatchNorm after every conv (required for threshold normalization)
- Adaptive avg pool before classifier (handles variable input sizes)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Union


# ---------------------------------------------------------------------------
# VGG configurations: int = out_channels, 'A' = AvgPool2d(2,2)
# Depth counts: conv layers only
# ---------------------------------------------------------------------------

VGG_CFGS: Dict[str, List[Union[int, str]]] = {
    # VGG-9: 6 conv + 3 FC = 9 layers (depth 6 for conv count)
    "vgg9": [
        64, "A",
        128, "A",
        256, 256, "A",
        512, 512, "A",
    ],
    # VGG-11: 8 conv + 3 FC = 11 layers
    "vgg11": [
        64, "A",
        128, "A",
        256, 256, "A",
        512, 512, "A",
        512, 512, "A",
    ],
    # VGG-16: 13 conv + 3 FC = 16 layers
    "vgg16": [
        64, 64, "A",
        128, 128, "A",
        256, 256, 256, "A",
        512, 512, 512, "A",
        512, 512, 512, "A",
    ],
}

VGG_DEPTHS = {"vgg9": 9, "vgg11": 11, "vgg16": 16}


def make_layers(cfg: List[Union[int, str]], in_channels: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    for v in cfg:
        if v == "A":
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            layers += [
                nn.Conv2d(in_channels, int(v), kernel_size=3, padding=1),
                nn.BatchNorm2d(int(v)),
                nn.ReLU(inplace=True),
            ]
            in_channels = int(v)
    return nn.Sequential(*layers)


class VGG(nn.Module):
    """
    VGG with BatchNorm and AvgPool, compatible with ANN-to-SNN conversion.
    Works for VGG-9, VGG-11, VGG-16.
    """

    def __init__(self, cfg_name: str, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        cfg = VGG_CFGS[cfg_name]
        self.features = make_layers(cfg, in_channels=in_channels)
        # AdaptiveAvgPool2d(1,1): after 4-5 AvgPool2d on 32×32 input → 1×1 or 2×2
        # (1,1) is always divisible on any size, and SpikingJelly can trace this
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Get final channel count from cfg
        final_channels = [v for v in cfg if isinstance(v, int)][-1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_channels, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


def VGG9(num_classes: int = 10, in_channels: int = 3) -> VGG:
    return VGG("vgg9", num_classes=num_classes, in_channels=in_channels)


def VGG11(num_classes: int = 10, in_channels: int = 3) -> VGG:
    return VGG("vgg11", num_classes=num_classes, in_channels=in_channels)


def VGG16(num_classes: int = 10, in_channels: int = 3) -> VGG:
    return VGG("vgg16", num_classes=num_classes, in_channels=in_channels)
