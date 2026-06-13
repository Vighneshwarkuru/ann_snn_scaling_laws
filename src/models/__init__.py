from .lenet import LeNet
from .vgg import VGG9, VGG11, VGG16
from .resnet import ResNet18, ResNet34

MODEL_REGISTRY = {
    "lenet":    LeNet,
    "vgg9":     VGG9,
    "vgg11":    VGG11,
    "vgg16":    VGG16,
    "resnet18": ResNet18,
    "resnet34": ResNet34,
}

# Approximate layer depth for scaling law analysis
MODEL_DEPTHS = {
    "lenet":    4,
    "vgg9":     9,
    "vgg11":    11,
    "vgg16":    16,
    "resnet18": 18,
    "resnet34": 34,
}

def get_model(name: str, num_classes: int, in_channels: int = 3, input_size: int = 32):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    cls = MODEL_REGISTRY[name]
    if name == "lenet":
        return cls(num_classes=num_classes, in_channels=in_channels, input_size=input_size)
    return cls(num_classes=num_classes, in_channels=in_channels)
