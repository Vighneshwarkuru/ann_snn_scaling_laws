"""
Dataset loaders for the ANN-to-SNN scaling laws project.

Supports: MNIST, FashionMNIST, CIFAR-10, CIFAR-100, Tiny-ImageNet.
Each dataset returns (train_loader, test_loader) with standardized preprocessing.
"""

import os
import zipfile
import urllib.request
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.datasets import (
    MNIST,
    FashionMNIST,
    CIFAR10,
    CIFAR100,
    ImageFolder,
)


# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------

DATASET_INFO: Dict[str, Dict[str, Any]] = {
    "mnist": {
        "num_classes": 10,
        "in_channels": 1,
        "input_size": 28,
        "complexity_rank": 1,
        "mean": (0.1307,),
        "std": (0.3081,),
    },
    "fashion_mnist": {
        "num_classes": 10,
        "in_channels": 1,
        "input_size": 28,
        "complexity_rank": 2,
        "mean": (0.2860,),
        "std": (0.3530,),
    },
    "cifar10": {
        "num_classes": 10,
        "in_channels": 3,
        "input_size": 32,
        "complexity_rank": 3,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    },
    "cifar100": {
        "num_classes": 100,
        "in_channels": 3,
        "input_size": 32,
        "complexity_rank": 4,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    "tiny_imagenet": {
        "num_classes": 200,
        "in_channels": 3,
        "input_size": 64,
        "complexity_rank": 5,
        "mean": (0.4802, 0.4481, 0.3975),
        "std": (0.2770, 0.2691, 0.2821),
    },
}

DATASET_REGISTRY = list(DATASET_INFO.keys())

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def get_dataset_info(name: str) -> Dict[str, Any]:
    if name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset '{name}'. Available: {DATASET_REGISTRY}")
    return DATASET_INFO[name]


# ---------------------------------------------------------------------------
# Tiny-ImageNet helpers
# ---------------------------------------------------------------------------

def _download_tiny_imagenet(root: str) -> str:
    """Download and extract Tiny-ImageNet if not already present."""
    root = Path(root)
    extract_dir = root / "tiny-imagenet-200"

    if extract_dir.exists():
        return str(extract_dir)

    zip_path = root / "tiny-imagenet-200.zip"
    print(f"Downloading Tiny-ImageNet to {zip_path} ...")
    root.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path)

    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)

    zip_path.unlink()  # remove zip after extraction
    return str(extract_dir)


def _prepare_tiny_imagenet_val(val_dir: str) -> None:
    """
    Re-organise the Tiny-ImageNet val/ folder into class sub-directories
    so that ImageFolder can load it correctly.
    """
    val_dir = Path(val_dir)
    annotations_file = val_dir / "val_annotations.txt"

    if not annotations_file.exists():
        return  # already organised

    # Parse annotations
    img_to_class: Dict[str, str] = {}
    with open(annotations_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            img_to_class[parts[0]] = parts[1]

    images_dir = val_dir / "images"
    for img_name, cls in img_to_class.items():
        cls_dir = val_dir / cls
        cls_dir.mkdir(exist_ok=True)
        src = images_dir / img_name
        dst = cls_dir / img_name
        if src.exists() and not dst.exists():
            src.rename(dst)

    # Remove now-empty images dir
    if images_dir.exists() and not any(images_dir.iterdir()):
        images_dir.rmdir()

    annotations_file.unlink()


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------

def _get_transforms(name: str, train: bool) -> T.Compose:
    info = DATASET_INFO[name]
    mean, std = info["mean"], info["std"]
    size = info["input_size"]

    normalize = T.Normalize(mean=mean, std=std)

    if train:
        if name in ("mnist", "fashion_mnist"):
            return T.Compose([
                T.RandomCrop(size, padding=4),
                T.ToTensor(),
                normalize,
            ])
        elif name in ("cifar10", "cifar100"):
            return T.Compose([
                T.RandomCrop(size, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize,
            ])
        else:  # tiny_imagenet
            return T.Compose([
                T.RandomCrop(size, padding=8),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize,
            ])
    else:
        return T.Compose([
            T.Resize(size) if name == "tiny_imagenet" else T.Lambda(lambda x: x),
            T.ToTensor(),
            normalize,
        ])


# ---------------------------------------------------------------------------
# Worker init (must be module-level to be picklable by multiprocessing)
# ---------------------------------------------------------------------------

def _seed_worker(worker_id: int) -> None:
    import numpy as np
    import random
    # Each worker gets a unique seed derived from the generator seed
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_dataloaders(
    dataset_name: str,
    data_root: str = "data",
    batch_size: int = 128,
    eval_batch_size: int = 256,
    seed: int = 42,
    num_workers: int = -1,  # -1 = auto: 4 on CUDA, 0 on MPS/CPU
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, test_loader) for the given dataset.

    Args:
        dataset_name:   One of DATASET_REGISTRY.
        data_root:      Directory where datasets are downloaded/stored.
        batch_size:     Batch size for training loader.
        eval_batch_size: Batch size for evaluation loader.
        seed:           Random seed for worker init.
        num_workers:    DataLoader workers.

    Returns:
        (train_loader, test_loader)
    """
    if dataset_name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {DATASET_REGISTRY}")

    # Auto-select num_workers: multiprocessing works well on CUDA, but on MPS/CPU
    # use 0 (single-process) to avoid pickle issues with spawned workers.
    if num_workers == -1:
        num_workers = 4 if torch.cuda.is_available() else 0

    train_tf = _get_transforms(dataset_name, train=True)
    test_tf = _get_transforms(dataset_name, train=False)

    root = os.path.join(data_root, dataset_name)

    g = torch.Generator()
    g.manual_seed(seed)

    # pin_memory only works on CUDA; disable on MPS and CPU
    pin = torch.cuda.is_available()

    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin,
        worker_init_fn=_seed_worker,
        generator=g,
    )

    if dataset_name == "mnist":
        train_ds = MNIST(root, train=True, download=True, transform=train_tf)
        test_ds = MNIST(root, train=False, download=True, transform=test_tf)

    elif dataset_name == "fashion_mnist":
        train_ds = FashionMNIST(root, train=True, download=True, transform=train_tf)
        test_ds = FashionMNIST(root, train=False, download=True, transform=test_tf)

    elif dataset_name == "cifar10":
        train_ds = CIFAR10(root, train=True, download=True, transform=train_tf)
        test_ds = CIFAR10(root, train=False, download=True, transform=test_tf)

    elif dataset_name == "cifar100":
        train_ds = CIFAR100(root, train=True, download=True, transform=train_tf)
        test_ds = CIFAR100(root, train=False, download=True, transform=test_tf)

    elif dataset_name == "tiny_imagenet":
        ti_root = _download_tiny_imagenet(data_root)
        val_dir = os.path.join(ti_root, "val")
        _prepare_tiny_imagenet_val(val_dir)
        train_ds = ImageFolder(os.path.join(ti_root, "train"), transform=train_tf)
        test_ds = ImageFolder(val_dir, transform=test_tf)

    else:
        raise ValueError(f"Unhandled dataset: {dataset_name}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_batch_size, shuffle=False, **loader_kwargs
    )

    return train_loader, test_loader
