"""
Script: Train (or load) ANN checkpoints for all (model, dataset) combinations.

Run this once before the main sweep to ensure all checkpoints exist.
Usage:
    python scripts/train_all_anns.py
    python scripts/train_all_anns.py --config configs/experiment.yaml
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml

from src.models import get_model
from src.data.datasets import get_dataloaders, get_dataset_info
from src.training.train_ann import load_or_train_ann


def get_device(cfg: dict) -> torch.device:
    requested = cfg.get("evaluation", {}).get("device", "auto")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(requested)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model", default=None, help="Train only this model")
    parser.add_argument("--dataset", default=None, help="Train only this dataset")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg)
    print(f"Using device: {device}")

    models = [args.model] if args.model else cfg["sweep"]["models"]
    datasets = [args.dataset] if args.dataset else cfg["sweep"]["datasets"]
    checkpoint_dir = cfg["experiment"]["checkpoint_dir"]
    train_cfg = cfg["training"]

    for dataset_name in datasets:
        ds_info = get_dataset_info(dataset_name)
        num_classes = ds_info["num_classes"]
        in_channels = ds_info["in_channels"]

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} | Classes: {num_classes} | Channels: {in_channels}")

        train_loader, val_loader = get_dataloaders(
            dataset_name,
            data_root="data",
            batch_size=train_cfg["batch_size"],
            eval_batch_size=cfg["evaluation"]["batch_size"],
            seed=42,
        )

        for model_name in models:
            print(f"\n  Model: {model_name}")
            ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_{dataset_name}.pth")

            model = get_model(
                model_name,
                num_classes=num_classes,
                in_channels=in_channels,
                input_size=ds_info["input_size"],
            )

            best_acc = load_or_train_ann(
                model,
                train_loader,
                val_loader,
                checkpoint_path=ckpt_path,
                epochs=train_cfg["epochs"],
                lr=train_cfg["lr"],
                weight_decay=train_cfg["weight_decay"],
                scheduler=train_cfg["scheduler"],
                device=device,
                verbose=True,
            )
            print(f"  → Best val acc: {best_acc * 100:.2f}%")

    print("\n\nAll ANN checkpoints ready.")


if __name__ == "__main__":
    main()
