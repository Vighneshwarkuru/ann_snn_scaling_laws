"""
Main experiment sweep script.

Iterates over all (dataset, model, timestep, seed) combinations,
converts ANN → SNN, runs inference, collects metrics, and writes results.

Usage:
    python scripts/run_sweep.py
    python scripts/run_sweep.py --config configs/experiment.yaml
    python scripts/run_sweep.py --dry-run            # 1 dataset, 1 model, 1 seed, T=[4,8]
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml

from src.models import get_model
from src.data.datasets import get_dataloaders, get_dataset_info
from src.training.train_ann import load_or_train_ann
from src.conversion.ann_to_snn import ANNtoSNNConverter
from src.evaluation.evaluate_ann import evaluate_ann
from src.evaluation.evaluate_snn import evaluate_snn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_depth(model_name: str) -> int:
    """Approximate layer depth for each model (used in scaling analysis)."""
    depth_map = {"lenet": 4, "vgg11": 11, "resnet18": 18, "resnet34": 34}
    return depth_map.get(model_name, 0)


def results_csv_path(raw_dir: str, model_name: str, dataset: str, T: int, seed: int) -> str:
    return os.path.join(raw_dir, f"{model_name}_{dataset}_T{T}_seed{seed}.csv")


FIELDNAMES = [
    "model", "dataset", "T", "seed", "depth",
    "ann_accuracy", "snn_accuracy", "accuracy_drop",
    "total_spikes", "avg_spikes_per_sample", "total_neurons", "spike_density",
    "snn_latency_ms_per_sample", "ann_latency_ms_per_sample",
    "E_SNN_pJ", "E_ANN_pJ", "energy_ratio",
    "synops", "mac_count",
    "memory_mb", "gpu_memory_mb",
]


def write_row(path: str, row: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Quick test: 1 dataset (mnist), 1 model (lenet), T=[4,8], 1 seed"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg)
    print(f"Using device: {device}\n")

    sweep = cfg["sweep"]
    train_cfg = cfg["training"]
    conv_cfg = cfg["conversion"]
    eval_cfg = cfg["evaluation"]
    output_dir = cfg["experiment"]["output_dir"]
    checkpoint_dir = cfg["experiment"]["checkpoint_dir"]
    raw_dir = os.path.join(output_dir, "raw")

    # Override for dry-run
    if args.dry_run:
        sweep = dict(
            datasets=["mnist"],
            models=["lenet"],
            timesteps=[4, 8],
            seeds=[42],
        )
        train_cfg = dict(train_cfg, epochs=3)
        print("[DRY-RUN MODE] Reduced sweep: mnist/lenet/T=[4,8]/seed=42/epochs=3\n")

    total_runs = (
        len(sweep["datasets"])
        * len(sweep["models"])
        * len(sweep["timesteps"])
        * len(sweep["seeds"])
    )
    run_idx = 0

    for dataset_name in sweep["datasets"]:
        ds_info = get_dataset_info(dataset_name)
        num_classes = ds_info["num_classes"]
        in_channels = ds_info["in_channels"]
        complexity_rank = ds_info["complexity_rank"]
        input_size = (in_channels, ds_info["input_size"], ds_info["input_size"])

        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name}  (complexity rank {complexity_rank})")

        for model_name in sweep["models"]:
            print(f"\n  MODEL: {model_name}")
            ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_{dataset_name}.pth")

            # ----- Load or train ANN (once per model/dataset) -----
            set_seed(42)
            ann = get_model(
                model_name,
                num_classes=num_classes,
                in_channels=in_channels,
                input_size=ds_info["input_size"],
            )

            train_loader, val_loader = get_dataloaders(
                dataset_name,
                data_root="data",
                batch_size=train_cfg["batch_size"],
                eval_batch_size=eval_cfg["batch_size"],
                seed=42,
            )

            ann_best_acc = load_or_train_ann(
                ann, train_loader, val_loader,
                checkpoint_path=ckpt_path,
                epochs=train_cfg["epochs"],
                lr=train_cfg["lr"],
                weight_decay=train_cfg["weight_decay"],
                scheduler=train_cfg["scheduler"],
                device=device,
                verbose=True,
            )

            # ----- Evaluate ANN baseline -----
            ann_metrics = evaluate_ann(ann, val_loader, device, input_size, verbose=True)

            # ----- Calibration loader for threshold normalization -----
            calib_loader, _ = get_dataloaders(
                dataset_name,
                data_root="data",
                batch_size=train_cfg["batch_size"],
                eval_batch_size=eval_cfg["batch_size"],
                seed=42,
            )

            for seed in sweep["seeds"]:
                set_seed(seed)

                for T in sweep["timesteps"]:
                    run_idx += 1
                    out_path = results_csv_path(raw_dir, model_name, dataset_name, T, seed)

                    if os.path.exists(out_path):
                        print(f"  [{run_idx}/{total_runs}] SKIP (exists): {os.path.basename(out_path)}")
                        continue

                    print(f"\n  [{run_idx}/{total_runs}] "
                          f"{model_name} | {dataset_name} | T={T} | seed={seed}")

                    # Convert ANN → SNN (always on CPU for torch.fx compatibility)
                    try:
                        converter = ANNtoSNNConverter(
                            ann, T=T,
                            norm_percentile=conv_cfg["norm_percentile"],
                            device=torch.device("cpu"),
                        )
                        snn = converter.convert(
                            calib_loader,
                            n_calib_batches=conv_cfg["calibration_batches"],
                        )
                        snn = snn.to(device)  # move to target device after conversion
                    except Exception as e:
                        print(f"  [ERROR] Conversion failed: {e}")
                        continue

                    # Evaluate SNN
                    _, test_loader = get_dataloaders(
                        dataset_name,
                        data_root="data",
                        batch_size=train_cfg["batch_size"],
                        eval_batch_size=eval_cfg["batch_size"],
                        seed=seed,
                    )

                    try:
                        snn_metrics = evaluate_snn(
                            snn, test_loader, T=T,
                            device=device,
                            mac_count_ann=ann_metrics["mac_count"],
                            verbose=True,
                        )
                    except Exception as e:
                        print(f"  [ERROR] SNN evaluation failed: {e}")
                        continue

                    # Compose result row
                    row = {
                        "model": model_name,
                        "dataset": dataset_name,
                        "T": T,
                        "seed": seed,
                        "depth": model_depth(model_name),
                        "ann_accuracy": ann_metrics["ann_accuracy"],
                        "snn_accuracy": snn_metrics["snn_accuracy"],
                        "accuracy_drop": ann_metrics["ann_accuracy"] - snn_metrics["snn_accuracy"],
                        "total_spikes": snn_metrics["total_spikes"],
                        "avg_spikes_per_sample": snn_metrics["avg_spikes_per_sample"],
                        "total_neurons": snn_metrics["total_neurons"],
                        "spike_density": snn_metrics["spike_density"],
                        "snn_latency_ms_per_sample": snn_metrics["snn_latency_ms_per_sample"],
                        "ann_latency_ms_per_sample": ann_metrics["ann_latency_ms_per_sample"],
                        "E_SNN_pJ": snn_metrics["E_SNN_pJ"],
                        "E_ANN_pJ": snn_metrics["E_ANN_pJ"],
                        "energy_ratio": snn_metrics["energy_ratio"],
                        "synops": snn_metrics["synops"],
                        "mac_count": ann_metrics["mac_count"],
                        "memory_mb": snn_metrics["memory_mb"],
                        "gpu_memory_mb": snn_metrics["gpu_memory_mb"],
                    }

                    write_row(out_path, row)
                    print(f"  Saved → {out_path}")

    print(f"\n\nSweep complete. {run_idx} runs processed.")
    print(f"Raw results in: {raw_dir}/")


if __name__ == "__main__":
    main()
