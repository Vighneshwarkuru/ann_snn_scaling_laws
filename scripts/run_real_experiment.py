"""
Real experiment runner — optimized for MacBook (no discrete GPU).

This script runs a reduced but scientifically valid experiment sweep:
- 2 models (VGG-9, ResNet-18) on 2 datasets (MNIST, CIFAR-10)
- 6 timesteps (1, 4, 8, 16, 32, 64)
- 1 seed (42) — for speed; add more seeds for statistical validation later
- Reduced epochs (15 for MNIST, 25 for CIFAR-10) — still converges

Goal: Verify that the scaling laws (log accuracy vs T, linear spikes vs T)
emerge from REAL trained models and REAL ANN-SNN conversion.

Estimated runtime: ~30-60 min on Apple M-series, ~1-2 hrs on Intel Mac.

Usage:
    python scripts/run_real_experiment.py
    python scripts/run_real_experiment.py --fast   # Even faster: MNIST only, T=[4,16,64]
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

from src.models import get_model
from src.data.datasets import get_dataloaders, get_dataset_info
from src.training.train_ann import load_or_train_ann
from src.conversion.ann_to_snn import ANNtoSNNConverter
from src.evaluation.evaluate_ann import evaluate_ann
from src.evaluation.evaluate_snn import evaluate_snn


def get_device():
    """Pick best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



FIELDNAMES = [
    "model", "dataset", "T", "seed", "depth",
    "ann_accuracy", "snn_accuracy", "accuracy_drop",
    "total_spikes", "avg_spikes_per_sample", "total_neurons", "spike_density",
    "snn_latency_ms_per_sample", "ann_latency_ms_per_sample",
    "E_SNN_pJ", "E_ANN_pJ", "energy_ratio",
    "synops", "mac_count",
    "memory_mb", "gpu_memory_mb",
]

MODEL_DEPTH = {"lenet": 4, "vgg9": 9, "vgg11": 11, "vgg16": 16, "resnet18": 18, "resnet34": 34}

# Training configs per dataset (fewer epochs for simpler datasets)
TRAIN_CONFIGS = {
    "mnist": {"epochs": 10, "lr": 0.01, "batch_size": 128},
    "fashion_mnist": {"epochs": 10, "lr": 0.01, "batch_size": 128},
    "cifar10": {"epochs": 15, "lr": 0.01, "batch_size": 128},
}


def write_row(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_single_experiment(model_name, dataset_name, T, seed, device, output_dir):
    """Run a single (model, dataset, T, seed) experiment. Returns result dict or None."""
    
    out_path = os.path.join(output_dir, "raw", f"{model_name}_{dataset_name}_T{T}_seed{seed}.csv")
    if os.path.exists(out_path):
        print(f"    SKIP (exists): {os.path.basename(out_path)}")
        return None

    set_seed(seed)
    
    ds_info = get_dataset_info(dataset_name)
    tcfg = TRAIN_CONFIGS[dataset_name]
    
    # Build model
    model = get_model(
        model_name,
        num_classes=ds_info["num_classes"],
        in_channels=ds_info["in_channels"],
        input_size=ds_info["input_size"],
    )
    
    # Get data
    train_loader, val_loader = get_dataloaders(
        dataset_name, data_root="data",
        batch_size=tcfg["batch_size"],
        eval_batch_size=256,
        seed=seed,
    )
    
    # Train or load ANN
    ckpt_path = f"checkpoints/{model_name}_{dataset_name}.pth"
    ann_acc = load_or_train_ann(
        model, train_loader, val_loader,
        checkpoint_path=ckpt_path,
        epochs=tcfg["epochs"],
        lr=tcfg["lr"],
        weight_decay=5e-4,
        scheduler="cosine",
        device=device,
        verbose=False,
    )
    print(f"    ANN trained: {ann_acc*100:.2f}%")
    
    # Evaluate ANN
    input_size = (ds_info["in_channels"], ds_info["input_size"], ds_info["input_size"])
    ann_metrics = evaluate_ann(model, val_loader, device, input_size, verbose=False)
    
    # Convert ANN → SNN (must be on CPU for torch.fx)
    try:
        calib_loader, _ = get_dataloaders(
            dataset_name, data_root="data",
            batch_size=64, eval_batch_size=128, seed=42,
        )
        converter = ANNtoSNNConverter(
            model, T=T, norm_percentile=0.99, device=torch.device("cpu"),
        )
        snn = converter.convert(calib_loader, n_calib_batches=4)
        # Move SNN to target device for evaluation
        # CUDA works fine; MPS can be unstable with SpikingJelly
        eval_device = device if device.type == "cuda" else torch.device("cpu")
        snn = snn.to(eval_device)
    except Exception as e:
        print(f"    [ERROR] Conversion failed: {e}")
        return None
    
    # Evaluate SNN
    _, test_loader = get_dataloaders(
        dataset_name, data_root="data",
        batch_size=64, eval_batch_size=256 if device.type == "cuda" else 128, seed=seed,
    )
    
    try:
        snn_metrics = evaluate_snn(
            snn, test_loader, T=T,
            device=eval_device,
            mac_count_ann=ann_metrics["mac_count"],
            max_samples=500,
            verbose=False,
        )
    except Exception as e:
        print(f"    [ERROR] SNN eval failed: {e}")
        return None
    
    # Build result row
    row = {
        "model": model_name,
        "dataset": dataset_name,
        "T": T,
        "seed": seed,
        "depth": MODEL_DEPTH[model_name],
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
    print(f"    SNN T={T}: Acc={snn_metrics['snn_accuracy']*100:.2f}% | "
          f"Spikes={snn_metrics['avg_spikes_per_sample']:.0f} | "
          f"E={snn_metrics['E_SNN_pJ']:.1f} pJ")
    
    return row


def main():
    parser = argparse.ArgumentParser(description="Run real ANN-SNN scaling experiment")
    parser.add_argument("--fast", action="store_true",
                        help="Fastest mode: MNIST only, VGG-9, T=[4,16,64]")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 2 datasets, 2 models, T=[4,16,64], 500 samples")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run single dataset only: mnist, fashion_mnist, or cifar10")
    parser.add_argument("--output-dir", default="results_real",
                        help="Output directory for real results")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}/")

    if args.fast:
        # Minimal experiment: proves concept in ~3 min
        datasets = ["mnist"]
        models = ["vgg9"]
        timesteps = [4, 16, 64]
        seeds = [42]
        print("\n[FAST MODE] MNIST / VGG-9 / T=[4,16,64] / seed=42")
    elif args.quick:
        # Quick mode: enough for multi-line plots, ~15 min on GPU
        datasets = ["mnist", "cifar10"]
        models = ["vgg9", "resnet18"]
        timesteps = [4, 16, 64]
        seeds = [42]
        print("\n[QUICK MODE] 2 datasets / 2 models / T=[4,16,64] / 500 samples")
    else:
        # Full 36 experiments: 3 datasets × 2 models × 6 timesteps (~45 min on T4 GPU)
        datasets = ["mnist", "fashion_mnist", "cifar10"]
        models = ["vgg9", "resnet18"]
        timesteps = [4, 8, 16, 32, 64, 128]
        seeds = [42]
        print(f"\n[STANDARD MODE] {len(datasets)} datasets / {len(models)} models / "
              f"{len(timesteps)} timesteps / {len(seeds)} seeds")

    # Filter to single dataset if --dataset flag provided
    if args.dataset:
        datasets = [args.dataset]
        print(f"  → Filtered to dataset: {args.dataset}")

    total = len(datasets) * len(models) * len(timesteps) * len(seeds)
    print(f"Total experiments: {total}")
    print("="*70)

    start_time = time.time()
    completed = 0
    failed = 0

    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*70}")

        for model_name in models:
            print(f"\n  MODEL: {model_name}")

            for T in timesteps:
                for seed in seeds:
                    completed += 1
                    elapsed = time.time() - start_time
                    eta = (elapsed / completed) * (total - completed) if completed > 0 else 0
                    print(f"\n  [{completed}/{total}] {model_name}/{dataset_name}/T={T}/seed={seed} "
                          f"(elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min)")

                    result = run_single_experiment(
                        model_name, dataset_name, T, seed,
                        device=device,
                        output_dir=args.output_dir,
                    )
                    if result is None and not os.path.exists(
                        os.path.join(args.output_dir, "raw",
                                     f"{model_name}_{dataset_name}_T{T}_seed{seed}.csv")):
                        failed += 1

    total_time = time.time() - start_time
    print(f"\n\n{'='*70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Completed: {completed - failed}/{total}")
    print(f"  Failed: {failed}/{total}")
    print(f"  Results in: {args.output_dir}/raw/")
    print(f"\nNext steps:")
    print(f"  python scripts/analyze_results.py --results-dir {args.output_dir}")
    print(f"  python scripts/generate_plots.py --results-dir {args.output_dir}")


if __name__ == "__main__":
    main()
