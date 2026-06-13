"""
Generate realistic synthetic results for all 225 experiment combinations.

This lets you see all 7 plots and the full analysis pipeline immediately,
before the real sweep finishes. The synthetic data follows the scaling trends
the project is designed to discover:

  - Accuracy saturates logarithmically with T
  - Spike count grows linearly with T
  - Energy grows linearly with depth
  - Harder datasets → denser spike activity
  - CSI increases (diminishing returns) at high T

Run:
    python scripts/generate_demo_data.py
    python scripts/analyze_results.py
    python scripts/generate_plots.py
"""

import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

# ---------------------------------------------------------------------------
# Ground-truth parameters (realistic, literature-inspired)
# ---------------------------------------------------------------------------

# ANN baseline accuracies per (model, dataset)
ANN_ACC = {
    ("vgg9",    "mnist"):          0.993,
    ("vgg9",    "fashion_mnist"):  0.928,
    ("vgg9",    "cifar10"):        0.895,
    ("vgg9",    "cifar100"):       0.641,
    ("vgg9",    "tiny_imagenet"):  0.498,
    ("vgg11",   "mnist"):          0.994,
    ("vgg11",   "fashion_mnist"):  0.934,
    ("vgg11",   "cifar10"):        0.912,
    ("vgg11",   "cifar100"):       0.692,
    ("vgg11",   "tiny_imagenet"):  0.531,
    ("vgg16",   "mnist"):          0.995,
    ("vgg16",   "fashion_mnist"):  0.939,
    ("vgg16",   "cifar10"):        0.931,
    ("vgg16",   "cifar100"):       0.721,
    ("vgg16",   "tiny_imagenet"):  0.561,
    ("resnet18","mnist"):          0.995,
    ("resnet18","fashion_mnist"):  0.941,
    ("resnet18","cifar10"):        0.931,
    ("resnet18","cifar100"):       0.731,
    ("resnet18","tiny_imagenet"):  0.612,
    ("resnet34","mnist"):          0.996,
    ("resnet34","fashion_mnist"):  0.945,
    ("resnet34","cifar10"):        0.941,
    ("resnet34","cifar100"):       0.751,
    ("resnet34","tiny_imagenet"):  0.641,
}

SATURATION = {
    ("vgg9",    "mnist"):          0.97,
    ("vgg9",    "fashion_mnist"):  0.96,
    ("vgg9",    "cifar10"):        0.95,
    ("vgg9",    "cifar100"):       0.92,
    ("vgg9",    "tiny_imagenet"):  0.89,
    ("vgg11",   "mnist"):          0.98,
    ("vgg11",   "fashion_mnist"):  0.97,
    ("vgg11",   "cifar10"):        0.96,
    ("vgg11",   "cifar100"):       0.93,
    ("vgg11",   "tiny_imagenet"):  0.90,
    ("vgg16",   "mnist"):          0.98,
    ("vgg16",   "fashion_mnist"):  0.97,
    ("vgg16",   "cifar10"):        0.96,
    ("vgg16",   "cifar100"):       0.93,
    ("vgg16",   "tiny_imagenet"):  0.91,
    ("resnet18","mnist"):          0.98,
    ("resnet18","fashion_mnist"):  0.97,
    ("resnet18","cifar10"):        0.97,
    ("resnet18","cifar100"):       0.94,
    ("resnet18","tiny_imagenet"):  0.92,
    ("resnet34","mnist"):          0.99,
    ("resnet34","fashion_mnist"):  0.97,
    ("resnet34","cifar10"):        0.97,
    ("resnet34","cifar100"):       0.95,
    ("resnet34","tiny_imagenet"):  0.93,
}

BASE_SPIKES = {
    ("vgg9",    "mnist"):          2800,
    ("vgg9",    "fashion_mnist"):  4100,
    ("vgg9",    "cifar10"):        6800,
    ("vgg9",    "cifar100"):       9500,
    ("vgg9",    "tiny_imagenet"): 14000,
    ("vgg11",   "mnist"):          3200,
    ("vgg11",   "fashion_mnist"):  4800,
    ("vgg11",   "cifar10"):        7500,
    ("vgg11",   "cifar100"):      11000,
    ("vgg11",   "tiny_imagenet"): 16000,
    ("vgg16",   "mnist"):          4500,
    ("vgg16",   "fashion_mnist"):  6800,
    ("vgg16",   "cifar10"):       10500,
    ("vgg16",   "cifar100"):      15500,
    ("vgg16",   "tiny_imagenet"): 23000,
    ("resnet18","mnist"):          4100,
    ("resnet18","fashion_mnist"):  6100,
    ("resnet18","cifar10"):        9800,
    ("resnet18","cifar100"):      14500,
    ("resnet18","tiny_imagenet"): 21000,
    ("resnet34","mnist"):          6200,
    ("resnet34","fashion_mnist"):  9100,
    ("resnet34","cifar10"):       14500,
    ("resnet34","cifar100"):      21000,
    ("resnet34","tiny_imagenet"): 31000,
}

TOTAL_NEURONS = {
    "vgg9":     98304,
    "vgg11":    131072,
    "vgg16":    196608,
    "resnet18": 262144,
    "resnet34": 393216,
}

MODEL_DEPTH = {"vgg9": 9, "vgg11": 11, "vgg16": 16, "resnet18": 18, "resnet34": 34}

MAC_COUNT = {
    "vgg9":    120_000_000,
    "vgg11":   160_000_000,
    "vgg16":   310_000_000,
    "resnet18":560_000_000,
    "resnet34":730_000_000,
}

BASE_LATENCY = {
    "vgg9":    0.95,
    "vgg11":   1.20,
    "vgg16":   1.85,
    "resnet18":2.10,
    "resnet34":2.90,
}

E_AC_PJ  = 0.9
E_MAC_PJ = 4.6

DATASETS   = ["mnist", "fashion_mnist", "cifar10", "cifar100", "tiny_imagenet"]
MODELS     = ["vgg9", "vgg11", "vgg16", "resnet18", "resnet34"]
TIMESTEPS  = [1, 2, 4, 8, 16, 32, 64, 128]
SEEDS      = [42, 123, 7]

FIELDNAMES = [
    "model", "dataset", "T", "seed", "depth",
    "ann_accuracy", "snn_accuracy", "accuracy_drop",
    "total_spikes", "avg_spikes_per_sample", "total_neurons", "spike_density",
    "snn_latency_ms_per_sample", "ann_latency_ms_per_sample",
    "E_SNN_pJ", "E_ANN_pJ", "energy_ratio",
    "synops", "mac_count",
    "memory_mb", "gpu_memory_mb",
]


def snn_accuracy(model, dataset, T, seed):
    """Logarithmic saturation + small noise."""
    rng = np.random.default_rng(seed + hash(model + dataset) % 10000)
    ann_acc = ANN_ACC[(model, dataset)]
    sat = SATURATION[(model, dataset)]
    # At T=64, reach sat * ann_acc. Fit: acc(T) = ann_acc * sat * log2(T)/log2(64)
    frac = sat * math.log2(T) / math.log2(64)
    frac = min(frac, sat)
    noise = rng.normal(0, 0.004)
    return float(np.clip(ann_acc * frac + noise, 0.05, ann_acc * 1.01))


def avg_spikes(model, dataset, T, seed):
    """Linear scaling with T + small multiplicative noise."""
    rng = np.random.default_rng(seed + hash(model + dataset + "spikes") % 10000)
    base = BASE_SPIKES[(model, dataset)]
    noise = rng.normal(1.0, 0.03)
    return float(base * (T / 4) * noise)


def latency(model, T, seed):
    """Roughly linear in T, with small noise."""
    rng = np.random.default_rng(seed + hash(model + "lat") % 10000)
    base = BASE_LATENCY[model]
    noise = rng.normal(1.0, 0.02)
    return float(base * (T / 4) * noise)


def make_row(model, dataset, T, seed):
    ann_acc  = ANN_ACC[(model, dataset)]
    snn_acc  = snn_accuracy(model, dataset, T, seed)
    spikes   = avg_spikes(model, dataset, T, seed)
    neurons  = TOTAL_NEURONS[model]
    lat_snn  = latency(model, T, seed)
    lat_ann  = BASE_LATENCY[model] / 4 * 0.5  # ANN is much faster
    macs     = MAC_COUNT[model]

    e_snn = spikes * E_AC_PJ
    e_ann = macs   * E_MAC_PJ
    ratio = e_snn / e_ann if e_ann > 0 else 0.0

    rng = np.random.default_rng(seed + T)
    mem = rng.uniform(50, 200)

    return {
        "model":                      model,
        "dataset":                    dataset,
        "T":                          T,
        "seed":                       seed,
        "depth":                      MODEL_DEPTH[model],
        "ann_accuracy":               ann_acc,
        "snn_accuracy":               snn_acc,
        "accuracy_drop":              ann_acc - snn_acc,
        "total_spikes":               int(spikes * 10000),
        "avg_spikes_per_sample":      spikes,
        "total_neurons":              neurons,
        "spike_density":              spikes / neurons,
        "snn_latency_ms_per_sample":  lat_snn,
        "ann_latency_ms_per_sample":  lat_ann,
        "E_SNN_pJ":                   e_snn,
        "E_ANN_pJ":                   e_ann,
        "energy_ratio":               ratio,
        "synops":                     int(spikes),
        "mac_count":                  macs,
        "memory_mb":                  mem,
        "gpu_memory_mb":              0.0,
    }


def main():
    raw_dir = os.path.join("results", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    total = len(DATASETS) * len(MODELS) * len(TIMESTEPS) * len(SEEDS)
    written = 0
    skipped = 0

    print(f"Generating {total} synthetic experiment results...")

    for dataset in DATASETS:
        for model in MODELS:
            for T in TIMESTEPS:
                for seed in SEEDS:
                    path = os.path.join(raw_dir, f"{model}_{dataset}_T{T}_seed{seed}.csv")
                    if os.path.exists(path):
                        skipped += 1
                        continue

                    row = make_row(model, dataset, T, seed)
                    with open(path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
                        writer.writeheader()
                        writer.writerow(row)
                    written += 1

    print(f"Done. Written: {written}, Skipped (already exist): {skipped}")
    print(f"\nNow run:")
    print(f"  python scripts/analyze_results.py")
    print(f"  python scripts/generate_plots.py")


if __name__ == "__main__":
    main()
