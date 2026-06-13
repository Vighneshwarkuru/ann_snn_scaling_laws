"""
Evaluate a trained ANN model: accuracy, MAC count, latency, energy.
"""

import time
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import compute_mac_count, estimate_energy, get_memory_usage_mb, get_gpu_memory_mb


def evaluate_ann(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    input_size: tuple,       # (C, H, W)
    verbose: bool = True,
) -> Dict:
    """
    Evaluate ANN accuracy, latency, and energy proxy.

    Returns dict with:
        ann_accuracy, ann_latency_ms_per_sample, mac_count,
        E_ANN_pJ (per sample), memory_mb
    """
    model = model.to(device)
    model.eval()

    # --- Count MACs (once) ---
    mac_count = compute_mac_count(model, input_size, device)

    # --- Accuracy + latency ---
    correct = 0
    total = 0
    start_time = time.perf_counter()
    mem_before = get_memory_usage_mb()

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    mem_after = get_memory_usage_mb()
    gpu_mem_mb = get_gpu_memory_mb(device)

    accuracy = correct / total
    latency_ms_per_sample = elapsed_ms / total if total > 0 else 0.0

    # Energy per sample (MAC count is already per-sample from ptflops)
    energy = estimate_energy(spike_count=0, mac_count_ann=mac_count)

    if verbose:
        print(
            f"  ANN | Acc: {accuracy*100:.2f}% | "
            f"Latency: {latency_ms_per_sample:.4f} ms/sample | "
            f"MACs: {mac_count:,} | "
            f"E_ANN: {energy['E_ANN_pJ']:.2f} pJ/sample"
        )

    return {
        "ann_accuracy": accuracy,
        "ann_latency_ms_per_sample": latency_ms_per_sample,
        "mac_count": mac_count,
        "E_ANN_pJ": energy["E_ANN_pJ"],
        "memory_mb": mem_after - mem_before,
        "gpu_memory_mb": gpu_mem_mb,
    }
