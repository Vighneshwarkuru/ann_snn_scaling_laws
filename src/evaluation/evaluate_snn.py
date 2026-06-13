"""
Evaluate a converted SNN: accuracy, spike counts, latency, energy.

Rate-coding inference: input image is repeated T times as constant current.
Spikes are accumulated over T timesteps to form the output logit.
"""

import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import (
    SpikeLogger,
    estimate_energy,
    compute_synops,
    get_memory_usage_mb,
    get_gpu_memory_mb,
)


def evaluate_snn(
    snn_model: nn.Module,
    test_loader: DataLoader,
    T: int,
    device: torch.device,
    mac_count_ann: int = 0,
    max_samples: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate SNN accuracy and collect spike/energy metrics.

    Args:
        snn_model:      Converted SNN (SpikingJelly-based).
        test_loader:    Test DataLoader.
        T:              Number of timesteps.
        device:         Compute device.
        mac_count_ann:  MAC count of source ANN (for energy ratio).
        max_samples:    Limit evaluation to this many samples (None = all).
        verbose:        Print progress.

    Returns:
        Dict of all metrics.
    """
    snn_model = snn_model.to(device)
    snn_model.eval()

    # Reset any stateful membrane potentials from prior runs
    _reset_snn_state(snn_model)

    # Attach spike logger
    logger = SpikeLogger(snn_model)

    correct = 0
    total = 0
    total_spikes_all = 0

    start_time = time.perf_counter()
    mem_before = get_memory_usage_mb()

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            batch_size = imgs.size(0)

            if max_samples is not None and total >= max_samples:
                break

            # Reset membrane potentials and spike logger for each batch
            _reset_snn_state(snn_model)
            logger.reset()

            # Rate coding: repeat input T times, accumulate output
            output_acc = torch.zeros(batch_size, snn_model_output_dim(snn_model, imgs, device), device=device)

            for t in range(T):
                out = snn_model(imgs)
                output_acc += out

            preds = output_acc.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size
            total_spikes_all += logger.get_total_spikes()

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    mem_after = get_memory_usage_mb()
    gpu_mem_mb = get_gpu_memory_mb(device)

    logger.remove_hooks()

    accuracy = correct / max(total, 1)
    latency_ms_per_sample = elapsed_ms / max(total, 1)
    avg_spikes_per_sample = total_spikes_all / max(total, 1)
    total_neurons = logger.get_total_neurons()
    spike_density = avg_spikes_per_sample / max(total_neurons, 1)
    per_layer_spikes = logger.get_per_layer_spikes()

    # Energy: use per-sample avg spikes
    energy = estimate_energy(
        spike_count=int(avg_spikes_per_sample),
        mac_count_ann=mac_count_ann,
    )
    synops = compute_synops(int(avg_spikes_per_sample))

    if verbose:
        print(
            f"  SNN T={T} | Acc: {accuracy*100:.2f}% | "
            f"Spikes/sample: {avg_spikes_per_sample:.1f} | "
            f"Density: {spike_density:.4f} | "
            f"Latency: {latency_ms_per_sample:.4f} ms/sample | "
            f"E_SNN: {energy['E_SNN_pJ']:.2f} pJ | "
            f"EnergyRatio: {energy['energy_ratio']:.4f}"
        )

    return {
        # Accuracy
        "snn_accuracy": accuracy,
        "T": T,
        # Spikes
        "total_spikes": total_spikes_all,
        "avg_spikes_per_sample": avg_spikes_per_sample,
        "total_neurons": total_neurons,
        "spike_density": spike_density,
        "per_layer_spikes": per_layer_spikes,
        # Latency
        "snn_latency_ms_per_sample": latency_ms_per_sample,
        "total_inference_ms": elapsed_ms,
        # Energy
        "E_SNN_pJ": energy["E_SNN_pJ"],
        "E_ANN_pJ": energy["E_ANN_pJ"],
        "energy_ratio": energy["energy_ratio"],
        # SynOps
        "synops": synops,
        # Memory
        "memory_mb": mem_after - mem_before,
        "gpu_memory_mb": gpu_mem_mb,
    }


def snn_model_output_dim(model: nn.Module, sample_imgs: torch.Tensor, device: torch.device) -> int:
    """
    Infer the output dimension of the SNN with a single dry-run forward pass.
    """
    model.eval()
    _reset_snn_state(model)
    with torch.no_grad():
        out = model(sample_imgs[:1])
    if isinstance(out, tuple):
        out = out[0]
    return out.shape[-1]


def _reset_snn_state(model: nn.Module) -> None:
    """
    Reset membrane potentials of all SpikingJelly stateful neurons.
    Falls back gracefully if SpikingJelly is not available.
    """
    try:
        from spikingjelly.activation_based import functional
        functional.reset_net(model)
    except ImportError:
        pass
    except Exception:
        pass
