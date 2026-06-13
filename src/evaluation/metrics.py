"""
Metric computation utilities.

- SpikeLogger: attaches forward hooks to SNN layers to count spikes per layer
- compute_mac_count: count MACs in an ANN using ptflops
- estimate_energy: convert spike/MAC counts to energy estimates
- compute_synops: compute SNN synaptic operations
"""

from __future__ import annotations

import psutil
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Energy constants (Horowitz 2014, 45nm CMOS)
# ---------------------------------------------------------------------------
E_AC_PJ  = 0.9    # pJ per accumulate operation (SNN spike-triggered)
E_MAC_PJ = 4.6    # pJ per multiply-accumulate operation (ANN)


# ---------------------------------------------------------------------------
# Spike Logger
# ---------------------------------------------------------------------------

class SpikeLogger:
    """
    Attaches PyTorch forward hooks to spiking layers in an SNN and counts
    the number of spikes (non-zero activations) per layer per forward pass.

    Usage:
        logger = SpikeLogger(snn_model)
        with torch.no_grad():
            for t in range(T):
                out = snn_model(x_t)
        stats = logger.get_stats(total_neurons=N)
        logger.remove_hooks()
    """

    def __init__(self, model: nn.Module):
        self.layer_spikes: Dict[str, int] = {}
        self.layer_neurons: Dict[str, int] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._register_hooks(model)

    def _register_hooks(self, model: nn.Module) -> None:
        """
        Register hooks on all spiking layers.
        Works for SpikingJelly IFNode / LIFNode, or any layer whose output
        is a binary {0, 1} tensor (we detect via output dtype / values).
        """
        # Try SpikingJelly layer types first; fall back to generic detection
        try:
            from spikingjelly.activation_based import neuron as sj_neuron
            spike_types = (sj_neuron.IFNode, sj_neuron.LIFNode, sj_neuron.ParametricLIFNode)
        except ImportError:
            spike_types = ()

        for name, module in model.named_modules():
            if spike_types and isinstance(module, spike_types):
                self._attach_hook(name, module)
            # Also catch any module whose name suggests it's a spike layer
            elif "if_node" in name.lower() or "lif" in name.lower() or "spike" in name.lower():
                if not any(name == k for k in self.layer_spikes):
                    self._attach_hook(name, module)

    def _attach_hook(self, name: str, module: nn.Module) -> None:
        layer_name = name if name else "root"
        self.layer_spikes[layer_name] = 0
        self.layer_neurons[layer_name] = 0

        def hook(mod, inp, out):
            # out may be (spikes, membrane) tuple in some SJ versions
            if isinstance(out, tuple):
                out = out[0]
            spikes = out.detach()
            self.layer_spikes[layer_name] += int(spikes.sum().item())
            # Track number of neurons (spatial * feature dims, not batch)
            if self.layer_neurons[layer_name] == 0:
                self.layer_neurons[layer_name] = spikes[0].numel()

        h = module.register_forward_hook(hook)
        self._hooks.append(h)

    def reset(self) -> None:
        """Reset spike counters (call before each new sample batch)."""
        for k in self.layer_spikes:
            self.layer_spikes[k] = 0

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def get_total_spikes(self) -> int:
        return sum(self.layer_spikes.values())

    def get_per_layer_spikes(self) -> Dict[str, int]:
        return dict(self.layer_spikes)

    def get_total_neurons(self) -> int:
        return sum(self.layer_neurons.values())

    def get_spike_density(self, n_samples: int = 1) -> float:
        """
        Spike density = total_spikes / (total_neurons * n_samples).
        """
        total_neurons = self.get_total_neurons()
        if total_neurons == 0 or n_samples == 0:
            return 0.0
        return self.get_total_spikes() / (total_neurons * n_samples)

    def get_stats(self, n_samples: int = 1) -> Dict:
        total = self.get_total_spikes()
        total_neurons = self.get_total_neurons()
        return {
            "total_spikes": total,
            "avg_spikes_per_sample": total / max(n_samples, 1),
            "total_neurons": total_neurons,
            "spike_density": self.get_spike_density(n_samples),
            "per_layer_spikes": self.get_per_layer_spikes(),
        }


# ---------------------------------------------------------------------------
# MAC counting
# ---------------------------------------------------------------------------

def compute_mac_count(model: nn.Module, input_size: Tuple, device: torch.device) -> int:
    """
    Count MACs for an ANN model using ptflops.
    Returns 0 if ptflops is not installed.

    Args:
        model:       ANN nn.Module
        input_size:  (C, H, W) — without batch dimension
        device:      torch device

    Returns:
        Number of multiply-accumulate operations (integer).
    """
    try:
        from ptflops import get_model_complexity_info
        model = model.to(device)
        macs, _ = get_model_complexity_info(
            model,
            input_size,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        return int(macs) if macs is not None else 0
    except ImportError:
        print("  [WARNING] ptflops not installed. MAC count will be 0.")
        return 0
    except Exception as e:
        print(f"  [WARNING] MAC count failed: {e}")
        return 0


# ---------------------------------------------------------------------------
# Energy estimation
# ---------------------------------------------------------------------------

def estimate_energy(
    spike_count: int,
    mac_count_ann: int,
) -> Dict[str, float]:
    """
    Estimate energy for SNN and equivalent ANN.

    Uses Horowitz 2014 constants:
        E_AC  = 0.9 pJ  (SNN spike-triggered accumulate)
        E_MAC = 4.6 pJ  (ANN multiply-accumulate)

    Returns dict with keys: E_SNN_pJ, E_ANN_pJ, energy_ratio
    """
    e_snn = spike_count * E_AC_PJ
    e_ann = mac_count_ann * E_MAC_PJ
    ratio = e_snn / e_ann if e_ann > 0 else float("inf")
    return {
        "E_SNN_pJ": e_snn,
        "E_ANN_pJ": e_ann,
        "energy_ratio": ratio,
    }


# ---------------------------------------------------------------------------
# Synaptic operations
# ---------------------------------------------------------------------------

def compute_synops(spike_count: int, fan_out_per_neuron: Optional[int] = None) -> int:
    """
    Compute synaptic operations (ACs) for an SNN.

    If fan_out_per_neuron is provided, SynOps = spike_count * fan_out.
    Otherwise, SynOps = spike_count (conservative lower bound).
    """
    if fan_out_per_neuron is not None:
        return spike_count * fan_out_per_neuron
    return spike_count


# ---------------------------------------------------------------------------
# Memory usage
# ---------------------------------------------------------------------------

def get_memory_usage_mb() -> float:
    """Returns current process RSS memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def get_gpu_memory_mb(device: torch.device) -> float:
    """Returns current GPU memory allocated in MB (0 if not CUDA)."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / (1024 ** 2)
    return 0.0
