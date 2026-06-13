"""
ANN-to-SNN conversion using SpikingJelly's ann2snn pipeline.

Uses rate-coding with threshold (weight) normalization.
The converted SNN's neurons fire based on a threshold derived from the
maximum activation statistics of the source ANN (percentile normalization).
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ANNtoSNNConverter:
    """
    Converts a trained ANN into a rate-coded SNN using SpikingJelly.

    Workflow:
        1. Run calibration forward passes to collect activation statistics.
        2. Compute per-layer firing thresholds via percentile normalization.
        3. Replace ReLU activations with IF neurons.
        4. Return the converted SNN module.

    Args:
        model:            Trained ANN (nn.Module with ReLU + BatchNorm).
        T:                Number of timesteps for rate coding.
        norm_percentile:  Percentile used to clip activation threshold (default 99.95%).
        device:           Compute device.
    """

    def __init__(
        self,
        model: nn.Module,
        T: int,
        norm_percentile: float = 0.9995,
        device: Optional[torch.device] = None,
    ):
        self.ann = model
        self.T = T
        self.norm_percentile = norm_percentile
        self.device = device or torch.device("cpu")
        self.snn: Optional[nn.Module] = None

    def convert(self, calibration_loader: DataLoader, n_calib_batches: int = 8) -> nn.Module:
        """
        Convert ANN to SNN.

        Always runs on CPU — torch.fx symbolic tracing (used internally by
        SpikingJelly's Converter) requires CPU. The returned SNN can then be
        moved to any device for inference.

        Args:
            calibration_loader: DataLoader for calibration samples.
            n_calib_batches:    How many batches to use for activation statistics.

        Returns:
            Converted SNN module (nn.Module), still on CPU.
        """
        try:
            from spikingjelly.activation_based import ann2snn
        except ImportError:
            raise ImportError(
                "SpikingJelly is required for ANN-to-SNN conversion. "
                "Install with: pip install spikingjelly"
            )

        # Conversion must happen on CPU (torch.fx tracing limitation)
        ann = copy.deepcopy(self.ann).cpu()
        ann.eval()

        converter = ann2snn.Converter(mode="max", dataloader=calibration_loader)
        snn = converter(ann)
        # Keep on CPU; caller moves to target device
        snn = snn.cpu()

        self.snn = snn
        return snn

    def set_timesteps(self, T: int) -> None:
        """Update timestep setting on the converted SNN."""
        self.T = T
        # SpikingJelly neurons are stateless by default; T is managed by
        # the inference loop, so nothing model-internal needs changing here.

    def get_snn(self) -> nn.Module:
        if self.snn is None:
            raise RuntimeError("Call .convert() before .get_snn()")
        return self.snn
