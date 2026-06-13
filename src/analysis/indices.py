"""
Complexity and efficiency indices for ANN-to-SNN scaling analysis.

From spec (ANN_SNN_Architecture.drawio + Results_Analysis_Parameters.docx):

Complexity Scaling Index (CSI):
    CSI(T, D, S) = α · T^β · D^γ · S^δ
    Power-law fitted via nonlinear least squares (scipy).
    T = timesteps, D = depth (# layers), S = dataset complexity rank (1–5)
    Exponents β, γ, δ are the primary scientific contribution.

Spike Complexity Index (SCI) — per layer l:
    SCI_l = (r_l × FLOPs_l × E_AC) / (FLOPs_ANN_l × E_MAC)
    r_l     = spike rate at layer l (spikes / (neurons_l × T))
    FLOPs_l = synaptic operations at layer l
    E_AC    = 0.9 pJ (45nm CMOS accumulate)
    E_MAC   = 4.6 pJ (45nm CMOS multiply-accumulate)
    SCI_l < 1.0 → SNN layer is more energy-efficient than ANN equivalent

Network-level SCI = mean(SCI_l) across all layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


E_AC_PJ  = 0.9    # pJ — accumulate op, 45nm CMOS (Horowitz 2014)
E_MAC_PJ = 4.6    # pJ — multiply-accumulate op, 45nm CMOS


# ---------------------------------------------------------------------------
# SCI — per layer and network level
# ---------------------------------------------------------------------------

def compute_SCI_layer(
    spike_rate_l: float,       # spikes / (neurons_l * T)
    flops_l: float,            # synaptic ops at layer l (ACs)
    flops_ann_l: float,        # MACs at layer l in ANN
) -> float:
    """
    SCI_l = (r_l × FLOPs_l × E_AC) / (FLOPs_ANN_l × E_MAC)

    SCI_l < 1.0 means the SNN layer uses less energy than the ANN layer.
    """
    denom = flops_ann_l * E_MAC_PJ
    if denom <= 0:
        return float("inf")
    return (spike_rate_l * flops_l * E_AC_PJ) / denom


def compute_SCI_network(
    per_layer_spike_rates: List[float],
    per_layer_flops_snn: List[float],
    per_layer_flops_ann: List[float],
) -> Dict:
    """
    Compute SCI for all layers and aggregate.

    Returns:
        per_layer_SCI: list of SCI_l values
        mean_SCI:      network-level SCI (mean across layers)
        fraction_below_1: fraction of layers where SNN is more efficient
    """
    sci_vals = [
        compute_SCI_layer(r, fs, fa)
        for r, fs, fa in zip(per_layer_spike_rates, per_layer_flops_snn, per_layer_flops_ann)
    ]
    finite = [v for v in sci_vals if np.isfinite(v)]
    return {
        "per_layer_SCI": sci_vals,
        "mean_SCI": float(np.mean(finite)) if finite else float("nan"),
        "fraction_below_1": sum(1 for v in finite if v < 1.0) / max(len(finite), 1),
    }


def compute_SCI_from_totals(
    avg_spikes_per_sample: float,
    total_neurons: int,
    T: int,
    mac_count_ann: int,
) -> float:
    """
    Simplified network-level SCI when per-layer breakdown is unavailable.

    Treats entire network as one layer:
        r = avg_spikes / (total_neurons * T)
        FLOPs_SNN ≈ avg_spikes (each spike triggers one AC)
        FLOPs_ANN = mac_count_ann
    """
    if total_neurons <= 0 or T <= 0 or mac_count_ann <= 0:
        return float("inf")
    r = avg_spikes_per_sample / (total_neurons * T)
    flops_snn = avg_spikes_per_sample
    return compute_SCI_layer(r, flops_snn, mac_count_ann)


# ---------------------------------------------------------------------------
# CSI — power-law fit
# ---------------------------------------------------------------------------

@dataclass
class CSIFitResult:
    alpha: float
    beta: float        # T exponent
    gamma: float       # D (depth) exponent
    delta: float       # S (dataset complexity) exponent
    r_squared: float
    equation: str
    ci_95: Dict[str, Tuple[float, float]] = field(default_factory=dict)


def _csi_power_law(X, alpha, beta, gamma, delta):
    """CSI(T, D, S) = alpha * T^beta * D^gamma * S^delta"""
    T, D, S = X
    return alpha * np.power(T, beta) * np.power(D, gamma) * np.power(S, delta)


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0


def fit_CSI_power_law(df: pd.DataFrame) -> Optional[CSIFitResult]:
    """
    Fit CSI(T, D, S) = α·T^β·D^γ·S^δ to the results DataFrame.

    Uses E_SNN_pJ as the complexity proxy (proportional to energy × T, which
    captures both energy and latency components in a single scalar).

    Requires columns: T, depth, complexity_rank, E_SNN_pJ
    """
    required = ["T", "depth", "complexity_rank", "E_SNN_pJ"]
    if not all(c in df.columns for c in required):
        return None

    sub = df[required].dropna()
    sub = sub[(sub["T"] > 0) & (sub["depth"] > 0) & (sub["complexity_rank"] > 0)]

    if len(sub) < 6:
        return None

    T = sub["T"].values.astype(float)
    D = sub["depth"].values.astype(float)
    S = sub["complexity_rank"].values.astype(float)
    Y = sub["E_SNN_pJ"].values.astype(float)

    try:
        popt, pcov = curve_fit(
            _csi_power_law, (T, D, S), Y,
            p0=[Y.mean(), 1.0, 1.0, 1.0],
            maxfev=20000,
            bounds=([0, -5, -5, -5], [1e12, 5, 5, 5]),
        )
        alpha, beta, gamma, delta = popt
        y_pred = _csi_power_law((T, D, S), *popt)
        r2 = _r_squared(Y, y_pred)

        # 95% CI from covariance matrix
        perr = np.sqrt(np.diag(pcov))
        ci = {
            "alpha": (float(alpha - 1.96*perr[0]), float(alpha + 1.96*perr[0])),
            "beta":  (float(beta  - 1.96*perr[1]), float(beta  + 1.96*perr[1])),
            "gamma": (float(gamma - 1.96*perr[2]), float(gamma + 1.96*perr[2])),
            "delta": (float(delta - 1.96*perr[3]), float(delta + 1.96*perr[3])),
        }

        eq = (f"CSI(T,D,S) = {alpha:.4f}·T^{beta:.3f}·D^{gamma:.3f}·S^{delta:.3f}"
              f"  [R²={r2:.3f}]")

        return CSIFitResult(
            alpha=float(alpha), beta=float(beta),
            gamma=float(gamma), delta=float(delta),
            r_squared=r2, equation=eq, ci_95=ci,
        )
    except Exception as e:
        print(f"  [WARNING] CSI power-law fit failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Optimal timestep T* predictor
# ---------------------------------------------------------------------------

def predict_T_star(
    target_accuracy: float,
    model_name: str,
    dataset_name: str,
    fit_params: Dict,  # from fit_accuracy_vs_T
) -> Optional[int]:
    """
    Predict the minimum T required to achieve target_accuracy.

    Given Acc(T) = a·log₂(T) + b  →  T* = 2^((target - b) / a)

    Returns the smallest T from the standard set {1,2,4,8,16,32,64,128}
    that meets or exceeds the target, or None if unachievable.
    """
    a = fit_params.get("a", None)
    b = fit_params.get("b", None)
    if a is None or b is None or a <= 0:
        return None

    T_star_continuous = 2 ** ((target_accuracy - b) / a)
    T_candidates = [1, 2, 4, 8, 16, 32, 64, 128]
    for T in T_candidates:
        if T >= T_star_continuous:
            return T
    return None  # target not achievable within sweep range


# ---------------------------------------------------------------------------
# Batch computation over a results DataFrame
# ---------------------------------------------------------------------------

def compute_all_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add SCI (network-level, from totals) column to results DataFrame.
    CSI is now fit as a power law in analyze_results.py, not per-row.

    Expected columns: avg_spikes_per_sample, total_neurons, T, mac_count
    """
    df = df.copy()

    df["SCI"] = df.apply(
        lambda r: compute_SCI_from_totals(
            r["avg_spikes_per_sample"],
            int(r["total_neurons"]),
            int(r["T"]),
            int(r["mac_count"]) if r["mac_count"] > 0 else 1,
        ),
        axis=1,
    )

    return df


def summarize_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate SCI across seeds: mean ± std per (model, dataset, T)."""
    agg = df.groupby(["model", "dataset", "T"]).agg(
        SCI_mean=("SCI", "mean"),
        SCI_std=("SCI", "std"),
    ).reset_index()
    return agg
