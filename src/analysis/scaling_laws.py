"""
Empirical scaling law fitting for ANN-to-SNN experiments.

Fits relationships between:
  - Timesteps T vs. Accuracy
  - Timesteps T vs. Spike Count
  - Network Depth vs. Energy
  - Dataset Complexity vs. Spike Density
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    model_name: str            # e.g. "log2", "linear", "power_law"
    params: Dict[str, float]   # fitted parameter values
    r_squared: float           # coefficient of determination
    x_fit: np.ndarray          # dense x values for plotting
    y_fit: np.ndarray          # predicted y values for plotting
    equation: str              # human-readable equation string


# ---------------------------------------------------------------------------
# Curve definitions
# ---------------------------------------------------------------------------

def _log2_model(T, a, b):
    return a * np.log2(np.maximum(T, 1e-8)) + b


def _linear_model(x, c, d):
    return c * x + d


def _power_law_model(x, a, b):
    return a * np.power(np.maximum(x, 1e-8), b)


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


def _fit(
    x: np.ndarray,
    y: np.ndarray,
    fn: Callable,
    p0: Optional[List] = None,
) -> Tuple[np.ndarray, float]:
    """
    Fit fn to (x, y). Returns (params, r_squared).
    Returns (zeros, 0) on failure.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(fn, x, y, p0=p0, maxfev=10000)
        y_pred = fn(x, *popt)
        r2 = _r_squared(y, y_pred)
        return popt, r2
    except Exception:
        n_params = len(p0) if p0 else 2
        return np.zeros(n_params), 0.0


# ---------------------------------------------------------------------------
# Public fitting functions
# ---------------------------------------------------------------------------

def fit_accuracy_vs_T(
    T_values: List[int],
    acc_values: List[float],
    model_label: str = "",
) -> FitResult:
    """
    Fit: Acc(T) = a * log2(T) + b

    Captures the diminishing-returns saturation observed empirically.
    """
    x = np.array(T_values, dtype=float)
    y = np.array(acc_values, dtype=float)

    popt, r2 = _fit(x, y, _log2_model, p0=[5.0, 70.0])
    a, b = popt

    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = _log2_model(x_fit, a, b)

    return FitResult(
        model_name="log2",
        params={"a": float(a), "b": float(b)},
        r_squared=r2,
        x_fit=x_fit,
        y_fit=y_fit,
        equation=f"Acc(T) = {a:.3f}·log₂(T) + {b:.3f}  [R²={r2:.3f}]",
    )


def fit_spikes_vs_T(
    T_values: List[int],
    spike_values: List[float],
    model_label: str = "",
) -> FitResult:
    """
    Fit: Spikes(T) = c * T + d  (linear)

    Spikes scale approximately linearly with timesteps.
    """
    x = np.array(T_values, dtype=float)
    y = np.array(spike_values, dtype=float)

    popt, r2 = _fit(x, y, _linear_model, p0=[y.mean() / x.mean(), 0.0])
    c, d = popt

    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = _linear_model(x_fit, c, d)

    return FitResult(
        model_name="linear",
        params={"c": float(c), "d": float(d)},
        r_squared=r2,
        x_fit=x_fit,
        y_fit=y_fit,
        equation=f"Spikes(T) = {c:.1f}·T + {d:.1f}  [R²={r2:.3f}]",
    )


def fit_energy_vs_depth(
    depth_values: List[int],
    energy_values: List[float],
    model_label: str = "",
) -> FitResult:
    """
    Fit both linear and power-law models to Energy vs. Depth.
    Returns the model with the higher R².
    """
    x = np.array(depth_values, dtype=float)
    y = np.array(energy_values, dtype=float)

    # Linear fit
    lin_popt, lin_r2 = _fit(x, y, _linear_model, p0=[y.mean(), 0.0])

    # Power-law fit
    pow_popt, pow_r2 = _fit(x, y, _power_law_model, p0=[y.mean(), 1.0])

    x_fit = np.linspace(x.min(), x.max(), 200)

    if lin_r2 >= pow_r2:
        c, d = lin_popt
        return FitResult(
            model_name="linear",
            params={"c": float(c), "d": float(d)},
            r_squared=lin_r2,
            x_fit=x_fit,
            y_fit=_linear_model(x_fit, c, d),
            equation=f"Energy(depth) = {c:.3f}·depth + {d:.3f}  [R²={lin_r2:.3f}]",
        )
    else:
        a, b = pow_popt
        return FitResult(
            model_name="power_law",
            params={"a": float(a), "b": float(b)},
            r_squared=pow_r2,
            x_fit=x_fit,
            y_fit=_power_law_model(x_fit, a, b),
            equation=f"Energy(depth) = {a:.3f}·depth^{b:.3f}  [R²={pow_r2:.3f}]",
        )


def fit_spikes_vs_complexity(
    complexity_ranks: List[int],
    spike_density_values: List[float],
) -> FitResult:
    """
    Fit Spike Density vs. Dataset Complexity Rank (1–5).
    Tries linear and power-law; returns best fit.
    """
    x = np.array(complexity_ranks, dtype=float)
    y = np.array(spike_density_values, dtype=float)

    lin_popt, lin_r2 = _fit(x, y, _linear_model, p0=[0.01, 0.0])
    pow_popt, pow_r2 = _fit(x, y, _power_law_model, p0=[0.01, 1.0])

    x_fit = np.linspace(1, 5, 100)

    if lin_r2 >= pow_r2:
        c, d = lin_popt
        return FitResult(
            model_name="linear",
            params={"c": float(c), "d": float(d)},
            r_squared=lin_r2,
            x_fit=x_fit,
            y_fit=_linear_model(x_fit, c, d),
            equation=f"SpikeDensity(rank) = {c:.4f}·rank + {d:.4f}  [R²={lin_r2:.3f}]",
        )
    else:
        a, b = pow_popt
        return FitResult(
            model_name="power_law",
            params={"a": float(a), "b": float(b)},
            r_squared=pow_r2,
            x_fit=x_fit,
            y_fit=_power_law_model(x_fit, a, b),
            equation=f"SpikeDensity(rank) = {a:.4f}·rank^{b:.4f}  [R²={pow_r2:.3f}]",
        )


# ---------------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------------

def run_all_fits(df) -> Dict[str, FitResult]:
    """
    Run all scaling law fits on a pandas DataFrame of aggregated results.

    Expected columns: model, dataset, T, snn_accuracy, avg_spikes_per_sample,
                      E_SNN_pJ, depth (network layer count), complexity_rank

    Returns dict of fit results keyed by descriptive name.
    """
    import pandas as pd

    results = {}

    for model_name in df["model"].unique():
        for dataset in df["dataset"].unique():
            subset = df[(df["model"] == model_name) & (df["dataset"] == dataset)].sort_values("T")
            if len(subset) < 3:
                continue

            key = f"{model_name}_{dataset}"

            # Accuracy vs T
            results[f"acc_vs_T_{key}"] = fit_accuracy_vs_T(
                subset["T"].tolist(),
                (subset["snn_accuracy"] * 100).tolist(),
                model_label=key,
            )

            # Spikes vs T
            results[f"spikes_vs_T_{key}"] = fit_spikes_vs_T(
                subset["T"].tolist(),
                subset["avg_spikes_per_sample"].tolist(),
                model_label=key,
            )

    # Energy vs depth (use T=32 as representative)
    if "depth" in df.columns:
        t32 = df[df["T"] == 32].groupby("model")["E_SNN_pJ"].mean().reset_index()
        depth_map = df.groupby("model")["depth"].first()
        t32["depth"] = t32["model"].map(depth_map)
        t32 = t32.dropna()
        if len(t32) >= 2:
            results["energy_vs_depth"] = fit_energy_vs_depth(
                t32["depth"].tolist(), t32["E_SNN_pJ"].tolist()
            )

    # Spike density vs dataset complexity
    if "complexity_rank" in df.columns:
        comp = df.groupby("dataset")[["spike_density", "complexity_rank"]].mean().reset_index()
        comp = comp.sort_values("complexity_rank")
        if len(comp) >= 3:
            results["spikes_vs_complexity"] = fit_spikes_vs_complexity(
                comp["complexity_rank"].tolist(),
                comp["spike_density"].tolist(),
            )

    return results
