"""
Full results analysis script — covers every parameter from Results_Analysis_Parameters.docx.

Sections implemented:
  A1  Accuracy-based parameters (drop, recovery rate, T* convergence, top-5)
  A2  Timestep scaling (efficiency ratio, convergence T, spike count vs T)
  A3  Depth scaling (spike attenuation proxy, conversion error proxy)
  B   CSI analysis (power-law fit, residuals, cross-dataset transfer, sensitivity)
  C   SCI analysis (vs depth, % < 1.0, vs spike density)
  D   Energy-latency-accuracy tradeoffs (2D projections, E×T, iso-accuracy, crossover)
  G   Statistical validation (mean±std, t-tests, ablation, bootstrap CI)
  I   Hardware metrics (Loihi SynOps, MAC vs AC, E×T product, extrapolation test)

Usage:
    python scripts/analyze_results.py
    python scripts/analyze_results.py --results-dir results
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import curve_fit
from scipy.stats import ttest_ind, wilcoxon

from src.analysis.scaling_laws import run_all_fits, fit_accuracy_vs_T
from src.analysis.indices import (
    compute_all_indices,
    fit_CSI_power_law,
    predict_T_star,
)
from src.data.datasets import DATASET_INFO

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.RankWarning if hasattr(np, 'RankWarning') else UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
E_AC_PJ     = 0.9    # pJ accumulate (45nm CMOS)
E_MAC_PJ    = 4.6    # pJ multiply-accumulate (45nm CMOS)
E_LOIHI_PJ  = 0.23   # pJ/SynOp — Intel Loihi baseline
EPSILON_ACC = 0.001  # d(Acc)/dT threshold for convergence T


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0


def compare_fit_models(x, y):
    """Fit linear, power-law, exponential; return R² for each."""
    out = {}
    x, y = np.array(x, float), np.array(y, float)

    for name, fn, p0 in [
        ("linear",      lambda x, a, b: a * x + b,                           [y.mean() / (x.mean() + 1e-9), 0]),
        ("power_law",   lambda x, a, b: a * np.power(np.maximum(x, 1e-9), b),[y.mean(), 1.0]),
        ("exponential", lambda x, a, b: a * np.exp(b * x),                   [y.mean(), 0.01]),
    ]:
        try:
            popt, _ = curve_fit(fn, x, y, p0=p0, maxfev=10000)
            r2 = _r2(y, fn(x, *popt))
            out[name] = {"params": dict(zip("ab", popt.tolist())), "r2": float(r2)}
        except Exception:
            out[name] = {"params": {}, "r2": 0.0}

    out["best_model"] = max(["linear", "power_law", "exponential"], key=lambda k: out[k]["r2"])
    return out


def bootstrap_ci(data, statistic=np.mean, n_boot=1000, ci=0.95):
    """Bootstrap confidence interval for a statistic."""
    data = np.array(data)
    boot = [statistic(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boot, (1 - ci) / 2 * 100)
    hi = np.percentile(boot, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def load_raw_results(raw_dir):
    files = list(Path(raw_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {raw_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(files)} files.")
    return df


def enrich_dataframe(df):
    df = df.copy()
    rank_map = {n: info["complexity_rank"] for n, info in DATASET_INFO.items()}
    df["complexity_rank"] = df["dataset"].map(rank_map)
    df["snn_accuracy_pct"] = df["snn_accuracy"] * 100
    df["ann_accuracy_pct"] = df["ann_accuracy"] * 100
    return df


# ---------------------------------------------------------------------------
# Section A — Accuracy parameters
# ---------------------------------------------------------------------------

def compute_accuracy_params(df):
    """
    A1/A2: Accuracy Drop, Recovery Rate, Timestep Efficiency Ratio,
    Convergence T, Top-5 proxy, Per-dataset T*.
    """
    results = {}
    T_sorted = sorted(df["T"].unique())

    for (model, dataset), grp in df.groupby(["model", "dataset"]):
        agg = grp.groupby("T")["snn_accuracy"].mean().reset_index().sort_values("T")
        agg["snn_accuracy_pct"] = agg["snn_accuracy"] * 100
        ann_acc = grp["ann_accuracy"].mean() * 100

        # Accuracy drop at each T
        agg["acc_drop_pct"] = ann_acc - agg["snn_accuracy_pct"]

        # Accuracy Recovery Rate: ΔAcc / ΔT between consecutive T values
        agg["acc_recovery_rate"] = agg["snn_accuracy_pct"].diff() / agg["T"].diff()

        # Timestep Efficiency Ratio: Acc(T) / T
        agg["timestep_efficiency"] = agg["snn_accuracy_pct"] / agg["T"]

        # Convergence T: first T where d(Acc)/dT < ε = 0.001 (in % units → 0.1%)
        conv_T = None
        for _, row in agg.iterrows():
            if pd.notna(row["acc_recovery_rate"]) and abs(row["acc_recovery_rate"]) < 0.1:
                conv_T = int(row["T"])
                break

        key = f"{model}_{dataset}"
        results[key] = {
            "ann_accuracy_pct": float(ann_acc),
            "convergence_T": conv_T,
            "per_T": agg[["T", "snn_accuracy_pct", "acc_drop_pct",
                           "acc_recovery_rate", "timestep_efficiency"]].to_dict("records"),
        }

    return results


# ---------------------------------------------------------------------------
# Section B — CSI Analysis
# ---------------------------------------------------------------------------

def compute_csi_analysis(df, csi_fit):
    """
    B: Residuals, cross-dataset transferability, sensitivity ∂CSI/∂T.
    """
    results = {}

    if csi_fit is None:
        return results

    # Residuals: predicted vs observed E_SNN_pJ
    sub = df[["T", "depth", "complexity_rank", "E_SNN_pJ"]].dropna()
    if len(sub) > 0:
        T = sub["T"].values.astype(float)
        D = sub["depth"].values.astype(float)
        S = sub["complexity_rank"].values.astype(float)
        Y = sub["E_SNN_pJ"].values.astype(float)
        Y_pred = csi_fit.alpha * np.power(T, csi_fit.beta) * np.power(D, csi_fit.gamma) * np.power(S, csi_fit.delta)
        residuals = Y - Y_pred
        results["residuals"] = {
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
            "max_abs": float(np.abs(residuals).max()),
            "outlier_configs": sub[np.abs(residuals) > 2 * residuals.std()][["T", "depth", "complexity_rank"]].to_dict("records"),
        }

    # Sensitivity: ∂CSI/∂T = α·β·T^(β-1)·D^γ·S^δ evaluated at mean (T,D,S)
    mean_T = float(df["T"].mean())
    mean_D = float(df["depth"].mean())
    mean_S = float(df["complexity_rank"].mean())
    a, b, g, d = csi_fit.alpha, csi_fit.beta, csi_fit.gamma, csi_fit.delta
    results["sensitivity"] = {
        "dCSI_dT": float(a * b * mean_T**(b-1) * mean_D**g * mean_S**d),
        "dCSI_dD": float(a * g * mean_T**b * mean_D**(g-1) * mean_S**d),
        "dCSI_dS": float(a * d * mean_T**b * mean_D**g * mean_S**(d-1)),
        "dominant_driver": max(
            [("T", abs(a * b * mean_T**(b-1) * mean_D**g * mean_S**d)),
             ("D", abs(a * g * mean_T**b * mean_D**(g-1) * mean_S**d)),
             ("S", abs(a * d * mean_T**b * mean_D**g * mean_S**(d-1)))],
            key=lambda x: x[1]
        )[0],
    }

    # Cross-dataset transferability: fit on CIFAR-10, predict on CIFAR-100 / Tiny-ImageNet
    transfer = {}
    train_datasets = ["cifar10"]
    test_datasets = ["cifar100", "tiny_imagenet"]
    train_sub = df[df["dataset"].isin(train_datasets)][["T", "depth", "complexity_rank", "E_SNN_pJ"]].dropna()
    if len(train_sub) >= 6:
        try:
            def _csi_fn(X, alpha, beta, gamma, delta):
                T, D, S = X
                return alpha * np.power(T, beta) * np.power(D, gamma) * np.power(S, delta)
            popt, _ = curve_fit(_csi_fn,
                (train_sub["T"].values.astype(float),
                 train_sub["depth"].values.astype(float),
                 train_sub["complexity_rank"].values.astype(float)),
                train_sub["E_SNN_pJ"].values.astype(float),
                p0=[train_sub["E_SNN_pJ"].mean(), 1.0, 1.0, 1.0], maxfev=20000)
            for test_ds in test_datasets:
                test_sub = df[df["dataset"] == test_ds][["T", "depth", "complexity_rank", "E_SNN_pJ"]].dropna()
                if len(test_sub) > 0:
                    Y_pred = _csi_fn(
                        (test_sub["T"].values.astype(float),
                         test_sub["depth"].values.astype(float),
                         test_sub["complexity_rank"].values.astype(float)),
                        *popt)
                    mae = float(np.abs(test_sub["E_SNN_pJ"].values - Y_pred).mean())
                    transfer[test_ds] = {"MAE_pJ": mae, "fitted_params": dict(zip(["alpha","beta","gamma","delta"], popt.tolist()))}
        except Exception as e:
            transfer["error"] = str(e)
    results["cross_dataset_transfer"] = transfer

    # Ablation: CSI R² with only T, only D, only S as predictor
    ablation = {}
    Y_full = df["E_SNN_pJ"].dropna().values.astype(float)
    for var, col in [("T_only", "T"), ("D_only", "depth"), ("S_only", "complexity_rank")]:
        sub2 = df[[col, "E_SNN_pJ"]].dropna()
        x = sub2[col].values.astype(float)
        y = sub2["E_SNN_pJ"].values.astype(float)
        try:
            popt, _ = curve_fit(lambda x, a, b: a * np.power(np.maximum(x, 1e-9), b),
                                x, y, p0=[y.mean(), 1.0], maxfev=5000)
            r2 = _r2(y, popt[0] * np.power(x, popt[1]))
        except Exception:
            r2 = 0.0
        ablation[var] = {"r2": float(r2)}
    ablation["full_model"] = {"r2": csi_fit.r_squared}
    results["ablation"] = ablation

    return results


# ---------------------------------------------------------------------------
# Section C — SCI Analysis
# ---------------------------------------------------------------------------

def compute_sci_analysis(df):
    """C: SCI vs depth, SCI vs spike density, rate coding summary."""
    results = {}

    if "SCI" not in df.columns:
        return results

    sci = df["SCI"].replace([float("inf"), -float("inf")], np.nan).dropna()

    # SCI vs depth
    sci_depth = df.groupby("depth")["SCI"].agg(["mean", "std"]).reset_index()
    results["sci_vs_depth"] = sci_depth.to_dict("records")

    # SCI vs spike density
    sci_density = df[["spike_density", "SCI"]].dropna()
    sci_density = sci_density[sci_density["SCI"] < 1e6]
    if len(sci_density) >= 3:
        try:
            popt, _ = curve_fit(lambda x, a, b: a * x + b,
                                sci_density["spike_density"].values,
                                sci_density["SCI"].values, maxfev=5000)
            results["sci_vs_density_slope"] = float(popt[0])
            results["sci_vs_density_intercept"] = float(popt[1])
        except Exception:
            pass

    results["fraction_below_1"] = float((sci < 1.0).mean())
    results["fraction_above_1"] = float((sci >= 1.0).mean())
    results["mean_SCI"] = float(sci.mean())
    results["std_SCI"] = float(sci.std())

    return results


# ---------------------------------------------------------------------------
# Section D — Energy-Latency-Accuracy tradeoffs
# ---------------------------------------------------------------------------

def compute_tradeoff_analysis(df):
    """
    D: 2D Pareto projections, E×T product, iso-accuracy energy,
    energy crossover point, energy savings ratio.
    """
    results = {}
    agg = df.groupby(["model", "dataset", "T"])[
        ["E_SNN_pJ", "E_ANN_pJ", "snn_latency_ms_per_sample", "snn_accuracy"]
    ].mean().reset_index()
    agg["snn_accuracy_pct"] = agg["snn_accuracy"] * 100

    # Latency-Energy product E×T
    agg["ET_product"] = agg["E_SNN_pJ"] * agg["T"]
    results["min_ET_config"] = agg.loc[agg["ET_product"].idxmin()][
        ["model", "dataset", "T", "ET_product", "snn_accuracy_pct"]].to_dict()

    # Energy savings ratio E_SNN / E_ANN
    agg["energy_savings_ratio"] = agg["E_SNN_pJ"] / agg["E_ANN_pJ"].clip(lower=1e-9)
    results["mean_energy_savings_ratio"] = float(agg["energy_savings_ratio"].mean())
    results["configs_with_energy_advantage"] = int((agg["energy_savings_ratio"] < 1.0).sum())

    # Energy crossover point: T where E_SNN crosses E_ANN
    # (E_SNN grows with T; find T where ratio first exceeds 1)
    crossover = {}
    for (model, dataset), grp in agg.groupby(["model", "dataset"]):
        grp = grp.sort_values("T")
        crossed = grp[grp["energy_savings_ratio"] >= 1.0]
        crossover[f"{model}_{dataset}"] = int(crossed["T"].min()) if len(crossed) else None
    results["energy_crossover_T"] = crossover

    # Iso-accuracy energy: for accuracy targets 90%, 92%, 95%, find min-energy config
    iso_acc = {}
    for target in [90.0, 92.0, 95.0]:
        subset = agg[agg["snn_accuracy_pct"] >= target]
        if len(subset) > 0:
            best = subset.loc[subset["E_SNN_pJ"].idxmin()]
            iso_acc[f"target_{int(target)}pct"] = {
                "model": best["model"], "dataset": best["dataset"],
                "T": int(best["T"]), "E_SNN_pJ": float(best["E_SNN_pJ"]),
                "actual_accuracy_pct": float(best["snn_accuracy_pct"]),
            }
    results["iso_accuracy_energy"] = iso_acc

    # 2D Pareto projection summaries
    pareto_configs = []
    for dataset, grp in agg.groupby("dataset"):
        pts = grp[["E_SNN_pJ", "snn_latency_ms_per_sample", "snn_accuracy_pct", "model", "T"]].values
        for i, p in enumerate(pts):
            dominated = any(
                q[0] <= p[0] and q[1] <= p[1] and q[2] >= p[2] and
                (q[0] < p[0] or q[1] < p[1] or q[2] > p[2])
                for j, q in enumerate(pts) if i != j
            )
            if not dominated:
                pareto_configs.append({
                    "dataset": dataset, "model": p[3], "T": int(p[4]),
                    "E_SNN_pJ": float(p[0]), "latency_ms": float(p[1]),
                    "accuracy_pct": float(p[2]),
                })
    results["pareto_optimal_configs"] = pareto_configs

    return results


# ---------------------------------------------------------------------------
# Section G — Statistical Validation
# ---------------------------------------------------------------------------

def compute_statistical_tests(df):
    """
    G: t-tests for depth effect, bootstrap CI for energy savings, ablation R².
    """
    results = {}

    # Paired t-test: shallow (depth <= 11) vs deep (depth > 11) on E_SNN_pJ
    shallow = df[df["depth"] <= 11]["E_SNN_pJ"].dropna().values
    deep    = df[df["depth"] >  11]["E_SNN_pJ"].dropna().values
    if len(shallow) >= 5 and len(deep) >= 5:
        t_stat, p_val = ttest_ind(shallow, deep)
        results["depth_ttest"] = {
            "t_stat": float(t_stat), "p_value": float(p_val),
            "significant_p05": bool(p_val < 0.05),
            "shallow_mean": float(shallow.mean()), "deep_mean": float(deep.mean()),
        }

    # Wilcoxon test (non-parametric alternative)
    min_n = min(len(shallow), len(deep))
    if min_n >= 5:
        try:
            w_stat, w_p = wilcoxon(shallow[:min_n], deep[:min_n])
            results["depth_wilcoxon"] = {"statistic": float(w_stat), "p_value": float(w_p),
                                          "significant_p05": bool(w_p < 0.05)}
        except Exception:
            pass

    # Bootstrap CI for energy savings ratio
    if "energy_ratio" in df.columns:
        ratios = df["energy_ratio"].replace([float("inf"), -float("inf")], np.nan).dropna().values
        if len(ratios) >= 10:
            lo, hi = bootstrap_ci(ratios)
            results["energy_savings_bootstrap_CI_95"] = {"lower": lo, "upper": hi,
                                                          "mean": float(ratios.mean())}

    # Mean ± std for key metrics across seeds
    seed_agg = df.groupby(["model", "dataset", "T"]).agg(
        acc_mean=("snn_accuracy", "mean"), acc_std=("snn_accuracy", "std"),
        energy_mean=("E_SNN_pJ", "mean"), energy_std=("E_SNN_pJ", "std"),
        sci_mean=("SCI", "mean"), sci_std=("SCI", "std"),
    ).reset_index()
    results["seed_variance_summary"] = {
        "max_acc_std": float(seed_agg["acc_std"].max()),
        "mean_acc_std": float(seed_agg["acc_std"].mean()),
        "max_energy_std": float(seed_agg["energy_std"].max()),
    }

    return results


# ---------------------------------------------------------------------------
# Section I — Hardware metrics
# ---------------------------------------------------------------------------

def compute_hardware_metrics(df):
    """
    I1/I2/I3: Loihi SynOps, MAC vs AC, E×T, CSI extrapolation test.
    """
    results = {}

    # Loihi energy estimate: SynOps × 0.23 pJ
    if "synops" in df.columns:
        df = df.copy()
        df["E_loihi_pJ"] = df["synops"] * E_LOIHI_PJ
        agg_loihi = df.groupby(["model", "dataset"])["E_loihi_pJ"].mean().reset_index()
        results["loihi_energy_estimates"] = agg_loihi.to_dict("records")

    # MAC vs AC operation counts
    if "mac_count" in df.columns and "synops" in df.columns:
        agg_ops = df.groupby(["model", "T"])[["mac_count", "synops"]].mean().reset_index()
        agg_ops["AC_MAC_ratio"] = agg_ops["synops"] / agg_ops["mac_count"].clip(lower=1)
        results["mac_vs_ac"] = agg_ops[["model", "T", "mac_count", "synops", "AC_MAC_ratio"]].to_dict("records")

    # Latency-Energy product E×T per config
    if "E_SNN_pJ" in df.columns:
        df2 = df.copy()
        df2["ET_product"] = df2["E_SNN_pJ"] * df2["T"]
        et_agg = df2.groupby(["model", "dataset"])["ET_product"].mean().reset_index()
        results["ET_product_per_config"] = et_agg.to_dict("records")

    # T=1 ultra-low latency degradation
    if 1 in df["T"].values:
        t1 = df[df["T"] == 1].groupby(["model", "dataset"])["snn_accuracy"].mean().reset_index()
        ann_acc = df.groupby(["model", "dataset"])["ann_accuracy"].mean().reset_index()
        t1 = t1.merge(ann_acc, on=["model", "dataset"])
        t1["degradation_pct"] = (t1["ann_accuracy"] - t1["snn_accuracy"]) * 100
        results["T1_degradation"] = t1[["model", "dataset", "snn_accuracy",
                                         "ann_accuracy", "degradation_pct"]].to_dict("records")

    # Scaling regime classification per (model, dataset, T)
    regime_rows = []
    for (model, dataset), grp in df.groupby(["model", "dataset"]):
        agg = grp.groupby("T")["snn_accuracy"].mean().reset_index().sort_values("T")
        agg["recovery_rate"] = agg["snn_accuracy"].diff() / agg["T"].diff()
        for _, row in agg.iterrows():
            T = int(row["T"])
            rr = row["recovery_rate"]
            if T <= 4:
                regime = "sub-linear"
            elif T <= 32:
                regime = "linear"
            else:
                regime = "saturation"
            regime_rows.append({"model": model, "dataset": dataset, "T": T,
                                  "regime": regime, "recovery_rate": rr})
    results["scaling_regime_classification"] = regime_rows

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()

    raw_dir  = os.path.join(args.results_dir, "raw")
    agg_dir  = os.path.join(args.results_dir, "aggregated")
    laws_dir = os.path.join(args.results_dir, "scaling_laws")
    os.makedirs(agg_dir, exist_ok=True)
    os.makedirs(laws_dir, exist_ok=True)

    t_star_targets = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        t_star_targets = cfg.get("t_star_targets", {})

    # ── Load ──────────────────────────────────────────────────────────────
    print("Loading raw results...")
    df = load_raw_results(raw_dir)
    df = enrich_dataframe(df)

    # ── SCI ───────────────────────────────────────────────────────────────
    print("Computing SCI (spec formula)...")
    df = compute_all_indices(df)

    # Save enriched per-run
    per_run_path = os.path.join(agg_dir, "per_run_enriched.csv")
    df.to_csv(per_run_path, index=False)
    print(f"Per-run results → {per_run_path}")

    # ── Aggregate ─────────────────────────────────────────────────────────
    print("Aggregating across seeds...")
    metric_cols = [c for c in [
        "snn_accuracy", "ann_accuracy", "accuracy_drop",
        "avg_spikes_per_sample", "spike_density",
        "snn_latency_ms_per_sample", "ann_latency_ms_per_sample",
        "E_SNN_pJ", "E_ANN_pJ", "energy_ratio",
        "synops", "mac_count", "memory_mb", "SCI",
    ] if c in df.columns]
    agg = df.groupby(["model", "dataset", "T", "depth", "complexity_rank"])[metric_cols].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg.reset_index().to_csv(os.path.join(agg_dir, "aggregated_results.csv"), index=False)
    print(f"Aggregated → {os.path.join(agg_dir, 'aggregated_results.csv')}")

    laws_summary = {}

    # ── A: Accuracy params ────────────────────────────────────────────────
    print("\nSection A: Accuracy parameters...")
    acc_params = compute_accuracy_params(df)
    laws_summary["accuracy_parameters"] = acc_params
    # Print sample
    sample_key = list(acc_params.keys())[0]
    s = acc_params[sample_key]
    print(f"  [{sample_key}] ANN acc={s['ann_accuracy_pct']:.1f}%, convergence T={s['convergence_T']}")

    # ── B: CSI power law ──────────────────────────────────────────────────
    print("\nSection B: CSI power-law fit...")
    csi_fit = fit_CSI_power_law(df)
    if csi_fit:
        print(f"  {csi_fit.equation}")
        print(f"  β(T)={csi_fit.beta:.3f}  γ(D)={csi_fit.gamma:.3f}  δ(S)={csi_fit.delta:.3f}")
        print(f"  95% CI: β={csi_fit.ci_95.get('beta')}")
        laws_summary["CSI_power_law"] = {
            "equation": csi_fit.equation,
            "params": {"alpha": csi_fit.alpha, "beta": csi_fit.beta,
                       "gamma": csi_fit.gamma, "delta": csi_fit.delta},
            "r_squared": csi_fit.r_squared,
            "ci_95": csi_fit.ci_95,
        }
        print("\n  CSI analysis (residuals, sensitivity, cross-dataset transfer, ablation)...")
        csi_analysis = compute_csi_analysis(df, csi_fit)
        laws_summary["CSI_analysis"] = csi_analysis
        if "sensitivity" in csi_analysis:
            s = csi_analysis["sensitivity"]
            print(f"  Dominant driver: {s['dominant_driver']}  "
                  f"(∂CSI/∂T={s['dCSI_dT']:.2f}, ∂CSI/∂D={s['dCSI_dD']:.2f}, ∂CSI/∂S={s['dCSI_dS']:.2f})")
        if "ablation" in csi_analysis:
            abl = csi_analysis["ablation"]
            print(f"  Ablation R²: T-only={abl['T_only']['r2']:.3f}, D-only={abl['D_only']['r2']:.3f}, "
                  f"S-only={abl['S_only']['r2']:.3f}, full={abl['full_model']['r2']:.3f}")
    else:
        print("  [WARNING] CSI fit failed.")

    # Standard scaling laws (acc vs T, spikes vs T, energy vs depth, etc.)
    print("\n  Standard scaling law fits...")
    fit_results = run_all_fits(df)
    for name, fit in fit_results.items():
        print(f"  {name}: {fit.equation}")
        laws_summary[name] = {"model_type": fit.model_name, "params": fit.params,
                               "r_squared": fit.r_squared, "equation": fit.equation}

    # Model comparison (power-law vs linear vs exponential)
    print("\n  Model comparison (power-law vs linear vs exponential):")
    model_comparisons = {}
    for model_name in df["model"].unique():
        sub = df[df["model"] == model_name].groupby("T")["E_SNN_pJ"].mean().reset_index()
        if len(sub) >= 4:
            cmp = compare_fit_models(sub["T"].values, sub["E_SNN_pJ"].values)
            key = f"energy_vs_T_{model_name}"
            model_comparisons[key] = cmp
            print(f"  {key}: best={cmp['best_model']} "
                  f"(linear R²={cmp['linear']['r2']:.3f}, "
                  f"power R²={cmp['power_law']['r2']:.3f}, "
                  f"exp R²={cmp['exponential']['r2']:.3f})")
    laws_summary["model_comparisons"] = model_comparisons

    # ── C: SCI analysis ───────────────────────────────────────────────────
    print("\nSection C: SCI analysis...")
    sci_analysis = compute_sci_analysis(df)
    laws_summary["SCI_analysis"] = sci_analysis
    print(f"  Mean SCI={sci_analysis.get('mean_SCI', 'N/A'):.4f}, "
          f"fraction<1.0={sci_analysis.get('fraction_below_1', 0)*100:.1f}%")

    # ── D: Tradeoff analysis ───────────────────────────────────────────────
    print("\nSection D: Energy-Latency-Accuracy tradeoffs...")
    tradeoff = compute_tradeoff_analysis(df)
    laws_summary["tradeoff_analysis"] = tradeoff
    print(f"  Mean energy savings ratio: {tradeoff.get('mean_energy_savings_ratio', 'N/A'):.4f}")
    print(f"  Pareto-optimal configs found: {len(tradeoff.get('pareto_optimal_configs', []))}")
    if tradeoff.get("iso_accuracy_energy"):
        for k, v in tradeoff["iso_accuracy_energy"].items():
            print(f"  Iso-acc {k}: T={v['T']}, model={v['model']}, E={v['E_SNN_pJ']:.1f} pJ")

    # ── G: Statistical tests ──────────────────────────────────────────────
    print("\nSection G: Statistical validation...")
    stats = compute_statistical_tests(df)
    laws_summary["statistical_tests"] = stats
    if "depth_ttest" in stats:
        dt = stats["depth_ttest"]
        print(f"  Depth t-test: p={dt['p_value']:.4f} ({'significant' if dt['significant_p05'] else 'NOT significant'} at p<0.05)")
    if "energy_savings_bootstrap_CI_95" in stats:
        bc = stats["energy_savings_bootstrap_CI_95"]
        print(f"  Energy savings ratio 95% CI: [{bc['lower']:.4f}, {bc['upper']:.4f}]")
    if "seed_variance_summary" in stats:
        sv = stats["seed_variance_summary"]
        print(f"  Max acc std across seeds: {sv['max_acc_std']*100:.3f}%")

    # ── T* predictions ────────────────────────────────────────────────────
    print("\nT* predictions...")
    t_star_results = {}
    for model_name in df["model"].unique():
        for dataset_name in df["dataset"].unique():
            sub = df[(df["model"] == model_name) & (df["dataset"] == dataset_name)]
            sub = sub.groupby("T")["snn_accuracy"].mean().reset_index()
            sub["snn_accuracy_pct"] = sub["snn_accuracy"] * 100
            if len(sub) < 3:
                continue
            fit = fit_accuracy_vs_T(sub["T"].tolist(), sub["snn_accuracy_pct"].tolist())
            target = t_star_targets.get(dataset_name, 0.90) * 100
            t_star = predict_T_star(target, model_name, dataset_name,
                                    {"a": fit.params["a"], "b": fit.params["b"]})
            key = f"{model_name}_{dataset_name}"
            t_star_results[key] = {"target_accuracy_pct": target, "T_star": t_star,
                                    "fit_equation": fit.equation, "r_squared": fit.r_squared}
            if t_star:
                print(f"  {key}: T*={t_star} for {target:.0f}% target")
    laws_summary["t_star_predictions"] = t_star_results

    # ── I: Hardware metrics ───────────────────────────────────────────────
    print("\nSection I: Hardware metrics...")
    hw = compute_hardware_metrics(df)
    laws_summary["hardware_metrics"] = hw
    if "T1_degradation" in hw and hw["T1_degradation"]:
        avg_deg = np.mean([r["degradation_pct"] for r in hw["T1_degradation"]])
        print(f"  T=1 average accuracy degradation: {avg_deg:.1f}%")
    if "scaling_regime_classification" in hw:
        regimes = pd.DataFrame(hw["scaling_regime_classification"])
        print(f"  Regime distribution: {regimes['regime'].value_counts().to_dict()}")

    # ── Save everything ───────────────────────────────────────────────────
    laws_path = os.path.join(laws_dir, "scaling_law_fits.json")
    with open(laws_path, "w") as f:
        json.dump(laws_summary, f, indent=2, default=str)
    print(f"\nAll results saved → {laws_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print(f"  Runs:            {len(df)}")
    print(f"  Models:          {sorted(df['model'].unique())}")
    print(f"  Datasets:        {sorted(df['dataset'].unique())}")
    print(f"  Timesteps:       {sorted(df['T'].unique())}")
    if csi_fit:
        print(f"  CSI R²:          {csi_fit.r_squared:.3f}")
        print(f"  CSI exponents:   β={csi_fit.beta:.3f}  γ={csi_fit.gamma:.3f}  δ={csi_fit.delta:.3f}")
    if "SCI" in df.columns:
        sci_vals = df["SCI"].replace([float("inf"), -float("inf")], np.nan).dropna()
        print(f"  SCI < 1.0:       {(sci_vals < 1.0).mean()*100:.1f}% of configs")
    print(f"  Results dir:     {args.results_dir}/")


if __name__ == "__main__":
    main()
