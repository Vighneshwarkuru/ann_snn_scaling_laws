"""
Generate all 16 plots from aggregated results.

Run after analyze_results.py.

Usage:
    python scripts/generate_plots.py
    python scripts/generate_plots.py --results-dir results
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.visualization.plots import generate_all_plots
from src.analysis.scaling_laws import run_all_fits
from src.analysis.indices import fit_CSI_power_law


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    agg_dir = os.path.join(args.results_dir, "aggregated")
    per_run_path = os.path.join(agg_dir, "per_run_enriched.csv")

    if not os.path.exists(per_run_path):
        print(f"ERROR: {per_run_path} not found. Run analyze_results.py first.")
        sys.exit(1)

    print(f"Loading results from {per_run_path}")
    df = pd.read_csv(per_run_path)

    print("Fitting scaling laws...")
    fit_results = run_all_fits(df)

    print("Fitting CSI power law...")
    csi_fit = fit_CSI_power_law(df)
    if csi_fit:
        print(f"  {csi_fit.equation}")

    # Load pre-computed analysis outputs
    t_star_data, ablation_data = None, None
    laws_path = os.path.join(args.results_dir, "scaling_laws", "scaling_law_fits.json")
    if os.path.exists(laws_path):
        with open(laws_path) as f:
            laws = json.load(f)
        t_star_data    = laws.get("t_star_predictions", {})
        ablation_data  = laws.get("CSI_analysis", {}).get("ablation", {})

    generate_all_plots(df, output_dir=args.results_dir,
                       fit_results=fit_results, csi_fit=csi_fit,
                       t_star_data=t_star_data, ablation_data=ablation_data)


if __name__ == "__main__":
    main()
