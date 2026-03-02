"""
Spearman rank correlation analysis between NR metrics and detection performance.

Computes correlation between image quality (NIQE, BRISQUE, LOE) and
detection accuracy (mAP) across scenarios.

Note: With only 4 data points (S1-S4), this is reported as an
observational trend, NOT a statistically significant finding.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing import Dict, List, Optional


def compute_spearman_correlation(
    detection_results: Dict[str, dict],
    nr_results: Dict[str, dict],
    output_dir: str,
    metrics_pairs: Optional[List[tuple]] = None,
) -> dict:
    """Compute Spearman rank correlation between NR metrics and mAP.

    Args:
        detection_results: Dict[scenario_name → {"mAP_50": float, "mAP_50_95": float, ...}]
        nr_results: Dict[scenario_name → {"niqe_mean": float, "brisque_mean": float, ...}]
        output_dir: Where to save results
        metrics_pairs: List of (nr_metric, detection_metric) tuples to correlate.
                       Default: all combinations

    Returns:
        Correlation results dict
    """
    if metrics_pairs is None:
        metrics_pairs = [
            ("niqe_mean", "mAP_50"),
            ("niqe_mean", "mAP_50_95"),
            ("brisque_mean", "mAP_50"),
            ("brisque_mean", "mAP_50_95"),
            ("loe_mean", "mAP_50"),
            ("loe_mean", "mAP_50_95"),
        ]

    os.makedirs(output_dir, exist_ok=True)

    # Build aligned data
    scenarios = sorted(set(detection_results.keys()) & set(nr_results.keys()))
    n = len(scenarios)

    print(f"\n[CORRELATION] Scenarios: {scenarios} (n={n})")
    if n < 3:
        print("[CORRELATION] ⚠ Too few data points (need ≥3). Skipping.")
        return {"error": "insufficient data points", "n": n}

    results = {
        "scenarios": scenarios,
        "n_datapoints": n,
        "note": "With n=4, interpret as observational trend (not statistically significant).",
        "correlations": [],
    }

    # Build data table
    data = {"scenario": scenarios}
    for s in scenarios:
        det = detection_results[s]
        nr = nr_results[s]
        for key in ["mAP_50", "mAP_50_95"]:
            data.setdefault(key, []).append(det.get(key, None))
        for key in ["niqe_mean", "brisque_mean", "loe_mean"]:
            data.setdefault(key, []).append(nr.get(key, None))

    df = pd.DataFrame(data)
    print(f"\n[CORRELATION] Data table:")
    print(df.to_string(index=False))

    # Compute correlations
    for nr_metric, det_metric in metrics_pairs:
        if nr_metric not in df.columns or det_metric not in df.columns:
            continue

        x = df[nr_metric].dropna()
        y = df[det_metric].dropna()

        # Align indices
        common = x.index.intersection(y.index)
        if len(common) < 3:
            continue

        x_aligned = x[common].values
        y_aligned = y[common].values

        # Spearman
        rho, p_value = scipy_stats.spearmanr(x_aligned, y_aligned)

        corr_entry = {
            "nr_metric": nr_metric,
            "det_metric": det_metric,
            "spearman_rho": float(rho) if not np.isnan(rho) else None,
            "p_value": float(p_value) if not np.isnan(p_value) else None,
            "n": len(common),
            "interpretation": _interpret_correlation(rho, p_value, len(common)),
        }
        results["correlations"].append(corr_entry)

        print(f"\n  {nr_metric} vs {det_metric}:")
        print(f"    Spearman ρ = {rho:.4f}, p = {p_value:.4f}")
        print(f"    {corr_entry['interpretation']}")

    # Save
    json_path = os.path.join(output_dir, "correlation_analysis.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[CORRELATION] Results saved: {json_path}")

    # Save data table
    csv_path = os.path.join(output_dir, "correlation_data.csv")
    df.to_csv(csv_path, index=False)

    return results


def _interpret_correlation(rho: float, p: float, n: int) -> str:
    """Generate human-readable interpretation of correlation."""
    if rho is None or np.isnan(rho):
        return "Could not compute"

    direction = "negative" if rho < 0 else "positive"
    strength = "strong" if abs(rho) > 0.7 else "moderate" if abs(rho) > 0.4 else "weak"

    # Note about significance with small n
    sig_note = ""
    if n <= 5:
        sig_note = f" (n={n}, interpret as trend only)"

    if p < 0.05:
        return f"{strength.title()} {direction} correlation (ρ={rho:.3f}, p={p:.3f}){sig_note}"
    else:
        return f"{strength.title()} {direction} trend, not significant (ρ={rho:.3f}, p={p:.3f}){sig_note}"
