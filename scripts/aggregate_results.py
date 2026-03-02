#!/usr/bin/env python3
"""
Fase 7: Aggregate Results & Correlation Analysis
=================================================
Aggregates all scenario results, computes correlations,
and generates publication-ready figures + LaTeX tables.

Usage:
    python scripts/aggregate_results.py [--quick-test]
"""

import argparse
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.evaluation.correlation import compute_spearman_correlation
from src.evaluation.eval_yolo import aggregate_detection_results
from src.utils.visualization import (
    generate_all_figures,
    export_detection_latex,
    plot_correlation_scatter,
)
from src.utils.io import load_json, save_json


SCENARIOS = ["s1_raw", "s2_hvi_cidnet", "s3_retinexformer", "s4_lyt_net"]
SCENARIO_NAMES = ["S1_Raw", "S2_HVI_CIDNet", "S3_RetinexFormer", "S4_LYT_Net"]


def main():
    parser = argparse.ArgumentParser(description="Aggregate results and generate figures")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--config-dir", default=None)
    args = parser.parse_args()

    # Load base config for paths
    cfg = load_config("s1_raw", config_dir=args.config_dir, quick_test=args.quick_test)
    output_root = cfg["paths"]["output_root"]

    summary_dir = os.path.join(output_root, "summary")
    figures_dir = os.path.join(summary_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("FASE 7: RESULTS AGGREGATION")
    print("=" * 60)

    # --- Collect detection metrics ---
    print("\n[1/5] Aggregating detection metrics...")
    det_metrics = {}
    for sc, sn in zip(SCENARIOS, SCENARIO_NAMES):
        metrics_path = os.path.join(output_root, "evaluation", sn, "metrics.json")
        if os.path.exists(metrics_path):
            det_metrics[sn] = load_json(metrics_path)
            print(f"  {sn}: mAP@0.5={det_metrics[sn].get('mAP_50', 'N/A')}")
        else:
            print(f"  {sn}: MISSING ({metrics_path})")

    save_json(det_metrics, os.path.join(summary_dir, "detection_all.json"))

    # --- Collect NR metrics ---
    print("\n[2/5] Aggregating NR-IQA metrics...")
    nr_metrics = {}
    for sc, sn in zip(SCENARIOS, SCENARIO_NAMES):
        nr_path = os.path.join(output_root, "evaluation", sn, "nr_metrics", "summary.json")
        if os.path.exists(nr_path):
            nr_metrics[sn] = load_json(nr_path)
            print(f"  {sn}: NIQE={nr_metrics[sn].get('niqe_mean', 'N/A'):.3f}")
        else:
            print(f"  {sn}: MISSING")

    save_json(nr_metrics, os.path.join(summary_dir, "nr_summary.json"))

    # --- Collect latency ---
    print("\n[3/5] Aggregating latency metrics...")
    lat_metrics = {}
    for sc, sn in zip(SCENARIOS, SCENARIO_NAMES):
        lat_path = os.path.join(output_root, "evaluation", sn, "latency", "latency.json")
        if os.path.exists(lat_path):
            lat_metrics[sn] = load_json(lat_path)
            print(f"  {sn}: T_total={lat_metrics[sn].get('T_total_mean', 'N/A'):.2f} ms")
        else:
            print(f"  {sn}: MISSING")

    save_json(lat_metrics, os.path.join(summary_dir, "latency_summary.json"))

    # --- Correlation analysis ---
    print("\n[4/5] Computing Spearman correlations...")
    if len(det_metrics) >= 3 and len(nr_metrics) >= 3:
        corr_dir = os.path.join(summary_dir, "correlation")
        corr_results = compute_spearman_correlation(
            detection_results=det_metrics,
            nr_results=nr_metrics,
            output_dir=corr_dir,
        )

        # Generate scatter plots
        if "correlations" in corr_results:
            merged = {}
            for sn in SCENARIO_NAMES:
                if sn in det_metrics and sn in nr_metrics:
                    merged[sn] = {**det_metrics[sn], **nr_metrics[sn]}

            for entry in corr_results["correlations"]:
                nr_m = entry["nr_metric"]
                det_m = entry["det_metric"]
                plot_correlation_scatter(
                    data=merged,
                    nr_metric=nr_m,
                    det_metric=det_m,
                    output_path=os.path.join(figures_dir, f"corr_{nr_m}_vs_{det_m}.png"),
                    rho=entry.get("spearman_rho"),
                    p_value=entry.get("p_value"),
                )
    else:
        print("  Skipped: insufficient data")

    # --- Generate figures & LaTeX ---
    print("\n[5/5] Generating figures and LaTeX tables...")
    generate_all_figures(summary_dir, figures_dir)

    if det_metrics:
        export_detection_latex(
            det_metrics,
            os.path.join(summary_dir, "table_detection.tex"),
        )

    print("\n" + "=" * 60)
    print("AGGREGATION COMPLETE")
    print(f"Summary directory: {summary_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
