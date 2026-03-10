#!/usr/bin/env python3
"""
Fase 5-6: Efficiency & Quality Measurement
==========================================
Runs NR-IQA metrics, latency benchmarks, and FLOPs computation.

Usage:
    python scripts/measure_efficiency.py --scenario s1_raw [--quick-test]
    python scripts/measure_efficiency.py --all [--quick-test]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.seed import set_global_seed
from src.evaluation.nr_metrics import compute_nr_metrics
from src.evaluation.latency import measure_latency
from src.evaluation.flops import compute_all_flops
from src.training.train_yolo import get_best_weights


ALL_SCENARIOS = ["s1_raw", "s2_hvi_cidnet", "s3_retinexformer", "s4_lyt_net"]


def run_scenario(scenario: str, quick_test: bool, config_dir: str = None):
    """Run all efficiency measurements for a single scenario."""
    cfg = load_config(scenario, config_dir=config_dir, quick_test=quick_test)
    set_global_seed(cfg["seed"])

    scenario_name = cfg.get("scenario_name", scenario)
    paths = cfg["paths"]
    output_root = paths["output_root"]
    eval_dir = os.path.join(output_root, "evaluation", scenario_name)

    enhancer_name = cfg.get("scenario", {}).get("enhancer", None)
    has_enhancer = enhancer_name and enhancer_name.lower() != "none"

    print("\n" + "=" * 60)
    print(f"EFFICIENCY MEASUREMENT: {scenario_name}")
    print("=" * 60)

    # NR-IQA Metrics
    print("\n[1/3] Computing NR-IQA metrics...")
    if has_enhancer:
        enhanced_dir = os.path.join(output_root, f"ExDark_enhanced_{enhancer_name}")
        raw_dir = os.path.join(output_root, "ExDark_yolo")
    else:
        enhanced_dir = os.path.join(output_root, "ExDark_yolo")
        raw_dir = None

    nr_dir = os.path.join(eval_dir, "nr_metrics")
    nr_results = compute_nr_metrics(
        image_dir=os.path.join(enhanced_dir, "images", "test"),
        raw_dir=os.path.join(raw_dir, "images", "test") if raw_dir else None,
        output_dir=nr_dir,
        cfg=cfg,
    )
    print(f"  NIQE: {nr_results.get('niqe_mean', 'N/A'):.3f}")
    print(f"  BRISQUE: {nr_results.get('brisque_mean', 'N/A'):.3f}")
    if "loe_mean" in nr_results:
        print(f"  LOE: {nr_results.get('loe_mean', 'N/A'):.2f}")

    # Latency
    print("\n[2/3] Measuring inference latency...")
    run_dir = os.path.join(output_root, "runs", scenario_name)
    weights_path = get_best_weights(run_dir)

    latency_dir = os.path.join(eval_dir, "latency")
    latency_results = measure_latency(
        weights_path=weights_path,
        enhancer_name=enhancer_name if has_enhancer else None,
        image_dir=os.path.join(output_root, "ExDark_yolo", "images", "test"),
        output_dir=latency_dir,
        cfg=cfg,
    )
    print(f"  T_enhance: {latency_results.get('T_enhance_mean', 0):.2f} ms")
    print(f"  T_detect: {latency_results.get('T_detect_mean', 0):.2f} ms")
    print(f"  T_total: {latency_results.get('T_total_mean', 0):.2f} ms")

    # FLOPs
    print("\n[3/3] Computing FLOPs...")
    flops_dir = os.path.join(eval_dir, "flops")
    flops_results = compute_all_flops(
        weights_path=weights_path,
        enhancer_name=enhancer_name if has_enhancer else None,
        output_dir=flops_dir,
        cfg=cfg,
    )
    print(f"  Total GFLOPs: {flops_results.get('total_gflops', 'N/A'):.2f}")

    print(f"\n  Results saved to: {eval_dir}")
    return {
        "scenario": scenario_name,
        "nr": nr_results,
        "latency": latency_results,
        "flops": flops_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Efficiency and quality measurement")
    parser.add_argument("--scenario", default=None, help="Single scenario to measure")
    parser.add_argument("--all", action="store_true", help="Run all 4 scenarios")
    parser.add_argument("--quick-test", action="store_true", help="Quick test mode")
    parser.add_argument("--config-dir", default=None, help="Config directory override")
    args = parser.parse_args()

    if args.all:
        scenarios = ALL_SCENARIOS
    elif args.scenario:
        scenarios = [args.scenario]
    else:
        parser.error("Specify --scenario or --all")
        return

    print("=" * 60)
    print("FASE 5-6: EFFICIENCY & QUALITY MEASUREMENT")
    print(f"Scenarios: {scenarios}")
    print("=" * 60)

    all_results = []
    for s in scenarios:
        result = run_scenario(s, args.quick_test, args.config_dir)
        all_results.append(result)

    print("\n" + "=" * 60)
    print("ALL MEASUREMENTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
