"""Visualization utilities for generating publication-quality figures.

All plots follow a consistent style suitable for academic papers:
- Grouped bar charts for comparisons across scenarios
- Scatter plots for correlation analysis
- Sample grids for qualitative comparison
- LaTeX-ready table export
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Colab/server
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# --- Style config ---
SCENARIO_COLORS = {
    "S1_Raw": "#5B9BD5",
    "S2_HVI_CIDNet": "#ED7D31",
    "S3_RetinexFormer": "#70AD47",
    "S4_LYT_Net": "#FFC000",
}
SCENARIO_ORDER = ["S1_Raw", "S2_HVI_CIDNet", "S3_RetinexFormer", "S4_LYT_Net"]
SCENARIO_LABELS = {
    "S1_Raw": "S1 (Raw)",
    "S2_HVI_CIDNet": "S2 (HVI-CIDNet)",
    "S3_RetinexFormer": "S3 (RetinexFormer)",
    "S4_LYT_Net": "S4 (LYT-Net)",
}

DPI = 300
FIGSIZE_WIDE = (10, 5)
FIGSIZE_SQUARE = (7, 6)
FONT_SIZE = 11


def _apply_style():
    """Apply consistent plot styling."""
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 2,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def plot_detection_comparison(
    metrics: Dict[str, dict],
    output_path: str,
    metric_keys: Optional[List[str]] = None,
) -> str:
    """Grouped bar chart comparing detection metrics across scenarios.

    Args:
        metrics: {scenario_name: {"mAP_50": float, "mAP_50_95": float, ...}}
        output_path: Save path for the figure
        metric_keys: Which metrics to plot. Default: mAP_50, mAP_50_95

    Returns:
        Saved figure path
    """
    _apply_style()

    if metric_keys is None:
        metric_keys = ["mAP_50", "mAP_50_95"]

    scenarios = [s for s in SCENARIO_ORDER if s in metrics]
    n_scenarios = len(scenarios)
    n_metrics = len(metric_keys)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    x = np.arange(n_scenarios)
    width = 0.8 / n_metrics

    metric_labels = {
        "mAP_50": "mAP@0.5",
        "mAP_50_95": "mAP@0.5:0.95",
        "precision": "Precision",
        "recall": "Recall",
    }

    for i, mk in enumerate(metric_keys):
        values = [metrics[s].get(mk, 0) for s in scenarios]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, values, width,
            label=metric_labels.get(mk, mk),
            color=plt.cm.Set2(i / n_metrics),
            edgecolor="black", linewidth=0.5,
        )
        # Value labels on bars
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=FONT_SIZE - 2,
            )

    labels = [SCENARIO_LABELS.get(s, s) for s in scenarios]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Detection Performance Comparison")
    ax.legend(loc="upper right")
    ax.set_ylim(0, min(1.0, max(max(metrics[s].get(mk, 0) for s in scenarios) for mk in metric_keys) * 1.15))
    ax.grid(axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[VIS] Detection comparison saved: {output_path}")
    return output_path


def plot_per_class_map(
    per_class_data: Dict[str, Dict[str, float]],
    output_path: str,
    metric_key: str = "mAP_50",
) -> str:
    """Grouped bar chart of per-class mAP across scenarios.

    Args:
        per_class_data: {scenario: {class_name: mAP_value}}
        output_path: Save path
        metric_key: Used for title only

    Returns:
        Saved figure path
    """
    _apply_style()

    scenarios = [s for s in SCENARIO_ORDER if s in per_class_data]
    if not scenarios:
        return ""

    # Get all classes from first scenario
    classes = list(per_class_data[scenarios[0]].keys())
    n_classes = len(classes)
    n_scenarios = len(scenarios)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_classes)
    width = 0.8 / n_scenarios

    for i, s in enumerate(scenarios):
        values = [per_class_data[s].get(c, 0) for c in classes]
        offset = (i - n_scenarios / 2 + 0.5) * width
        ax.bar(
            x + offset, values, width,
            label=SCENARIO_LABELS.get(s, s),
            color=SCENARIO_COLORS.get(s, f"C{i}"),
            edgecolor="black", linewidth=0.3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel(metric_key.replace("_", "@").replace("50@95", "0.5:0.95"))
    ax.set_title(f"Per-Class {metric_key} Comparison")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[VIS] Per-class mAP saved: {output_path}")
    return output_path


def plot_nr_metrics(
    nr_data: Dict[str, dict],
    output_path: str,
) -> str:
    """Grouped bar chart for NR-IQA metrics (NIQE, BRISQUE, LOE).

    Args:
        nr_data: {scenario: {"niqe_mean": float, "brisque_mean": float, "loe_mean": float}}
        output_path: Save path

    Returns:
        Saved figure path
    """
    _apply_style()

    scenarios = [s for s in SCENARIO_ORDER if s in nr_data]
    metric_keys = ["niqe_mean", "brisque_mean", "loe_mean"]
    metric_labels = ["NIQE ↓", "BRISQUE ↓", "LOE ↓"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, mk, ml in zip(axes, metric_keys, metric_labels):
        values = [nr_data[s].get(mk, 0) for s in scenarios]
        colors = [SCENARIO_COLORS.get(s, "gray") for s in scenarios]
        labels = [SCENARIO_LABELS.get(s, s) for s in scenarios]

        bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.2f}", ha="center", va="bottom", fontsize=FONT_SIZE - 2,
            )
        ax.set_title(ml)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("No-Reference Image Quality Metrics", fontsize=FONT_SIZE + 2, y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[VIS] NR metrics saved: {output_path}")
    return output_path


def plot_latency_breakdown(
    latency_data: Dict[str, dict],
    output_path: str,
) -> str:
    """Stacked bar chart for latency (T_enhance + T_detect = T_total).

    Args:
        latency_data: {scenario: {"T_enhance_mean": float, "T_detect_mean": float, ...}}
        output_path: Save path

    Returns:
        Saved figure path
    """
    _apply_style()

    scenarios = [s for s in SCENARIO_ORDER if s in latency_data]

    t_enhance = [latency_data[s].get("T_enhance_mean", 0) for s in scenarios]
    t_detect = [latency_data[s].get("T_detect_mean", 0) for s in scenarios]
    labels = [SCENARIO_LABELS.get(s, s) for s in scenarios]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    x = np.arange(len(scenarios))

    ax.bar(x, t_enhance, label="Enhancement", color="#FF6B6B", edgecolor="black", linewidth=0.5)
    ax.bar(x, t_detect, bottom=t_enhance, label="Detection", color="#4ECDC4", edgecolor="black", linewidth=0.5)

    # Total time label
    for i, (te, td) in enumerate(zip(t_enhance, t_detect)):
        total = te + td
        ax.text(i, total + 0.5, f"{total:.1f}", ha="center", va="bottom", fontsize=FONT_SIZE - 1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency Breakdown")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[VIS] Latency breakdown saved: {output_path}")
    return output_path


def plot_correlation_scatter(
    data: Dict[str, dict],
    nr_metric: str,
    det_metric: str,
    output_path: str,
    rho: Optional[float] = None,
    p_value: Optional[float] = None,
) -> str:
    """Scatter plot for NR metric vs mAP correlation.

    Args:
        data: {scenario: {"nr_metric_val": float, "det_metric_val": float}}
        nr_metric: Key for NR metric (e.g., "niqe_mean")
        det_metric: Key for detection metric (e.g., "mAP_50")
        output_path: Save path
        rho: Spearman correlation coefficient (shown in title)
        p_value: p-value (shown in title)

    Returns:
        Saved figure path
    """
    _apply_style()

    scenarios = [s for s in SCENARIO_ORDER if s in data]
    x = [data[s].get(nr_metric, 0) for s in scenarios]
    y = [data[s].get(det_metric, 0) for s in scenarios]
    colors = [SCENARIO_COLORS.get(s, "gray") for s in scenarios]

    fig, ax = plt.subplots(figsize=(6, 5))

    for i, s in enumerate(scenarios):
        ax.scatter(x[i], y[i], c=colors[i], s=120, label=SCENARIO_LABELS.get(s, s),
                   edgecolor="black", linewidth=0.8, zorder=3)

    # Trend line
    if len(x) >= 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x) * 0.95, max(x) * 1.05, 100)
        ax.plot(x_line, p(x_line), "--", color="gray", alpha=0.5, linewidth=1)

    nr_label = nr_metric.replace("_mean", "").upper()
    det_label = det_metric.replace("_", "@").replace("50@95", "0.5:0.95")

    ax.set_xlabel(nr_label)
    ax.set_ylabel(det_label)

    title = f"{nr_label} vs {det_label}"
    if rho is not None:
        title += f"\n(Spearman ρ = {rho:.3f}, p = {p_value:.3f})"
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[VIS] Correlation scatter saved: {output_path}")
    return output_path


def plot_enhancement_samples(
    sample_images: Dict[str, np.ndarray],
    output_path: str,
    title: str = "Enhancement Comparison",
) -> str:
    """Grid showing same image processed by different LLIE methods.

    Args:
        sample_images: {"S1 (Raw)": img_rgb, "S2 (HVI-CIDNet)": img_rgb, ...}
        output_path: Save path
        title: Figure title

    Returns:
        Saved figure path
    """
    _apply_style()
    import cv2

    n = len(sample_images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (label, img) in zip(axes, sample_images.items()):
        if img.ndim == 3 and img.shape[2] == 3:
            # Assume BGR if from cv2
            img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if isinstance(img, np.ndarray) else img
        else:
            img_show = img
        ax.imshow(img_show)
        ax.set_title(label, fontsize=FONT_SIZE - 1)
        ax.axis("off")

    plt.suptitle(title, fontsize=FONT_SIZE + 1)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[VIS] Enhancement samples saved: {output_path}")
    return output_path


# --- LaTeX table export ---

def export_detection_latex(
    metrics: Dict[str, dict],
    output_path: str,
    metric_keys: Optional[List[str]] = None,
) -> str:
    """Export detection metrics as LaTeX table.

    Args:
        metrics: {scenario: {metric_key: value}}
        output_path: .tex file path
        metric_keys: Columns to include

    Returns:
        Saved path
    """
    if metric_keys is None:
        metric_keys = ["mAP_50", "mAP_50_95", "precision", "recall"]

    header_map = {
        "mAP_50": "mAP@0.5",
        "mAP_50_95": "mAP@0.5:0.95",
        "precision": "Precision",
        "recall": "Recall",
    }

    scenarios = [s for s in SCENARIO_ORDER if s in metrics]

    # Find best values per column for bolding
    best = {}
    for mk in metric_keys:
        vals = [metrics[s].get(mk, 0) for s in scenarios]
        best[mk] = max(vals) if vals else 0

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    cols = "l" + "c" * len(metric_keys)
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\toprule")

    header = "Scenario & " + " & ".join(header_map.get(mk, mk) for mk in metric_keys) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for s in scenarios:
        row_vals = []
        for mk in metric_keys:
            v = metrics[s].get(mk, 0)
            cell = f"{v:.4f}"
            if abs(v - best[mk]) < 1e-6 and v > 0:
                cell = f"\\textbf{{{cell}}}"
            row_vals.append(cell)
        label = SCENARIO_LABELS.get(s, s)
        lines.append(f"{label} & " + " & ".join(row_vals) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Detection performance comparison on ExDark test set.}")
    lines.append("\\label{tab:detection-comparison}")
    lines.append("\\end{table}")

    tex = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(tex)
    print(f"[VIS] LaTeX table saved: {output_path}")
    return output_path


def generate_all_figures(
    results_dir: str,
    figures_dir: str,
) -> List[str]:
    """Generate all publication figures from aggregated results.

    Expects results_dir to contain:
        - detection_summary.csv or per-scenario metrics.json
        - nr_summary.json per scenario
        - latency.json per scenario
        - correlation_analysis.json

    Returns:
        List of generated figure paths
    """
    os.makedirs(figures_dir, exist_ok=True)
    generated = []

    # Load detection metrics
    det_path = os.path.join(results_dir, "detection_summary.csv")
    if os.path.exists(det_path):
        det_df = pd.read_csv(det_path)
        det_metrics = {}
        for _, row in det_df.iterrows():
            det_metrics[row["scenario"]] = row.to_dict()

        path = plot_detection_comparison(
            det_metrics,
            os.path.join(figures_dir, "detection_comparison.png"),
        )
        generated.append(path)

    # Load NR metrics
    nr_path = os.path.join(results_dir, "nr_summary.json")
    if os.path.exists(nr_path):
        with open(nr_path) as f:
            nr_data = json.load(f)
        path = plot_nr_metrics(nr_data, os.path.join(figures_dir, "nr_metrics.png"))
        generated.append(path)

    # Load latency
    lat_path = os.path.join(results_dir, "latency_summary.json")
    if os.path.exists(lat_path):
        with open(lat_path) as f:
            lat_data = json.load(f)
        path = plot_latency_breakdown(lat_data, os.path.join(figures_dir, "latency_breakdown.png"))
        generated.append(path)

    print(f"\n[VIS] Generated {len(generated)} figures in {figures_dir}")
    return generated
