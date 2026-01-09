"""
Visualize size-based evaluation results.

Creates publication-ready plots for size analysis.

Usage:
    python eval/visualize_size_analysis.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
GT_PATH = (
    ROOT
    / "data"
    / "coco2017"
    / "annotations_trainval2017"
    / "annotations"
    / "instances_val2017.json"
)

# Model configuration
MODELS = {
    "Faster R-CNN": {
        "color": "#1f77b4",
        "marker": "o",
        "finegrained": "fasterrcnn_finegrained_metrics.json",
        "longest_edge": "fasterrcnn_longest_edge.json",
    },
    "FCOS": {
        "color": "#ff7f0e",
        "marker": "s",
        "finegrained": "fcos_finegrained_metrics.json",
        "longest_edge": "fcos_longest_edge.json",
    },
    "RetinaNet": {
        "color": "#2ca02c",
        "marker": "^",
        "finegrained": "retinanet_finegrained_metrics.json",
        "longest_edge": "retinanet_longest_edge.json",
    },
}

# Bin definitions
AREA_BINS = ["tiny", "xs", "small", "medium", "large", "xl", "huge"]
AREA_BIN_LABELS = ["<16^2", "16^2-32^2", "32^2-64^2", "64^2-128^2", "128^2-256^2", "256^2-512^2", ">=512^2"]

EDGE_BINS = ["tiny", "xs", "small", "medium", "large", "xl", "huge"]
EDGE_BIN_LABELS = ["<16", "16-32", "32-64", "64-128", "128-256", "256-512", ">=512"]

# COCO standard thresholds
COCO_SMALL = 32 * 32  # 1024
COCO_MEDIUM = 96 * 96  # 9216


def load_gt_areas():
    """Load GT annotation areas."""
    print(f"Loading GT: {GT_PATH}")
    with GT_PATH.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    areas = []
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        areas.append(w * h)
    return np.array(areas)


def load_metrics():
    """Load all model metrics from JSON files."""
    metrics = {}
    for model_name, config in MODELS.items():
        metrics[model_name] = {}

        # Load finegrained metrics
        finegrained_file = RESULTS_DIR / config["finegrained"]
        if finegrained_file.exists():
            with finegrained_file.open("r") as f:
                metrics[model_name]["finegrained"] = json.load(f)
        else:
            print(f"  Warning: {finegrained_file} not found")
            metrics[model_name]["finegrained"] = None

        # Load longest edge metrics
        edge_file = RESULTS_DIR / config["longest_edge"]
        if edge_file.exists():
            with edge_file.open("r") as f:
                metrics[model_name]["longest_edge"] = json.load(f)
        else:
            print(f"  Warning: {edge_file} not found")
            metrics[model_name]["longest_edge"] = None

    return metrics


def plot_gt_distribution(areas, output_path):
    """Plot histogram of GT object sizes with COCO bin boundaries."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use log scale for better visualization
    log_areas = np.log10(areas + 1)

    # Create histogram
    bins = np.linspace(0, 6, 60)  # log10 scale: 1 to 1,000,000
    ax.hist(log_areas, bins=bins, color="#4C72B0", alpha=0.7, edgecolor="white", linewidth=0.5)

    # Add COCO boundary lines
    ax.axvline(np.log10(COCO_SMALL), color="red", linestyle="--", linewidth=2, label=f"Small/Medium (32^2={COCO_SMALL})")
    ax.axvline(np.log10(COCO_MEDIUM), color="orange", linestyle="--", linewidth=2, label=f"Medium/Large (96^2={COCO_MEDIUM})")

    # Add fine-grained bin boundaries
    fine_boundaries = [16**2, 32**2, 64**2, 128**2, 256**2, 512**2]
    for b in fine_boundaries:
        ax.axvline(np.log10(b), color="gray", linestyle=":", linewidth=1, alpha=0.5)

    # Labels and formatting
    ax.set_xlabel("Object Area (log10 scale)", fontsize=12)
    ax.set_ylabel("Number of Objects", fontsize=12)
    ax.set_title("Ground Truth Object Size Distribution (COCO val2017)", fontsize=14, fontweight="bold")

    # Custom x-tick labels
    tick_positions = [np.log10(x) for x in [10, 100, 1000, 10000, 100000]]
    tick_labels = ["10", "100", "1K", "10K", "100K"]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"Total: {len(areas):,}\nMedian: {np.median(areas):,.0f} px^2\nMean: {np.mean(areas):,.0f} px^2"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_ap_by_area_bins(metrics, output_path):
    """Plot AP vs area bins for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(AREA_BINS))

    for model_name, config in MODELS.items():
        model_metrics = metrics[model_name].get("finegrained")
        if model_metrics is None:
            print(f"  Warning: No finegrained metrics for {model_name}, skipping")
            continue

        aps = [model_metrics.get(bin_name, {}).get("AP", 0) for bin_name in AREA_BINS]

        ax.plot(x, aps, marker=config["marker"], color=config["color"],
                linewidth=2, markersize=8, label=model_name)

    ax.set_xlabel("Object Size Bin (by Area)", fontsize=12)
    ax.set_ylabel("Average Precision (AP)", fontsize=12)
    ax.set_title("Detection Performance by Object Area", fontsize=14, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(AREA_BIN_LABELS, rotation=45, ha="right")
    ax.set_ylim(0, 0.7)

    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add vertical line separating COCO small from medium
    ax.axvline(1.5, color="red", linestyle="--", alpha=0.5, label="COCO small/medium")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_ap_by_longest_edge(metrics, output_path):
    """Plot AP vs longest edge bins for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(EDGE_BINS))

    for model_name, config in MODELS.items():
        model_metrics = metrics[model_name].get("longest_edge")
        if model_metrics is None:
            print(f"  Warning: No longest_edge metrics for {model_name}, skipping")
            continue

        aps = [model_metrics.get(bin_name, {}).get("AP", 0) for bin_name in EDGE_BINS]

        ax.plot(x, aps, marker=config["marker"], color=config["color"],
                linewidth=2, markersize=8, label=model_name)

    ax.set_xlabel("Object Size Bin (by Longest Edge, pixels)", fontsize=12)
    ax.set_ylabel("Average Precision (AP)", fontsize=12)
    ax.set_title("Detection Performance by Longest Edge", fontsize=14, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(EDGE_BIN_LABELS, rotation=45, ha="right")
    ax.set_ylim(0, 0.45)

    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Highlight undetectable region
    ax.axvspan(-0.5, 1.5, alpha=0.2, color="red", label="Undetectable (<32px)")
    ax.text(0.5, 0.35, "Undetectable\nRegion", ha="center", fontsize=9, color="darkred")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("Generating Size Analysis Visualizations")
    print("=" * 60)

    # Load GT areas
    areas = load_gt_areas()
    print(f"Loaded {len(areas)} GT annotations")

    # Load model metrics
    print("\nLoading model metrics...")
    metrics = load_metrics()

    # Generate plots
    print("\nGenerating plots...")

    # Plot 1: GT size distribution
    plot_gt_distribution(areas, RESULTS_DIR / "gt_size_distribution.png")

    # Plot 2: AP by area bins
    plot_ap_by_area_bins(metrics, RESULTS_DIR / "ap_by_area_bins.png")

    # Plot 3: AP by longest edge
    plot_ap_by_longest_edge(metrics, RESULTS_DIR / "ap_by_longest_edge.png")

    print("\n" + "=" * 60)
    print("All plots saved to results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
