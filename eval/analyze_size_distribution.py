"""
Analyze size distribution of objects in COCO ground truth.

Outputs statistics and suggests fine-grained size bins.
"""

import json
import math
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    gt_path = (
        root
        / "data"
        / "coco2017"
        / "annotations_trainval2017"
        / "annotations"
        / "instances_val2017.json"
    )

    print(f"Loading GT: {gt_path}")
    with gt_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    annotations = coco["annotations"]
    print(f"Total annotations: {len(annotations)}\n")

    # Calculate areas
    areas = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        area = w * h
        areas.append(area)

    areas.sort()

    # Basic statistics
    min_area = min(areas)
    max_area = max(areas)
    mean_area = sum(areas) / len(areas)
    median_area = areas[len(areas) // 2]

    print("=" * 60)
    print("AREA STATISTICS")
    print("=" * 60)
    print(f"  Min area:    {min_area:>12.2f} px^2 ({math.sqrt(min_area):>7.1f}px side)")
    print(f"  Max area:    {max_area:>12.2f} px^2 ({math.sqrt(max_area):>7.1f}px side)")
    print(f"  Mean area:   {mean_area:>12.2f} px^2 ({math.sqrt(mean_area):>7.1f}px side)")
    print(f"  Median area: {median_area:>12.2f} px^2 ({math.sqrt(median_area):>7.1f}px side)")

    # Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("\n  Percentiles:")
    for p in percentiles:
        idx = int(len(areas) * p / 100)
        val = areas[min(idx, len(areas) - 1)]
        print(f"    P{p:02d}: {val:>12.2f} px^2 ({math.sqrt(val):>7.1f}px side)")

    # COCO standard bins
    SMALL_THRESH = 32 * 32  # 1024
    MEDIUM_THRESH = 96 * 96  # 9216

    small = sum(1 for a in areas if a < SMALL_THRESH)
    medium = sum(1 for a in areas if SMALL_THRESH <= a < MEDIUM_THRESH)
    large = sum(1 for a in areas if a >= MEDIUM_THRESH)

    print("\n" + "=" * 60)
    print("COCO STANDARD SIZE BINS")
    print("=" * 60)
    print(f"  Small  (area < 32^2):       {small:>6} ({100*small/len(areas):>5.1f}%)")
    print(f"  Medium (32^2 <= area < 96^2): {medium:>6} ({100*medium/len(areas):>5.1f}%)")
    print(f"  Large  (area >= 96^2):      {large:>6} ({100*large/len(areas):>5.1f}%)")

    # Suggest fine-grained bins based on percentiles
    print("\n" + "=" * 60)
    print("SUGGESTED FINE-GRAINED BINS (8 bins)")
    print("=" * 60)

    # Use percentile-based bins for more even distribution
    bin_percentiles = [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100]
    bin_edges = []
    for p in bin_percentiles:
        idx = int(len(areas) * p / 100)
        idx = min(idx, len(areas) - 1)
        bin_edges.append(areas[idx])

    # Round bin edges to nice numbers
    def round_to_nice(val):
        if val < 100:
            return round(val / 10) * 10
        elif val < 1000:
            return round(val / 50) * 50
        elif val < 10000:
            return round(val / 500) * 500
        else:
            return round(val / 1000) * 1000

    nice_edges = [round_to_nice(e) for e in bin_edges]
    # Ensure unique and sorted
    nice_edges = sorted(set(nice_edges))

    print("\n  Percentile-based bins (equal count):")
    for i in range(len(bin_edges) - 1):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        count = sum(1 for a in areas if lo <= a < hi) if i < len(bin_edges) - 2 else sum(1 for a in areas if lo <= a <= hi)
        pct = 100 * count / len(areas)
        print(f"    Bin {i+1}: [{lo:>10.0f}, {hi:>10.0f}) px^2 -> {count:>5} objects ({pct:>5.1f}%)")

    # Also suggest geometric bins (more interpretable)
    print("\n  Geometric bins (doubling side length):")
    geo_edges = [0, 16*16, 32*32, 64*64, 128*128, 256*256, 512*512, float('inf')]
    geo_labels = [
        "tiny       (< 16^2)    ",
        "very small (16^2-32^2) ",
        "small      (32^2-64^2) ",
        "medium     (64^2-128^2)",
        "large      (128^2-256^2)",
        "very large (256^2-512^2)",
        "huge       (>= 512^2)  ",
    ]

    for i, label in enumerate(geo_labels):
        lo = geo_edges[i]
        hi = geo_edges[i + 1]
        count = sum(1 for a in areas if lo <= a < hi)
        pct = 100 * count / len(areas)
        side_lo = math.sqrt(lo) if lo > 0 else 0
        side_hi = math.sqrt(hi) if hi != float('inf') else float('inf')
        print(f"    {label}: {count:>6} objects ({pct:>5.1f}%)")

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print("""
  For size-based evaluation, consider using geometric bins:
  - Bins based on doubling side length are more interpretable
  - They align with how CNNs process features at different scales
  - Suggested thresholds: 16^2, 32^2, 64^2, 128^2, 256^2, 512^2

  Alternative: use COCO's 3-bin system but report per-bin breakdown
  in your evaluation to show detector behavior across object sizes.
""")


if __name__ == "__main__":
    main()
