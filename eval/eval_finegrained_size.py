"""
Fine-grained size-based COCO evaluation.

Evaluates AP across 7 geometric size bins based on object area.

Usage:
    python eval/eval_finegrained_size.py --pred predictions.json
    python eval/eval_finegrained_size.py --gt path/to/gt.json --pred predictions.json --out results.json
    python eval/eval_finegrained_size.py --gt coco/annotations/instances_val2017.json --pred inference_results/cascade_dn_detr_005_cocoval2017.json --out results/cascadedetr_finegrained_metrics.json
"""

import argparse
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# Fine-grained size bins (area thresholds based on side^2)
SIZE_BINS = {
    "xss": (0, 12**2),
    "xs": (12**2, 20**2),
    "s": (20**2, 32**2),
    "m": (32**2, 64**2),
    "l": (64**2, 128**2),
    "xl": (128**2, 256**2),
    "xxl": (256**2, 1e10),
}



def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-grained size-based COCO evaluation"
    )
    parser.add_argument(
        "--gt",
        type=str,
        default=None,
        help="Path to ground truth JSON (default: data/coco2017/.../instances_val2017.json)",
    )
    parser.add_argument(
        "--pred",
        type=str,
        required=True,
        help="Path to predictions JSON (COCO results format)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to save metrics JSON (optional)",
    )
    return parser.parse_args()


def get_default_gt_path() -> Path:
    """Return default path to COCO val2017 ground truth."""
    root = Path(__file__).resolve().parents[1]
    return (
        root
        / "data"
        / "coco2017"
        / "annotations_trainval2017"
        / "annotations"
        / "instances_val2017.json"
    )


def count_gt_in_range(coco_gt: COCO, area_min: float, area_max: float) -> int:
    """Count GT annotations within area range."""
    count = 0
    for ann_id in coco_gt.getAnnIds():
        ann = coco_gt.anns[ann_id]
        area = ann["area"]
        if area_min <= area < area_max:
            count += 1
    return count


def eval_size_bin(
    coco_gt: COCO, coco_dt, bin_name: str, area_min: float, area_max: float
) -> dict:
    """
    Run COCO evaluation for a specific size bin.

    Returns dict with AP metrics for this bin.
    """
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")

    # Set custom area range for this bin
    # areaRng format: [[all], [small], [medium], [large]]
    # We override to evaluate only our custom range
    coco_eval.params.areaRng = [[area_min, area_max]]
    coco_eval.params.areaRngLbl = [bin_name]

    coco_eval.evaluate()
    coco_eval.accumulate()

    # Extract AP from precision array
    # precision has shape [T, R, K, A, M]
    # T: IoU thresholds, R: recall thresholds, K: categories, A: area ranges, M: max dets
    precision = coco_eval.eval["precision"]

    # AP @ IoU=0.50:0.95 (mean over all IoU thresholds)
    # precision[:, :, :, 0, 2] -> all IoU, all recall, all cats, first area range, maxDets=100
    ap = precision[:, :, :, 0, 2].mean()
    if ap < 0:
        ap = 0.0

    # AP @ IoU=0.50 (first threshold is 0.5)
    ap50 = precision[0, :, :, 0, 2].mean()
    if ap50 < 0:
        ap50 = 0.0

    # AP @ IoU=0.75 (threshold index 5 corresponds to 0.75)
    ap75 = precision[5, :, :, 0, 2].mean()
    if ap75 < 0:
        ap75 = 0.0

    return {
        "AP": float(ap),
        "AP50": float(ap50),
        "AP75": float(ap75),
    }


def run_finegrained_eval(gt_path: str, pred_path: str) -> dict:
    """
    Run fine-grained size-based evaluation.

    Returns dict with metrics for each size bin.
    """
    print(f"Loading ground truth: {gt_path}")
    coco_gt = COCO(gt_path)

    print(f"Loading predictions: {pred_path}")
    coco_dt = coco_gt.loadRes(pred_path)

    print("\nRunning fine-grained size evaluation...")
    print("=" * 70)

    results = {}
    total_gt = len(coco_gt.getAnnIds())

    for bin_name, (area_min, area_max) in SIZE_BINS.items():
        gt_count = count_gt_in_range(coco_gt, area_min, area_max)
        gt_pct = 100 * gt_count / total_gt if total_gt > 0 else 0

        metrics = eval_size_bin(coco_gt, coco_dt, bin_name, area_min, area_max)
        metrics["gt_count"] = gt_count
        metrics["gt_percent"] = round(gt_pct, 1)
        metrics["area_range"] = [area_min, area_max]

        results[bin_name] = metrics

    # Also compute overall AP for comparison
    print("\nComputing overall metrics...")
    coco_eval_all = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval_all.evaluate()
    coco_eval_all.accumulate()
    coco_eval_all.summarize()

    results["overall"] = {
        "AP": float(coco_eval_all.stats[0]),
        "AP50": float(coco_eval_all.stats[1]),
        "AP75": float(coco_eval_all.stats[2]),
        "AP_small": float(coco_eval_all.stats[3]),
        "AP_medium": float(coco_eval_all.stats[4]),
        "AP_large": float(coco_eval_all.stats[5]),
        "gt_count": total_gt,
    }

    return results


def print_results(results: dict):
    """Pretty print results table."""
    print("\n" + "=" * 70)
    print("FINE-GRAINED SIZE EVALUATION RESULTS")
    print("=" * 70)

    # Header
    print(f"\n{'Bin':<10} {'Area Range':<20} {'GT Count':>10} {'AP':>8} {'AP50':>8} {'AP75':>8}")
    print("-" * 70)

    # Per-bin results
    for bin_name in SIZE_BINS.keys():
        if bin_name in results:
            r = results[bin_name]
            area_min, area_max = r["area_range"]
            if area_max >= 1e9:
                range_str = f">= {int(area_min)}"
            else:
                range_str = f"[{int(area_min)}, {int(area_max)})"
            print(
                f"{bin_name:<10} {range_str:<20} {r['gt_count']:>10} "
                f"{r['AP']:>8.4f} {r['AP50']:>8.4f} {r['AP75']:>8.4f}"
            )

    print("-" * 70)

    # Overall
    if "overall" in results:
        r = results["overall"]
        print(f"{'OVERALL':<10} {'all':<20} {r['gt_count']:>10} {r['AP']:>8.4f} {r['AP50']:>8.4f} {r['AP75']:>8.4f}")

    # COCO standard comparison
    print("\n" + "-" * 70)
    print("COCO Standard Bins (for comparison):")
    if "overall" in results:
        r = results["overall"]
        print(f"  AP_small  (< 32^2):  {r['AP_small']:.4f}")
        print(f"  AP_medium (32^2-96^2): {r['AP_medium']:.4f}")
        print(f"  AP_large  (>= 96^2): {r['AP_large']:.4f}")

    print("=" * 70)


def main():
    args = parse_args()

    # Resolve ground truth path
    if args.gt is None:
        gt_path = get_default_gt_path()
    else:
        gt_path = Path(args.gt)

    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    pred_path = Path(args.pred)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    # Run evaluation
    results = run_finegrained_eval(str(gt_path), str(pred_path))

    # Print results
    print_results(results)

    # Save metrics if output path specified
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nMetrics saved to: {out_path}")


if __name__ == "__main__":
    main()
