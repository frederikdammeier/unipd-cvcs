"""
COCO Evaluation with Size-Based Analysis.

Usage:
    python eval/eval_coco_size.py --pred predictions.json
    python eval/eval_coco_size.py --gt path/to/gt.json --pred predictions.json --out results.json
"""

import argparse
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_args():
    parser = argparse.ArgumentParser(
        description="COCO evaluation with size-based metrics"
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


def run_coco_eval(gt_path: str, pred_path: str) -> dict:
    """
    Run COCO evaluation and return metrics dict.

    Args:
        gt_path: Path to ground truth annotations JSON.
        pred_path: Path to predictions JSON in COCO results format.

    Returns:
        Dictionary with AP and AR metrics.
    """
    print(f"Loading ground truth: {gt_path}")
    coco_gt = COCO(gt_path)

    print(f"Loading predictions: {pred_path}")
    coco_dt = coco_gt.loadRes(pred_path)

    print("Running evaluation...")
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract metrics from COCOeval.stats
    # stats[0]:  AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
    # stats[1]:  AP @[ IoU=0.50      | area=   all | maxDets=100 ]
    # stats[2]:  AP @[ IoU=0.75      | area=   all | maxDets=100 ]
    # stats[3]:  AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]
    # stats[4]:  AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
    # stats[5]:  AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]
    # stats[6]:  AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]
    # stats[7]:  AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]
    # stats[8]:  AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
    # stats[9]:  AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]
    # stats[10]: AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
    # stats[11]: AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]

    stats = coco_eval.stats

    metrics = {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR_max1": float(stats[6]),
        "AR_max10": float(stats[7]),
        "AR_max100": float(stats[8]),
        "AR_small": float(stats[9]),
        "AR_medium": float(stats[10]),
        "AR_large": float(stats[11]),
    }

    return metrics


def print_metrics(metrics: dict):
    """Pretty print metrics."""
    print("\n" + "=" * 50)
    print("COCO Evaluation Results (Size-Based)")
    print("=" * 50)

    print("\n--- Average Precision (AP) ---")
    print(f"  AP  @[IoU=0.50:0.95 | all   ]: {metrics['AP']:.4f}")
    print(f"  AP  @[IoU=0.50      | all   ]: {metrics['AP50']:.4f}")
    print(f"  AP  @[IoU=0.75      | all   ]: {metrics['AP75']:.4f}")
    print(f"  AP  @[IoU=0.50:0.95 | small ]: {metrics['AP_small']:.4f}")
    print(f"  AP  @[IoU=0.50:0.95 | medium]: {metrics['AP_medium']:.4f}")
    print(f"  AP  @[IoU=0.50:0.95 | large ]: {metrics['AP_large']:.4f}")

    print("\n--- Average Recall (AR) ---")
    print(f"  AR  @[IoU=0.50:0.95 | maxDets=  1]: {metrics['AR_max1']:.4f}")
    print(f"  AR  @[IoU=0.50:0.95 | maxDets= 10]: {metrics['AR_max10']:.4f}")
    print(f"  AR  @[IoU=0.50:0.95 | maxDets=100]: {metrics['AR_max100']:.4f}")
    print(f"  AR  @[IoU=0.50:0.95 | small      ]: {metrics['AR_small']:.4f}")
    print(f"  AR  @[IoU=0.50:0.95 | medium     ]: {metrics['AR_medium']:.4f}")
    print(f"  AR  @[IoU=0.50:0.95 | large      ]: {metrics['AR_large']:.4f}")

    print("\n" + "=" * 50)


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
    metrics = run_coco_eval(str(gt_path), str(pred_path))

    # Print results
    print_metrics(metrics)

    # Save metrics if output path specified
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {out_path}")


if __name__ == "__main__":
    main()
