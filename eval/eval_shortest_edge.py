"""
Shortest-edge-based COCO evaluation.

Evaluates AP across 7 bins based on object shortest edge (min(w, h)).

Usage:
    python eval/eval_shortest_edge.py --pred predictions.json
    python eval/eval_shortest_edge.py --gt coco/annotations/instances_val2017.json --pred inference_results/cascade_dn_detr_005_cocoval2017.json --out results/cascadedetr_shortest_edge.json
    python eval/eval_shortest_edge.py --gt coco/annotations/instances_val2017.json --pred inference_results/fasterrcnn_resnet50_fpn_v2_cocoval2017.json --out results/fasterrcnn_resnet50_fpn_v2_shortest_edge.json
    python eval/eval_shortest_edge.py --gt coco/annotations/instances_val2017.json --pred inference_results/retinanet_resnet50_fpn_v2_cocoval2017.json --out results/retinanet_resnet50_fpn_v2.json
"""

import argparse
import copy
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# Shortest edge bins (min(w, h) thresholds)
EDGE_BINS = {
    "xxs": (0, 12),
    "xs": (12, 20),
    "s": (20, 32),
    "m": (32, 64),
    "l": (64, 128),
    "xl": (128, 256),
    "xxl": (256, float("inf")),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Shortest-edge-based COCO evaluation"
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


def get_shortest_edge(bbox):
    """Calculate shortest edge from bbox [x, y, w, h]."""
    _, _, w, h = bbox
    return min(w, h)


def filter_annotations_by_edge(coco_gt, edge_min, edge_max):
    """
    Filter GT annotations by shortest edge range.

    Returns list of annotation IDs that fall within the range.
    """
    filtered_ids = []
    for ann_id, ann in coco_gt.anns.items():
        shortest_edge = get_shortest_edge(ann["bbox"])
        if edge_min <= shortest_edge < edge_max:
            filtered_ids.append(ann_id)
    return filtered_ids


def create_filtered_coco(coco_gt, ann_ids):
    """
    Create a new COCO object with only the specified annotation IDs.
    """
    # Get the image IDs that have these annotations
    image_ids = set()
    for ann_id in ann_ids:
        image_ids.add(coco_gt.anns[ann_id]["image_id"])

    # Create filtered dataset dict
    filtered_dataset = {
        "images": [img for img in coco_gt.dataset["images"] if img["id"] in image_ids],
        "annotations": [coco_gt.anns[ann_id] for ann_id in ann_ids],
        "categories": coco_gt.dataset["categories"],
    }

    # Create new COCO object
    filtered_coco = COCO()
    filtered_coco.dataset = filtered_dataset
    filtered_coco.createIndex()

    return filtered_coco


def filter_predictions_by_images(predictions, image_ids):
    """Filter predictions to only include those for specified images."""
    return [p for p in predictions if p["image_id"] in image_ids]


def eval_edge_bin(coco_gt, predictions, bin_name, edge_min, edge_max):
    """
    Run COCO evaluation for a specific shortest edge bin.

    Returns dict with AP metrics for this bin.
    """
    # Filter GT annotations by shortest edge
    filtered_ann_ids = filter_annotations_by_edge(coco_gt, edge_min, edge_max)

    if len(filtered_ann_ids) == 0:
        return {
            "AP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "gt_count": 0,
        }

    # Create filtered COCO GT
    filtered_gt = create_filtered_coco(coco_gt, filtered_ann_ids)

    # Get image IDs in filtered GT
    image_ids = set(filtered_gt.getImgIds())

    # Filter predictions to only include relevant images
    filtered_preds = filter_predictions_by_images(predictions, image_ids)

    if len(filtered_preds) == 0:
        return {
            "AP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "gt_count": len(filtered_ann_ids),
        }

    # Load predictions into filtered COCO
    coco_dt = filtered_gt.loadRes(filtered_preds)

    # Run evaluation
    coco_eval = COCOeval(filtered_gt, coco_dt, iouType="bbox")

    coco_eval.params.areaRng = [[0, 1e10]]
    coco_eval.params.areaRngLbl = ['all']
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Use stats from COCOeval (properly computed AP values)
    # stats[0] = AP @ IoU=0.50:0.95
    # stats[1] = AP @ IoU=0.50
    # stats[2] = AP @ IoU=0.75
    return {
        "AP": float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
        "gt_count": len(filtered_ann_ids),
    }


def run_shortest_edge_eval(gt_path: str, pred_path: str) -> dict:
    """
    Run shortest-edge-based evaluation.

    Returns dict with metrics for each edge bin.
    """
    print(f"Loading ground truth: {gt_path}")
    coco_gt = COCO(gt_path)

    print(f"Loading predictions: {pred_path}")
    with open(pred_path, "r") as f:
        predictions = json.load(f)

    print(f"\nRunning shortest-edge-based evaluation...")
    print("=" * 70)

    results = {}
    total_gt = len(coco_gt.getAnnIds())

    for bin_name, (edge_min, edge_max) in EDGE_BINS.items():
        print(f"  Evaluating bin: {bin_name} [{edge_min}, {edge_max}) px...")

        metrics = eval_edge_bin(coco_gt, predictions, bin_name, edge_min, edge_max)
        metrics["edge_range"] = [edge_min, edge_max if edge_max != float("inf") else 1e10]
        metrics["gt_percent"] = round(100 * metrics["gt_count"] / total_gt, 1)

        results[bin_name] = metrics

    # Also compute overall metrics for comparison
    print("\nComputing overall metrics...")
    coco_dt = coco_gt.loadRes(pred_path)
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
    print("SHORTEST-EDGE-BASED EVALUATION RESULTS")
    print("=" * 70)

    # Header
    print(f"\n{'Bin':<10} {'Edge Range':<15} {'GT Count':>10} {'%':>6} {'AP':>8} {'AP50':>8} {'AP75':>8}")
    print("-" * 70)

    # Per-bin results
    for bin_name in EDGE_BINS.keys():
        if bin_name in results:
            r = results[bin_name]
            edge_min, edge_max = r["edge_range"]
            if edge_max >= 1e9:
                range_str = f">= {int(edge_min)} px"
            else:
                range_str = f"[{int(edge_min)}, {int(edge_max)}) px"
            print(
                f"{bin_name:<10} {range_str:<15} {r['gt_count']:>10} {r['gt_percent']:>5.1f}% "
                f"{r['AP']:>8.4f} {r['AP50']:>8.4f} {r['AP75']:>8.4f}"
            )

    print("-" * 70)

    # Overall
    if "overall" in results:
        r = results["overall"]
        print(f"{'OVERALL':<10} {'all':<15} {r['gt_count']:>10} {'100.0':>5}% {r['AP']:>8.4f} {r['AP50']:>8.4f} {r['AP75']:>8.4f}")

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
    results = run_shortest_edge_eval(str(gt_path), str(pred_path))

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