"""
Generate dummy predictions by adding noise to ground truth annotations.

This is useful for testing the evaluation pipeline.
"""

import json
import random
from pathlib import Path


def main():
    random.seed(42)

    root = Path(__file__).resolve().parents[1]
    gt_path = (
        root
        / "data"
        / "coco2017"
        / "annotations_trainval2017"
        / "annotations"
        / "instances_val2017.json"
    )
    out_path = root / "results" / "dummy_predictions.json"

    print(f"Loading GT: {gt_path}")
    with gt_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    annotations = coco["annotations"]
    print(f"Found {len(annotations)} GT annotations")

    predictions = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]

        # Add random noise ±5-10% to each coordinate
        noise_x = random.uniform(-0.10, 0.10) * w
        noise_y = random.uniform(-0.10, 0.10) * h
        noise_w = random.uniform(-0.10, 0.10) * w
        noise_h = random.uniform(-0.10, 0.10) * h

        new_x = max(0, x + noise_x)
        new_y = max(0, y + noise_y)
        new_w = max(1, w + noise_w)
        new_h = max(1, h + noise_h)

        # Random score between 0.3 and 0.99
        score = random.uniform(0.3, 0.99)

        pred = {
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "bbox": [round(new_x, 2), round(new_y, 2), round(new_w, 2), round(new_h, 2)],
            "score": round(score, 4),
        }
        predictions.append(pred)

    # Save predictions
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f)

    print(f"Generated {len(predictions)} predictions")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
