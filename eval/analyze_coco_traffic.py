"""
Analyze traffic-related object classes in COCO dataset.

Usage:
    python eval/analyze_coco_traffic.py
"""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GT_PATH = (
    ROOT
    / "data"
    / "coco2017"
    / "annotations_trainval2017"
    / "annotations"
    / "instances_val2017.json"
)


def main():
    print(f"Annotations file: {GT_PATH}")
    assert GT_PATH.exists(), f"instances_val2017.json not found: {GT_PATH}"

    # Load JSON
    with GT_PATH.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    print(f"Total images in val2017: {len(images)}")
    print(f"Total annotations in val2017: {len(annotations)}")
    print(f"Total categories: {len(categories)}")
    print()

    # List all COCO classes
    print("All COCO categories (id, name, supercategory):")
    for cat in categories:
        print(f"{cat['id']:>2}  {cat['name']:15s}  (supercategory: {cat['supercategory']})")
    print()

    # Find traffic classes
    traffic_class_names = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

    name_to_id = {cat["name"]: cat["id"] for cat in categories}
    id_to_name = {cat["id"]: cat["name"] for cat in categories}

    traffic_ids = []
    print("Traffic classes and their IDs:")
    for name in traffic_class_names:
        cid = name_to_id.get(name)
        if cid is None:
            print(f"  WARNING: class '{name}' not found in COCO categories!")
        else:
            traffic_ids.append(cid)
            print(f"  {name:10s} -> id = {cid}")
    print()

    if not traffic_ids:
        raise RuntimeError("No traffic class IDs found, check names!")

    # Count annotations and images for traffic classes
    ann_count_per_class = defaultdict(int)
    image_ids_per_class = defaultdict(set)

    for ann in annotations:
        cid = ann["category_id"]
        if cid in traffic_ids:
            ann_count_per_class[cid] += 1
            image_ids_per_class[cid].add(ann["image_id"])

    print("Traffic stats (per class):")
    for cid in traffic_ids:
        name = id_to_name[cid]
        num_anns = ann_count_per_class[cid]
        num_imgs = len(image_ids_per_class[cid])
        print(f"  {name:10s} (id={cid:2d}): {num_anns:6d} annotations in {num_imgs:5d} images")
    print()

    # Total images with at least one traffic object
    all_traffic_image_ids = set()
    for s in image_ids_per_class.values():
        all_traffic_image_ids |= s

    print(f"Total images with at least one traffic object: {len(all_traffic_image_ids)}")
    print("Done.")


if __name__ == "__main__":
    main()
