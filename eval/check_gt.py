import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GT_JSON = (
    ROOT
    / "data"
    / "coco2017"
    / "annotations_trainval2017"
    / "annotations"
    / "instances_val2017.json"
)

print("GT path:", GT_JSON)
assert GT_JSON.exists(), f"GT file not found: {GT_JSON}"

with GT_JSON.open("r", encoding="utf-8") as f:
    coco = json.load(f)

print("Top-level keys:", list(coco.keys()))
print("images:", len(coco["images"]))          # ожидаемо 5000
print("categories:", len(coco["categories"]))  # ожидаемо 80
print("annotations:", len(coco["annotations"]))
print("Example category:", coco["categories"][0])
