import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import os

# -------------------------
# Paths (adjust as needed)
# -------------------------
COCO_ROOT = "./coco"
IMG_DIR = os.path.join(COCO_ROOT, "val2017")
ANN_FILE = os.path.join(COCO_ROOT, "annotations/instances_val2017.json")
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Load model
# -------------------------
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights)
model.eval()

# -------------------------
# Load COCO dataset
# -------------------------
dataset = CocoDetection(
    root=IMG_DIR,
    annFile=ANN_FILE,
)

# Select an example image
idx = 0
img, targets = dataset[idx]

# Convert image to tensor
img_tensor = F.to_tensor(img)

# -------------------------
# Inference
# -------------------------
with torch.no_grad():
    prediction = model([img_tensor])[0]

# -------------------------
# Draw predictions
# -------------------------
def draw_boxes(image, boxes, labels=None, scores=None, score_thr=0.5, color="red"):
    draw = ImageDraw.Draw(image)
    for i, box in enumerate(boxes):
        if scores is not None and scores[i] < score_thr:
            continue
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    return image

# Prediction boxes
pred_img = img.copy()
pred_img = draw_boxes(
    pred_img,
    prediction["boxes"].tolist(),
    scores=prediction["scores"].tolist(),
    score_thr=0.5,
    color="red",
)

pred_path = os.path.join(OUTPUT_DIR, "prediction.png")
pred_img.save(pred_path)

# -------------------------
# Draw ground-truth boxes
# -------------------------
gt_img = img.copy()
gt_draw = ImageDraw.Draw(gt_img)

for ann in targets:
    x, y, w, h = ann["bbox"]
    gt_draw.rectangle(
        [x, y, x + w, y + h],
        outline="green",
        width=3,
    )

gt_path = os.path.join(OUTPUT_DIR, "ground_truth.png")
gt_img.save(gt_path)

print(f"Saved prediction to {pred_path}")
print(f"Saved ground truth to {gt_path}")
