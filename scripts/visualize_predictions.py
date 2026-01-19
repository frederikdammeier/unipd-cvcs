"""
Visualize COCO-style predictions vs ground-truths on example images.

Example usage:
python visualize_predictions.py \
    --images_dir coco/val2017 \
    --preds /home/sc.uni-leipzig.de/kv99fuda/dev/Repositories/cascade-detr/cascade_dn_detr/inference-results/cascade_dn_detr_005_cocoval2017.json \
    --gts coco/annotations/instances_val2017.json \
    --out_dir visualizations \
    --conf_threshold 0.25 \
    --num_images 25 \
    --seed 1

Predictions JSON format (one entry per detection):
[{'image_id': int, 'category_id': int, 'bbox': [x,y,w,h], 'score': float}, ...]

Ground-truths should be a COCO annotations JSON with 'images', 'annotations', 'categories'.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def load_predictions(preds_path, conf_thr=0.0, topk=None):
    with open(preds_path, 'r') as f:
        preds = json.load(f)
    by_image = defaultdict(list)
    for p in preds:
        if p.get('score', 1.0) < conf_thr:
            continue
        by_image[int(p['image_id'])].append(p)
    if topk is not None:
        for k, v in by_image.items():
            by_image[k] = sorted(v, key=lambda x: x.get('score', 0.0), reverse=True)[:topk]
    return by_image


def load_gts(gts_path):
    with open(gts_path, 'r') as f:
        coco = json.load(f)
    id2file = {int(i['id']): i['file_name'] for i in coco.get('images', [])}
    cats = {int(c['id']): c.get('name', str(c.get('id'))) for c in coco.get('categories', [])}
    anns_by_image = defaultdict(list)
    for a in coco.get('annotations', []):
        anns_by_image[int(a['image_id'])].append(a)
    return id2file, cats, anns_by_image


def draw_text(draw, xy, text, font, fill=(255, 255, 255), bgcolor=(0, 0, 0)):
    x, y = xy
    # Robust text size measurement: prefer ImageDraw.textbbox, then font.getsize, then font.getbbox, else heuristic
    try:
        bbox = draw.textbbox((x, y), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except AttributeError:
        try:
            tw, th = font.getsize(text)
        except AttributeError:
            try:
                bbox = font.getbbox(text)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except Exception:
                tw = len(text) * getattr(font, 'size', 10) // 2
                th = getattr(font, 'size', 10)
    draw.rectangle([x - 1, y - 1, x + tw + 1, y + th + 1], fill=bgcolor)
    draw.text((x, y), text, font=font, fill=fill)


def visualize_image(img_path, gts, preds, cats, out_path, show_scores=True):
    image = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', size=max(12, image.width // 100))
        font_size = font.size
    except Exception:
        font = ImageFont.load_default()
        font_size = getattr(font, 'size', max(12, image.width // 100))

    # draw GTs in green
    for a in gts:
        x, y, w, h = a['bbox']
        label = cats.get(int(a['category_id']), str(a.get('category_id')))
        draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=max(1, image.width // 300))
        draw_text(draw, (x, y - (font_size + 2)), f'GT: {label}', font, fill=(0, 0, 0), bgcolor=(144, 238, 144))

    # draw Predictions in red
    for p in preds:
        x, y, w, h = p['bbox']
        label = cats.get(int(p['category_id']), str(p.get('category_id')))
        text = f'P: {label}' + (f' {p.get("score",0):.2f}' if show_scores else '')
        draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=max(1, image.width // 300))
        draw_text(draw, (x, y + h + 2), text, font, fill=(255, 255, 255), bgcolor=(255, 160, 160))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def parse_args():
    p = argparse.ArgumentParser(description='Visualize COCO-style predictions vs ground-truths')
    p.add_argument('--images_dir', required=False, help='Directory containing images (optional if GT JSON has file paths)')
    p.add_argument('--preds', required=True, help='Predictions JSON path')
    p.add_argument('--gts', required=True, help='Ground-truth COCO annotations JSON path')
    p.add_argument('--out_dir', default='visualizations', help='Output folder to save annotated images')
    p.add_argument('--num_images', type=int, default=10, help='Number of images to visualize (random sample)')
    p.add_argument('--image_ids', nargs='*', type=int, help='Specific image ids to visualize (overrides num_images)')
    p.add_argument('--conf_threshold', type=float, default=0.05, help='Min score to keep a prediction')
    p.add_argument('--topk', type=int, default=100, help='Max predictions per image to draw')
    p.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    return p.parse_args()


def main():
    args = parse_args()
    preds_by_image = load_predictions(args.preds, conf_thr=args.conf_threshold, topk=args.topk)
    id2file, cats, gts_by_image = load_gts(args.gts)

    all_image_ids = set(list(id2file.keys()) + list(preds_by_image.keys()) + list(gts_by_image.keys()))

    if args.image_ids:
        image_ids = args.image_ids
    else:
        random.seed(args.seed)
        image_ids = random.sample(sorted(list(all_image_ids)), min(args.num_images, len(all_image_ids)))

    print(f'Visualizing {len(image_ids)} images to {args.out_dir}')

    for img_id in image_ids:
        if img_id not in id2file:
            print(f'Warning: image_id {img_id} not found in GT JSON; skipping')
            continue
        fname = id2file[img_id]
        if args.images_dir:
            img_path = Path(args.images_dir) / fname
        else:
            img_path = Path(fname)
        if not img_path.exists():
            print(f'Image file not found: {img_path}; skipping')
            continue
        gts = gts_by_image.get(img_id, [])
        preds = preds_by_image.get(img_id, [])
        out_path = Path(args.out_dir) / f'{img_id}_viz.png'
        visualize_image(img_path, gts, preds, cats, out_path)
        print(f'Saved {out_path} (GT: {len(gts)} anns, Preds: {len(preds)})')

    print('Done')


if __name__ == '__main__':
    main()
