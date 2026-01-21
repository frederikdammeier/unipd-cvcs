"""
Inference script for Cascade_DNDetr models (COCO-style predictions)

Produces a COCO-format JSON file with detections (image_id, category_id, bbox, score).
This script re-uses the project's model builder, dataset transforms and post-processing.

Example:
python coco_dn_inference.py \
    --data_path /home/sc.uni-leipzig.de/kv99fuda/dev/Repositories/unipd-cvcs/coco \
    --dataset_file coco \
    --checkpoint checkpoints/coco.pth \
    --output inference-results/cascade_dn_detr_04_cocoval2017.json \
    --image_set val \
    --batch_size 1 \
    --device cuda:0 \
    --conf_threshold 0.4 \
    --modelname cascade_dn_detr \
    --lr_drop 10    \
    --transformer_activation relu   \
    --use_dn        \
    --cascade_attn

If your checkpoint contains a dict with key 'model', that entry will be loaded; otherwise the checkpoint itself is treated as a state_dict.
"""

import json
import tqdm
from pathlib import Path
import torch
from torch.utils.data import DataLoader

import datasets
import util.misc as utils
from main import get_args_parser, build_model_main
from util.utils import clean_state_dict


def convert_xyxy_to_xywh(boxes):
    """Convert Nx4 boxes from [xmin, ymin, xmax, ymax] to [x, y, w, h]."""
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def parse_args():
    parser = get_args_parser()
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint .pth file (or state_dict)')
    parser.add_argument('--output', required=True, help='Output JSON path to save COCO-style predictions')
    parser.add_argument('--image_set', default='val', choices=['train','val','test','eval_debug'], help='Which split to run inference on')
    parser.add_argument('--conf_threshold', default=0.5, type=float, help='Minimum score to keep a prediction')
    return parser.parse_args()


def main():
    args = parse_args()

    # init distributed mode if invoked with torch.distributed launcher or SLURM
    utils.init_distributed_mode(args)

    # device selection: use local rank if using distributed GPUs
    if getattr(args, 'distributed', False):
        device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    print(f'Using device: {device}')

    # Build model (re-uses defaults from training parser)
    model, criterion, postprocessors = build_model_main(args)
    model.to(device)

    # If distributed, wrap model in DDP after moving to device
    if getattr(args, 'distributed', False):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    model.eval()

    # Load checkpoint onto CPU then load into model (module if DDP)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state = ckpt['model']
    else:
        state = ckpt
    state = clean_state_dict(state)

    model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    missing, unexpected = model_to_load.load_state_dict(state, strict=False)
    print('Loaded checkpoint:', args.checkpoint)
    if missing:
        print('Missing keys (model not loaded):', missing)
    if unexpected:
        print('Unexpected keys (ignored):', unexpected)

    # Build dataset
    dataset = datasets.build_dataset(image_set=args.image_set, args=args)

    # Use DistributedSampler in distributed mode so each process works on a subset
    if getattr(args, 'distributed', False):
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    else:
        sampler = None

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=utils.collate_fn, sampler=sampler)

    all_predictions = []

    with torch.no_grad():
        # set sampler epoch for determinism (though no shuffle here)
        if sampler is not None:
            sampler.set_epoch(0)

        for samples, targets in tqdm(data_loader, disable=args.rank != 0):
            samples = samples.to(device)

            # run model (DN handling only if explicitly requested)
            # Always pass dn_args=(targets, num_patterns) to the model in eval mode
            # The model's DN components expect dn_args to be present (it reads num_patterns even in inference).
            outputs = model(samples, dn_args=(targets, args.num_patterns))
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # original sizes: list of [h,w]
            orig_target_sizes = torch.stack([t['orig_size'] for t in targets], dim=0).to(outputs['pred_boxes'].device)

            # apply postprocess (use the 'bbox' postprocessor)
            results = postprocessors['bbox'](outputs, orig_target_sizes)

            # Convert to COCO-style dicts and filter by conf_threshold
            preds_local = []
            for res, tgt in zip(results, targets):
                image_id = int(tgt['image_id'].item())
                if len(res['scores']) == 0:
                    continue
                # ensure tensors on cpu
                scores = res['scores'].cpu()
                labels = res['labels'].cpu()
                boxes = res['boxes'].cpu()

                keep_mask = scores >= args.conf_threshold
                if keep_mask.sum() == 0:
                    continue
                scores = scores[keep_mask]
                labels = labels[keep_mask]
                boxes = boxes[keep_mask]

                # boxes are in xyxy form (abs pixels). Convert to xywh
                boxes_xywh = convert_xyxy_to_xywh(boxes)

                for box, score, label in zip(boxes_xywh.tolist(), scores.tolist(), labels.tolist()):
                    preds_local.append({
                        'image_id': image_id,
                        'category_id': int(label),
                        'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        'score': float(score),
                    })

            # gather preds across processes
            if getattr(args, 'distributed', False):
                gathered = utils.all_gather(preds_local)
                if utils.is_main_process():
                    # flatten list of lists
                    for g in gathered:
                        all_predictions.extend(g)
            else:
                all_predictions.extend(preds_local)

    # Only main process writes output
    if not getattr(args, 'distributed', False) or utils.is_main_process():
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(all_predictions, f)
        print(f'Saved {len(all_predictions)} predictions to {out_path}')
    else:
        print(f'Rank {args.rank} finished; main process will gather and save results.')


if __name__ == '__main__':
    main()
