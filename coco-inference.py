"""
COCO-style Dataset Inference Script using PyTorch Distributed Data Parallel.

This script performs inference on COCO-style datasets using torchvision's
pretrained detection models and outputs predictions in COCO format.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import models
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from tqdm import tqdm


def get_model(model_name: str, num_classes: int = 91, pretrained: bool = True):
    """
    Load a pretrained detection model from torchvision.models.detection.
    
    Args:
        model_name: Name of the model (e.g., 'fasterrcnn_resnet50_fpn')
        num_classes: Number of classes (default: 91 for COCO)
        pretrained: Whether to load pretrained weights
    
    Returns:
        The detection model
    """
    detection_models = {
        'fcos_resnet50_fpn': models.detection.fcos_resnet50_fpn,
        'fasterrcnn_resnet50_fpn': models.detection.fasterrcnn_resnet50_fpn,
        'fasterrcnn_resnet50_fpn_v2': models.detection.fasterrcnn_resnet50_fpn_v2,
        'retinanet_resnet50_fpn': models.detection.retinanet_resnet50_fpn,
        'retinanet_resnet50_fpn_v2': models.detection.retinanet_resnet50_fpn_v2,
    }
    
    if model_name not in detection_models:
        raise ValueError(f"Model {model_name} not found. Available: {list(detection_models.keys())}")
    
    model_fn = detection_models[model_name]
    model = model_fn(pretrained=pretrained, num_classes=num_classes)
    return model


def collate_fn(batch):
    """Custom collate function for detection datasets."""
    images = []
    targets = []
    image_ids = []
    
    for img, target, img_id in batch:
        images.append(img)
        targets.append(target)
        image_ids.append(img_id)
    
    return images, targets, image_ids


class CocoDetectionWithImageId(CocoDetection):
    """CocoDetection that returns image ID along with image and target."""
    
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        img_id = self.ids[idx]
        return img, target, img_id


def run_inference(
    model,
    data_loader: DataLoader,
    device: torch.device,
    conf_threshold: float = 0.5,
    rank: int = 0,
) -> List[Dict[str, Any]]:
    """
    Run inference on the dataset.
    
    Args:
        model: Detection model
        data_loader: DataLoader for the dataset
        device: Device to run inference on
        conf_threshold: Confidence threshold for predictions
        rank: Rank of the process (for distributed logging)
    
    Returns:
        List of COCO-format predictions
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        iterator = tqdm(data_loader, disable=rank != 0)
        for images, targets, image_ids in iterator:
            # Move images to device
            images = [img.to(device) for img in images]
            
            # Run inference
            outputs = model(images)
            
            # Process outputs
            for output, img_id in zip(outputs, image_ids):
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                
                # Filter by confidence threshold
                keep = scores >= conf_threshold
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
                
                # Convert to COCO format
                for box, score, label in zip(boxes, scores, labels):
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    prediction = {
                        'image_id': int(img_id),
                        'category_id': int(label),
                        'bbox': [float(x_min), float(y_min), float(width), float(height)],
                        'score': float(score),
                    }
                    predictions.append(prediction)
    
    return predictions


def setup_distributed():
    """Setup distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(rank)
        return rank, world_size
    return 0, 1


def cleanup_distributed():
    """Cleanup distributed training environment."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def gather_predictions(predictions: List[Dict], rank: int, world_size: int):
    """Gather predictions from all processes."""
    if world_size == 1:
        return predictions
    
    # Serialize predictions
    predictions_json = json.dumps(predictions)
    predictions_bytes = predictions_json.encode('utf-8')
    
    if rank == 0:
        # Gather from all processes
        gathered_list = [None] * world_size
        dist.gather_object(predictions_bytes, gathered_list, dst=0)
        
        # Merge predictions
        all_predictions = []
        for pred_bytes in gathered_list:
            all_predictions.extend(json.loads(pred_bytes.decode('utf-8')))
        return all_predictions
    else:
        dist.gather_object(predictions_bytes, dst=0)
        return []


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='COCO-style Dataset Inference')
    parser.add_argument('--images_path', type=str, required=True,
                        help='Path to COCO images directory')
    parser.add_argument('--annotations_path', type=str, required=True,
                        help='Path to COCO annotations JSON file')
    parser.add_argument('--model', type=str, default='fasterrcnn_resnet50_fpn',
                        help='Detection model name (default: fasterrcnn_resnet50_fpn)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output path for predictions JSON')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference (default: 8)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading (default: 4)')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help='Confidence threshold for predictions (default: 0.5)')
    parser.add_argument('--num_classes', type=int, default=91,
                        help='Number of classes in the model (default: 91 for COCO)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Use model without pretrained weights')
    
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size = setup_distributed()
    
    # Create output directory
    if rank == 0:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Device setup
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}' if world_size > 1 else 'cuda')
    else:
        device = torch.device('cpu')
    
    if rank == 0:
        print(f'Using device: {device}')
        print(f'World size: {world_size}')
        print(f'Model: {args.model}')
    
    # Load model
    model = get_model(args.model, num_classes=args.num_classes, 
                      pretrained=not args.no_pretrained)
    model = model.to(device)
    
    # Wrap with DDP if distributed
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # Load dataset
    dataset = CocoDetectionWithImageId(
        root=args.images_path,
        annFile=args.annotations_path,
    )
    
    # Create sampler and dataloader
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    ) if world_size > 1 else None
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=False if sampler else False,
    )
    
    # Run inference
    if rank == 0:
        print('Starting inference...')
    
    predictions = run_inference(
        model,
        data_loader,
        device,
        conf_threshold=args.conf_threshold,
        rank=rank,
    )
    
    # Gather predictions from all processes
    all_predictions = gather_predictions(predictions, rank, world_size)
    
    # Save predictions
    if rank == 0:
        with open(args.output_path, 'w') as f:
            json.dump(all_predictions, f)
        print(f'Saved {len(all_predictions)} predictions to {args.output_path}')
    
    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    main()
