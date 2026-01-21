#!/usr/bin/env python3
"""
Bounding Box Comparison Visualizer for Object Detection

This script generates a publication-ready visualization comparing predictions from
three object detectors against ground truth annotations in COCO format.

Author: Claude
Date: January 2026
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from PIL import Image
import numpy as np


def load_coco_annotations(annotation_path):
    """Load COCO format annotations from JSON file."""
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    return data


def load_coco_predictions(prediction_path):
    """Load COCO format predictions from JSON file."""
    with open(prediction_path, 'r') as f:
        predictions = json.load(f)
    return predictions


def get_annotations_for_image(annotations, image_id):
    """Extract all annotations for a specific image ID."""
    if isinstance(annotations, dict) and 'annotations' in annotations:
        # Ground truth format
        return [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
    elif isinstance(annotations, list):
        # Prediction format
        return [ann for ann in annotations if ann['image_id'] == image_id]
    else:
        return []


def get_image_info(coco_data, image_id):
    """Get image information from COCO annotations."""
    for img in coco_data['images']:
        if img['id'] == image_id:
            return img
    return None


def draw_bboxes_on_axis(ax, image_path, bboxes, color, linewidth=2, square_crop=True):
    """
    Draw bounding boxes on a matplotlib axis.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to draw on
    image_path : str
        Path to image file
    bboxes : list
        List of bounding box annotations
    color : str
        Color for bounding boxes
    linewidth : float
        Width of bounding box lines
    square_crop : bool
        If True, apply centered square crop to image
    """
    # Load image
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Apply square crop if requested
    crop_offset_x = 0
    crop_offset_y = 0
    
    if square_crop:
        # Calculate square crop dimensions (centered)
        crop_size = min(img_width, img_height)
        crop_offset_x = (img_width - crop_size) // 2
        crop_offset_y = (img_height - crop_size) // 2
        
        # Crop the image
        img = img.crop((
            crop_offset_x,
            crop_offset_y,
            crop_offset_x + crop_size,
            crop_offset_y + crop_size
        ))
    
    # Display image
    ax.imshow(img)
    
    # Draw each bounding box, adjusted for crop
    for bbox in bboxes:
        # COCO format: [x, y, width, height]
        x, y, w, h = bbox['bbox']
        
        # Adjust coordinates for crop
        adjusted_x = x - crop_offset_x
        adjusted_y = y - crop_offset_y
        
        # Only draw if bbox is at least partially visible in crop
        if square_crop:
            crop_size = min(img_width, img_height)
            # Check if bbox intersects with crop region
            if (adjusted_x + w > 0 and adjusted_x < crop_size and
                adjusted_y + h > 0 and adjusted_y < crop_size):
                
                # Create rectangle patch
                rect = patches.Rectangle(
                    (adjusted_x, adjusted_y), w, h,
                    linewidth=linewidth,
                    edgecolor=color,
                    facecolor='none'
                )
                ax.add_patch(rect)
        else:
            # Create rectangle patch without adjustment
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=linewidth,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    
    # Remove all spines (borders)
    for spine in ax.spines.values():
        spine.set_visible(False)


def create_comparison_visualization(
    gt_path,
    pred1_path,
    pred2_path,
    pred3_path,
    image_folder,
    image_ids,
    output_path,
    subtitles=None,
    square_crop=True,
    show_titles=True
):
    """
    Create the main comparison visualization.
    
    Parameters:
    -----------
    gt_path : str
        Path to ground truth COCO annotations
    pred1_path, pred2_path, pred3_path : str
        Paths to prediction files in COCO format
    image_folder : str
        Path to folder containing images
    image_ids : list
        List of 4 image IDs to visualize
    output_path : str
        Path for output PDF file
    subtitles : list, optional
        List of 4 subtitles for the prediction sets (GT + 3 predictions)
    square_crop : bool, optional
        If True, apply centered square crop to all images (default: True)
    show_titles : bool, optional
        If True, show titles above each column (default: True)
    """
    # Default subtitles
    if subtitles is None:
        subtitles = ['Ground Truth', 'Detector A', 'Detector B', 'Detector C']
    
    # Load data
    print("Loading annotations and predictions...")
    gt_data = load_coco_annotations(gt_path)
    pred1_data = load_coco_predictions(pred1_path)
    pred2_data = load_coco_predictions(pred2_path)
    pred3_data = load_coco_predictions(pred3_path)
    
    # Define colors for each set
    colors = ['#00FF00', '#FF4444', '#4444FF', '#FF8800']  # Green, Red, Blue, Orange
    
    # Verify we have exactly 4 image IDs
    if len(image_ids) != 4:
        raise ValueError(f"Expected 4 image IDs, got {len(image_ids)}")
    
    # Create figure with custom layout
    # Figure size optimized for full-width report figure
    # Adjust height based on whether titles are shown
    if show_titles:
        fig = plt.figure(figsize=(16, 4.3))  # Shorter figure with titles
        # Leave minimal space at top for titles
        top_margin = 0.88
        bottom_margin = 0.02
        title_y_pos = 0.93  # Position titles very close to images
    else:
        fig = plt.figure(figsize=(16, 4))  # Even shorter without titles
        top_margin = 0.98
        bottom_margin = 0.02
    
    all_data = [gt_data, pred1_data, pred2_data, pred3_data]
    
    # Calculate column positions for tight 2x2 grids with gaps between groups
    # Each group gets equal space, with small gaps between groups
    group_width = 0.245  # Width allocated to each 2x2 grid
    gap_width = 0.005    # Small gap between groups
    
    # For perfectly square 2x2 grids with equal spacing
    # Use the same spacing value for both wspace and hspace
    # This needs to be a ratio relative to the subplot size
    internal_spacing_ratio = 0.02  # Small equal spacing between images
    
    # Iterate through each column (GT + 3 predictions)
    for col_idx in range(4):
        data = all_data[col_idx]
        color = colors[col_idx]
        
        # Calculate left position for this group
        left_pos = 0.01 + col_idx * (group_width + gap_width)
        
        # Create tight 2×2 grid for this column with equal spacing
        inner_gs = GridSpec(
            2, 2,
            left=left_pos,
            right=left_pos + group_width,
            top=top_margin,
            bottom=bottom_margin,
            wspace=internal_spacing_ratio,  # Equal spacing horizontally
            hspace=internal_spacing_ratio   # Equal spacing vertically
        )
        
        for img_idx, image_id in enumerate(image_ids):
            # Calculate position in 2×2 grid
            row = img_idx // 2
            col = img_idx % 2
            
            ax = fig.add_subplot(inner_gs[row, col])
            
            # Get image info and path
            img_info = get_image_info(gt_data, image_id)
            if img_info is None:
                print(f"Warning: Image ID {image_id} not found in ground truth data")
                continue
            
            image_filename = img_info['file_name']
            image_path = Path(image_folder) / image_filename
            
            if not image_path.exists():
                print(f"Warning: Image {image_path} not found")
                continue
            
            # Get annotations for this image
            if col_idx == 0:
                # Ground truth
                bboxes = get_annotations_for_image(data, image_id)
            else:
                # Predictions
                bboxes = get_annotations_for_image(data, image_id)
            
            # Draw bounding boxes with square crop
            draw_bboxes_on_axis(ax, image_path, bboxes, color, linewidth=2.5, 
                              square_crop=square_crop)
        
        # Add subtitle at the top of each column if titles are enabled
        if show_titles:
            # Calculate center position of the 2×2 grid
            title_x = left_pos + group_width / 2
            
            fig.text(
                title_x, title_y_pos,
                subtitles[col_idx],
                ha='center',
                va='bottom',
                fontsize=14,
                fontweight='bold'
            )
    
    # Save figure
    print(f"Saving visualization to {output_path}...")
    plt.savefig(
        output_path,
        format='pdf',
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.1
    )
    plt.close()
    
    print("Visualization complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Generate comparison visualization for object detection predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python bbox_comparison_visualizer.py \\
        --gt annotations.json \\
        --pred1 predictions_detector1.json \\
        --pred2 predictions_detector2.json \\
        --pred3 predictions_detector3.json \\
        --images ./images \\
        --image-ids 1 2 3 4 \\
        --output comparison.pdf \\
        --subtitles "Ground Truth" "YOLOv8" "Faster R-CNN" "RetinaNet"
        """
    )
    
    parser.add_argument(
        '--gt',
        required=True,
        help='Path to ground truth annotations in COCO format'
    )
    parser.add_argument(
        '--pred1',
        required=True,
        help='Path to first set of predictions in COCO format'
    )
    parser.add_argument(
        '--pred2',
        required=True,
        help='Path to second set of predictions in COCO format'
    )
    parser.add_argument(
        '--pred3',
        required=True,
        help='Path to third set of predictions in COCO format'
    )
    parser.add_argument(
        '--images',
        required=True,
        help='Path to folder containing images'
    )
    parser.add_argument(
        '--image-ids',
        nargs=4,
        type=int,
        required=True,
        help='Four image IDs to visualize'
    )
    parser.add_argument(
        '--output',
        default='bbox_comparison.pdf',
        help='Output PDF path (default: bbox_comparison.pdf)'
    )
    parser.add_argument(
        '--subtitles',
        nargs=4,
        help='Custom subtitles for the four columns (GT, Pred1, Pred2, Pred3)'
    )
    parser.add_argument(
        '--no-square-crop',
        action='store_true',
        help='Disable centered square cropping of images (default: cropping enabled)'
    )
    parser.add_argument(
        '--no-titles',
        action='store_true',
        help='Hide column titles/subtitles (default: titles shown)'
    )
    
    args = parser.parse_args()
    
    create_comparison_visualization(
        gt_path=args.gt,
        pred1_path=args.pred1,
        pred2_path=args.pred2,
        pred3_path=args.pred3,
        image_folder=args.images,
        image_ids=args.image_ids,
        output_path=args.output,
        subtitles=args.subtitles,
        square_crop=not args.no_square_crop,
        show_titles=not args.no_titles
    )


if __name__ == '__main__':
    main()