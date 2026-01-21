#!/usr/bin/env python3
"""
COCO Annotation Size Distribution Analysis and Visualization

This script analyzes COCO-style annotation files and generates:
1. Size distribution histogram (vertical bars)
2. Size-category distribution heatmap

Both visualizations are optimized for two-column document layouts.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
import argparse


# Hardcoded size bins (area in pixels)
SIZE_BINS = {
    "xss": (0, 12**2),
    "xs": (12**2, 20**2),
    "s": (20**2, 32**2),
    "m": (32**2, 64**2),
    "l": (64**2, 128**2),
    "xl": (128**2, 256**2),
    "xxl": (256**2, 1e10),
}


def load_coco_annotations(json_path):
    """Load COCO-style annotation file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def compute_annotation_area(ann):
    """Compute area of an annotation bounding box."""
    if 'area' in ann and ann['area'] > 0:
        return ann['area']
    elif 'bbox' in ann:
        # bbox format: [x, y, width, height]
        return ann['bbox'][2] * ann['bbox'][3]
    return 0


def categorize_by_size(annotations, size_bins):
    """Categorize annotations by size bins."""
    size_distribution = defaultdict(int)
    
    for ann in annotations:
        area = compute_annotation_area(ann)
        
        for bin_name, (lower, upper) in size_bins.items():
            if lower <= area < upper:
                size_distribution[bin_name] += 1
                break
    
    return size_distribution


def compute_size_category_matrix(annotations, categories, size_bins):
    """Compute a matrix of size bins vs categories."""
    # Create category id to name mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
    
    # Initialize matrix: rows=size_bins, cols=categories
    bin_names = list(size_bins.keys())
    cat_ids = sorted(cat_id_to_name.keys())
    
    matrix = np.zeros((len(bin_names), len(cat_ids)))
    
    # Populate matrix
    for ann in annotations:
        area = compute_annotation_area(ann)
        cat_id = ann['category_id']
        
        if cat_id not in cat_id_to_name:
            continue
            
        cat_idx = cat_ids.index(cat_id)
        
        for bin_idx, (bin_name, (lower, upper)) in enumerate(size_bins.items()):
            if lower <= area < upper:
                matrix[bin_idx, cat_idx] += 1
                break
    
    # Compute total instances per category for sorting
    cat_totals = matrix.sum(axis=0)
    
    # Sort categories by total count (descending)
    sorted_indices = np.argsort(-cat_totals)
    matrix = matrix[:, sorted_indices]
    sorted_cat_ids = [cat_ids[i] for i in sorted_indices]
    sorted_cat_names = [cat_id_to_name[cid] for cid in sorted_cat_ids]
    
    return matrix, bin_names, sorted_cat_names


def plot_size_distribution(size_distribution, size_bins, output_path):
    """Create vertical bar chart of size distribution."""
    # Get ordered bin names and counts
    bin_names = list(size_bins.keys())
    counts = [size_distribution[bn] for bn in bin_names]
    
    # Create figure optimized for two-column layout
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    # Create bars
    x_pos = np.arange(len(bin_names))
    bars = ax.bar(x_pos, counts, color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    # Labels and formatting
    ax.set_xlabel('Size bin', fontsize=9)
    ax.set_ylabel('Annotation count', fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_names, fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    
    # Add value labels on top of bars
    # for bar in bars:
    #     height = bar.get_height()
    #     ax.text(bar.get_x() + bar.get_width()/2., height,
    #             f'{int(height)}',
    #             ha='center', va='bottom', fontsize=7)
    
    # Tight layout
    plt.tight_layout()
    
    # Save as PDF
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    
    plt.close(fig)
    print(f"Size distribution saved to: {output_path}")


def plot_size_category_heatmap(matrix, bin_names, cat_names, output_path):
    """Create heatmap of size-category distribution."""
    # Create figure optimized for two-column layout
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    # Create heatmap
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    # Set ticks
    ax.set_xticks(np.arange(len(cat_names)))
    ax.set_yticks(np.arange(len(bin_names)))
    
    # Label only first and last for cleanliness
    x_labels = [''] * len(cat_names)
    if len(cat_names) > 0:
        x_labels[0] = cat_names[0]
        if len(cat_names) > 1:
            x_labels[-1] = cat_names[-1]
    
    y_labels = [''] * len(bin_names)
    if len(bin_names) > 0:
        y_labels[0] = bin_names[0]
        if len(bin_names) > 1:
            y_labels[-1] = bin_names[-1]
    
    ax.set_xticklabels(x_labels, fontsize=8, rotation=90, ha='right')
    ax.set_yticklabels(y_labels, fontsize=8)
    
    # Labels
    ax.set_xlabel('Category (ordered by frequency)', fontsize=9)
    ax.set_ylabel('Size Bin', fontsize=9)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('Count', fontsize=8)
    
    # Tight layout
    plt.tight_layout()
    
    # Save as PDF
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    
    plt.close(fig)
    print(f"Size-category heatmap saved to: {output_path}")


def main():
    """
    python eval/visualize_size_gt.py coco/annotations/instances_val2017.json --hist-output results/gt_size_distribution.pdf --heatmap-output results/gt_size_category_heatmap.pdf
    """
    parser = argparse.ArgumentParser(
        description='Visualize COCO annotation size distributions'
    )
    parser.add_argument(
        'annotation_file',
        help='Path to COCO-style annotations JSON file'
    )
    parser.add_argument(
        '--hist-output',
        default='size_distribution.pdf',
        help='Output path for size distribution histogram (default: size_distribution.pdf)'
    )
    parser.add_argument(
        '--heatmap-output',
        default='size_category_heatmap.pdf',
        help='Output path for size-category heatmap (default: size_category_heatmap.pdf)'
    )
    
    args = parser.parse_args()
    
    # Load annotations
    print(f"Loading annotations from: {args.annotation_file}")
    data = load_coco_annotations(args.annotation_file)
    
    annotations = data.get('annotations', [])
    categories = data.get('categories', [])
    
    print(f"Found {len(annotations)} annotations across {len(categories)} categories")
    
    # Compute size distribution
    print("\nComputing size distribution...")
    size_distribution = categorize_by_size(annotations, SIZE_BINS)
    
    for bin_name, count in size_distribution.items():
        print(f"  {bin_name}: {count}")
    
    # Plot size distribution
    print("\nGenerating size distribution plot...")
    plot_size_distribution(size_distribution, SIZE_BINS, args.hist_output)
    
    # Compute and plot size-category matrix
    print("\nComputing size-category matrix...")
    matrix, bin_names, cat_names = compute_size_category_matrix(
        annotations, categories, SIZE_BINS
    )
    
    print(f"Matrix shape: {matrix.shape} (bins x categories)")
    
    print("\nGenerating size-category heatmap...")
    plot_size_category_heatmap(matrix, bin_names, cat_names, args.heatmap_output)
    
    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()