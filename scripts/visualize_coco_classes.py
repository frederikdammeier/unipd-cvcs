#!/usr/bin/env python3
"""
COCO Dataset Class Distribution Visualizer

This script creates a publication-ready bar chart of class counts from a COCO-style
instances.json file, formatted for a two-column LaTeX document.
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from pathlib import Path


def load_coco_annotations(json_path):
    """Load COCO annotations from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def count_instances_per_class(annotations_data):
    """Count the number of instances for each class."""
    # Create mapping from category_id to category name
    categories = {cat['id']: cat['name'] for cat in annotations_data['categories']}
    
    # Count instances per category
    category_counts = Counter()
    for ann in annotations_data['annotations']:
        category_counts[ann['category_id']] += 1
    
    # Convert to list of (name, count) tuples, including zero counts
    cat_id_to_count = {cat_id: count for cat_id, count in category_counts.items()}
    class_counts = [(categories[cat_id], cat_id_to_count.get(cat_id, 0)) 
                    for cat_id in categories.keys()]
    
    # Sort by count in descending order
    class_counts.sort(key=lambda x: x[1], reverse=True)
    
    return class_counts


def create_visualization(class_counts, output_path):
    """Create and save the bar chart visualization."""
    # Extract names and counts
    class_names = [item[0] for item in class_counts]
    counts = [item[1] for item in class_counts]
    
    # Find min class (now at the end due to descending sort)
    min_idx = len(counts) - 1
    # Find the first non-zero minimum
    for i in range(len(counts) - 1, -1, -1):
        if counts[i] > 0:
            continue
        else:
            min_idx = i
            break
    # If all are non-zero, use the last one
    if counts[min_idx] > 0:
        min_idx = len(counts) - 1
    else:
        # Find first zero from the end
        for i in range(len(counts) - 1, -1, -1):
            if counts[i] == 0:
                min_idx = i
                break
    
    # Set up the figure with appropriate size for single column (3.5 inches width typical)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    # Create bar chart
    x_pos = np.arange(len(class_names))
    bars = ax.bar(x_pos, counts, width=0.8, color='#2E86AB', edgecolor='none', alpha=0.85)
    
    # Highlight only min bar
    bars[min_idx].set_color('#F18F01')
    
    # Configure axes
    ax.set_xlim(-0.5, len(class_names) - 0.5)
    ax.set_ylim(0, max(counts) * 1.05)
    
    # Remove x-axis ticks and labels (except min/max)
    # Max is at index 0 due to descending sort
    ax.set_xticks([0, min_idx])
    ax.set_xticklabels([class_names[0], class_names[min_idx]], 
                       rotation=90, ha='center', va='top', fontsize=8)
    
    # Configure y-axis
    ax.set_ylabel('Instance Count', fontsize=9)
    ax.tick_params(axis='y', labelsize=8)
    
    # Add light grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Tight layout to minimize white space
    plt.tight_layout(pad=0.2)
    
    # Save as PDF (vectorized format)
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Visualization saved to: {output_path}")
    
    # Also show statistics
    print(f"\nDataset Statistics:")
    print(f"Total classes: {len(class_names)}")
    print(f"Total instances: {sum(counts)}")
    print(f"Max class: '{class_names[0]}' with {counts[0]} instances")
    print(f"Min class: '{class_names[min_idx]}' with {counts[min_idx]} instances")
    print(f"Mean instances per class: {np.mean(counts):.1f}")
    print(f"Median instances per class: {np.median(counts):.1f}")


def main():
    """Main function to parse arguments and generate visualization."""
    parser = argparse.ArgumentParser(
        description='Visualize class distribution in COCO-style datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python coco_class_visualizer.py instances_train2017.json
  python coco_class_visualizer.py instances_val2017.json -o val_distribution.pdf
        """
    )
    
    parser.add_argument('input_json', type=str,
                       help='Path to COCO instances JSON file')
    parser.add_argument('-o', '--output', type=str, default='class_distribution.pdf',
                       help='Output PDF file path (default: class_distribution.pdf)')
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_json)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_json}' not found.")
        return 1
    
    # Load and process data
    print(f"Loading annotations from: {args.input_json}")
    annotations_data = load_coco_annotations(args.input_json)
    
    print("Counting instances per class...")
    class_counts = count_instances_per_class(annotations_data)
    
    # Create visualization
    print("Creating visualization...")
    create_visualization(class_counts, args.output)
    
    return 0


if __name__ == '__main__':
    exit(main())