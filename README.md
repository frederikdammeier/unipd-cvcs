# unipd-cvcs

Exam project on object detection for the UniPD CVCS course.

## Overview

This repository contains code and results for evaluating various object detection models on the COCO 2017 validation dataset. It includes inference scripts, evaluation tools, and analysis of model performance across different metrics and size categories. Specifically, the evaluation scripts contain size-based analysis on seven custom size bins.

## Repository Structure

### Core Scripts
- `coco_inference.py`: Inference script for running object detection models (Faster R-CNN, RetinaNet, FCOS) on COCO dataset using PyTorch.
- `coco_dn_inference.py`: Inference script specifically for Cascade DETR models.

### Evaluation (`eval/`)
- `eval_coco_size.py`: Evaluates predictions with size-based metrics.
- `eval_finegrained_size.py`: Fine-grained size analysis.
- `eval_longest_edge.py`, `eval_shortest_edge.py`: Edge-based evaluations.
- `analyze_coco_traffic.py`: Traffic analysis on COCO data.
- `analyze_size_distribution.py`: Size distribution analysis.
- `visualize_*.py`: Various visualization scripts for predictions, bounding boxes, and metrics.

### Results (`results/`)
Contains evaluation metrics in JSON format for different models:
- Cascade DETR (cascadedetr_*.json)
- Faster R-CNN (fasterrcnn_*.json)
- FCOS (fcos_*.json)
- RetinaNet (retinanet_*.json)

### Inference Results (`inference_results/`)
Raw prediction files in COCO format for various models on COCO val2017.

### Job Files (`jobfiles/`)
SLURM job scripts for training and validation on GPU clusters.

### Dependencies
- `requirements.txt`: Python dependencies including PyTorch, torchvision, pycocotools, FiftyOne, etc.

## Usage

The following is only valid for the regular torchvison models:
1. Create a conda environment based on the requirements.txt
2. Run inference: `python coco_inference.py --model fasterrcnn_resnet50_fpn --data_path /path/to/coco`
3. Evaluate results: `python eval/eval_coco_size.py --pred predictions.json`

To perform analysis on Cascade-DETR, first install the source repository following its own instructions and use the python environment provided therein: https://github.com/SysCV/cascade-detr

## Models Evaluated
- Faster R-CNN (ResNet-50 FPN variants)
- RetinaNet (ResNet-50 FPN variants)
- FCOS (ResNet-50 FPN)
- Cascade DETR (with different thresholds)
