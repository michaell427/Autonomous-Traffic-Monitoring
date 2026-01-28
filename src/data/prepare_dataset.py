"""Dataset preparation script for training

This script:
1. Generates comprehensive dataset statistics for all splits
2. Optionally converts BDD100K format to YOLO format
3. Creates dataset configuration files
"""

import sys
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import BDD100KDetectionDataset
from src.utils.config import load_config


def generate_dataset_statistics(config: dict, output_file: str = "outputs/dataset_statistics.json"):
    """Generate comprehensive statistics for all dataset splits.
    
    Args:
        config: Data configuration dictionary
        output_file: Path to save statistics JSON file
    """
    data_cfg = config['data']
    detection_classes = config.get('detection_classes', [])
    
    stats = {
        'splits': {},
        'overall': {
            'total_images': 0,
            'total_boxes': 0,
            'class_distribution': defaultdict(int),
            'image_sizes': [],
            'boxes_per_image': []
        }
    }
    
    print("\n" + "="*70)
    print("Generating Dataset Statistics")
    print("="*70 + "\n")
    
    for split in ['train', 'val', 'test']:
        print(f"Processing {split} split...")
        
        try:
            dataset = BDD100KDetectionDataset(
                images_dir=data_cfg['images_dir'],
                labels_dir=data_cfg['det_labels_dir'],
                split=split,
                transform=None,
                target_classes=detection_classes
            )
            
            split_stats = {
                'num_images': len(dataset),
                'num_boxes': 0,
                'class_distribution': defaultdict(int),
                'images_with_boxes': 0,
                'images_without_boxes': 0,
                'avg_boxes_per_image': 0,
                'min_boxes': float('inf'),
                'max_boxes': 0,
                'image_sizes': [],
                'bbox_sizes': [],
                'bbox_aspect_ratios': []
            }
            
            if len(dataset) > 0:
                for idx in tqdm(range(len(dataset)), desc=f"  Analyzing {split}"):
                    sample = dataset[idx]
                    num_boxes = len(sample['boxes'])
                    
                    split_stats['num_boxes'] += num_boxes
                    stats['overall']['total_boxes'] += num_boxes
                    
                    if num_boxes > 0:
                        split_stats['images_with_boxes'] += 1
                        split_stats['min_boxes'] = min(split_stats['min_boxes'], num_boxes)
                        split_stats['max_boxes'] = max(split_stats['max_boxes'], num_boxes)
                        
                        # Image size
                        img = sample['image']
                        if isinstance(img, np.ndarray):
                            h, w = img.shape[:2]
                        else:
                            h, w = img.size[1], img.size[0]
                        split_stats['image_sizes'].append((w, h))
                        stats['overall']['image_sizes'].append((w, h))
                        
                        # Box statistics
                        for box, label in zip(sample['boxes'], sample['labels']):
                            if label >= 0:
                                split_stats['class_distribution'][label] += 1
                                stats['overall']['class_distribution'][label] += 1
                                
                                x1, y1, x2, y2 = box
                                box_w = x2 - x1
                                box_h = y2 - y1
                                area = box_w * box_h
                                
                                split_stats['bbox_sizes'].append(area)
                                if box_h > 0:
                                    aspect_ratio = box_w / box_h
                                    split_stats['bbox_aspect_ratios'].append(aspect_ratio)
                    else:
                        split_stats['images_without_boxes'] += 1
                    
                    split_stats['boxes_per_image'] = num_boxes
                    stats['overall']['boxes_per_image'].append(num_boxes)
                
                # Calculate averages
                if split_stats['images_with_boxes'] > 0:
                    split_stats['avg_boxes_per_image'] = (
                        split_stats['num_boxes'] / split_stats['images_with_boxes']
                    )
                
                # Convert defaultdicts to regular dicts for JSON serialization
                split_stats['class_distribution'] = dict(split_stats['class_distribution'])
                
                # Calculate bbox statistics
                if split_stats['bbox_sizes']:
                    split_stats['bbox_size_stats'] = {
                        'mean': float(np.mean(split_stats['bbox_sizes'])),
                        'std': float(np.std(split_stats['bbox_sizes'])),
                        'min': float(np.min(split_stats['bbox_sizes'])),
                        'max': float(np.max(split_stats['bbox_sizes'])),
                        'median': float(np.median(split_stats['bbox_sizes']))
                    }
                
                if split_stats['bbox_aspect_ratios']:
                    split_stats['bbox_aspect_ratio_stats'] = {
                        'mean': float(np.mean(split_stats['bbox_aspect_ratios'])),
                        'std': float(np.std(split_stats['bbox_aspect_ratios'])),
                        'min': float(np.min(split_stats['bbox_aspect_ratios'])),
                        'max': float(np.max(split_stats['bbox_aspect_ratios'])),
                        'median': float(np.median(split_stats['bbox_aspect_ratios']))
                    }
                
                # Image size statistics
                if split_stats['image_sizes']:
                    widths = [w for w, h in split_stats['image_sizes']]
                    heights = [h for w, h in split_stats['image_sizes']]
                    split_stats['image_size_stats'] = {
                        'width': {
                            'mean': float(np.mean(widths)),
                            'std': float(np.std(widths)),
                            'min': int(np.min(widths)),
                            'max': int(np.max(widths))
                        },
                        'height': {
                            'mean': float(np.mean(heights)),
                            'std': float(np.std(heights)),
                            'min': int(np.min(heights)),
                            'max': int(np.max(heights))
                        }
                    }
                
                if split_stats['min_boxes'] == float('inf'):
                    split_stats['min_boxes'] = 0
            else:
                print(f"  WARNING: No images found in {split} split")
            
            stats['splits'][split] = split_stats
            stats['overall']['total_images'] += split_stats['num_images']
            
            print(f"  {split}: {split_stats['num_images']} images, {split_stats['num_boxes']} boxes")
            
        except Exception as e:
            print(f"  ERROR processing {split}: {e}")
            import traceback
            traceback.print_exc()
    
    # Overall statistics
    if stats['overall']['total_images'] > 0:
        stats['overall']['avg_boxes_per_image'] = (
            stats['overall']['total_boxes'] / stats['overall']['total_images']
        )
    
    # Convert defaultdict to dict
    stats['overall']['class_distribution'] = dict(stats['overall']['class_distribution'])
    
    # Overall box statistics
    if stats['overall']['boxes_per_image']:
        stats['overall']['boxes_per_image_stats'] = {
            'mean': float(np.mean(stats['overall']['boxes_per_image'])),
            'std': float(np.std(stats['overall']['boxes_per_image'])),
            'min': int(np.min(stats['overall']['boxes_per_image'])),
            'max': int(np.max(stats['overall']['boxes_per_image'])),
            'median': float(np.median(stats['overall']['boxes_per_image']))
        }
    
    # Save statistics
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Statistics saved to: {output_path}")
    print(f"{'='*70}\n")
    
    # Print summary
    print("Dataset Summary:")
    print(f"  Total images: {stats['overall']['total_images']}")
    print(f"  Total boxes: {stats['overall']['total_boxes']}")
    print(f"  Average boxes per image: {stats['overall'].get('avg_boxes_per_image', 0):.2f}")
    print("\nClass Distribution:")
    class_names = {
        0: "car", 1: "truck", 2: "bus", 3: "motorcycle",
        4: "bike", 5: "person", 6: "rider", 7: "traffic light",
        8: "traffic sign", 9: "train"
    }
    for class_id, count in sorted(stats['overall']['class_distribution'].items()):
        class_name = class_names.get(class_id, f"Class {class_id}")
        percentage = (count / stats['overall']['total_boxes'] * 100) if stats['overall']['total_boxes'] > 0 else 0
        print(f"    {class_name}: {count} ({percentage:.1f}%)")
    
    return stats


def convert_to_yolo_format(config: dict, output_dir: str = "bdd100k_yolo_format"):
    """Convert BDD100K format to YOLO format.
    
    YOLO format: one .txt file per image with lines:
    class_id center_x center_y width height
    (all normalized to [0, 1])
    
    Args:
        config: Data configuration dictionary
        output_dir: Directory to save YOLO format labels
    """
    data_cfg = config['data']
    detection_classes = config.get('detection_classes', [])
    
    # Class mapping (BDD100K class names to YOLO class IDs)
    class_to_id = {
        "car": 0,
        "truck": 1,
        "bus": 2,
        "motorcycle": 3,
        "bike": 4,
        "person": 5,
        "rider": 6,
        "traffic light": 7,
        "traffic sign": 8,
        "train": 9
    }
    
    # Filter to only include target classes
    if detection_classes:
        class_to_id = {k: v for k, v in class_to_id.items() if k in detection_classes}
        # Renumber classes to be consecutive starting from 0
        class_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(class_to_id.values()))}
        class_to_id = {k: class_mapping[v] for k, v in class_to_id.items()}
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("Converting to YOLO Format")
    print("="*70 + "\n")
    
    for split in ['train', 'val', 'test']:
        print(f"Converting {split} split...")
        
        dataset = BDD100KDetectionDataset(
            images_dir=data_cfg['images_dir'],
            labels_dir=data_cfg['det_labels_dir'],
            split=split,
            transform=None,
            target_classes=detection_classes
        )
        
        split_output_dir = output_path / split / "labels"
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        converted = 0
        skipped = 0
        
        for idx in tqdm(range(len(dataset)), desc=f"  Converting {split}"):
            sample = dataset[idx]
            image_id = sample['image_id']
            boxes = sample['boxes']
            labels = sample['labels']
            
            # Get image dimensions
            img = sample['image']
            if isinstance(img, np.ndarray):
                img_h, img_w = img.shape[:2]
            else:
                img_w, img_h = img.size
            
            # Create YOLO format label file
            label_file = split_output_dir / f"{image_id}.txt"
            
            with open(label_file, 'w') as f:
                for box, label in zip(boxes, labels):
                    if label < 0:
                        continue
                    
                    x1, y1, x2, y2 = box
                    
                    # Convert to YOLO format (normalized center_x, center_y, width, height)
                    center_x = ((x1 + x2) / 2.0) / img_w
                    center_y = ((y1 + y2) / 2.0) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    
                    # Clip to [0, 1]
                    center_x = max(0, min(1, center_x))
                    center_y = max(0, min(1, center_y))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))
                    
                    # Get YOLO class ID
                    # Need to map from BDD100K class ID to YOLO class ID
                    # This assumes labels are already in the correct range
                    yolo_class_id = label
                    
                    f.write(f"{yolo_class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
            if len(boxes) > 0:
                converted += 1
            else:
                skipped += 1
        
        print(f"  {split}: Converted {converted} images, skipped {skipped} (no boxes)")
    
    # Create dataset.yaml file for YOLO
    yolo_config = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_to_id),
        'names': list(class_to_id.keys())
    }
    
    yaml_file = output_path / "dataset.yaml"
    import yaml
    with open(yaml_file, 'w') as f:
        yaml.dump(yolo_config, f, default_flow_style=False)
    
    print(f"\nYOLO format conversion complete!")
    print(f"Output directory: {output_path}")
    print(f"Dataset config: {yaml_file}")
    print(f"Note: You'll need to create symlinks or copy images to train/val/test/images directories")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument("--config", type=str, default="configs/data_config.yaml",
                       help="Path to data config file")
    parser.add_argument("--stats-only", action="store_true",
                       help="Only generate statistics, don't convert to YOLO format")
    parser.add_argument("--yolo-only", action="store_true",
                       help="Only convert to YOLO format, don't generate statistics")
    parser.add_argument("--output-dir", type=str, default="bdd100k_yolo_format",
                       help="Output directory for YOLO format conversion")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if not args.yolo_only:
        generate_dataset_statistics(config)
    
    if not args.stats_only:
        convert_to_yolo_format(config, args.output_dir)

