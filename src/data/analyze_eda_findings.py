"""Script to analyze and summarize EDA findings"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.dataset import BDD100KDetectionDataset


def analyze_class_distribution(dataset, sample_size=1000):
    """Analyze class distribution in dataset.
    
    Args:
        dataset: BDD100KDetectionDataset instance
        sample_size: Number of samples to analyze
        
    Returns:
        Dictionary with class statistics
    """
    print("Analyzing class distribution...")
    
    class_counts = Counter()
    total_objects = 0
    
    # Sample dataset
    sample_indices = np.random.choice(
        min(len(dataset), sample_size),
        size=min(len(dataset), sample_size),
        replace=False
    )
    
    for idx in sample_indices:
        try:
            sample = dataset[idx]
            labels = sample['labels']
            for label in labels:
                class_name = dataset.id_to_class.get(label, f"unknown_{label}")
                class_counts[class_name] += 1
                total_objects += 1
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Calculate statistics
    stats = {
        'class_counts': dict(class_counts),
        'total_objects': total_objects,
        'unique_classes': len(class_counts),
        'most_common': class_counts.most_common(1)[0] if class_counts else None,
        'least_common': class_counts.most_common()[-1] if class_counts else None,
    }
    
    if stats['most_common'] and stats['least_common']:
        stats['imbalance_ratio'] = (
            stats['most_common'][1] / stats['least_common'][1]
            if stats['least_common'][1] > 0 else float('inf')
        )
    else:
        stats['imbalance_ratio'] = 0
    
    return stats


def analyze_bounding_boxes(dataset, sample_size=1000):
    """Analyze bounding box characteristics.
    
    Args:
        dataset: BDD100KDetectionDataset instance
        sample_size: Number of samples to analyze
        
    Returns:
        Dictionary with bounding box statistics
    """
    print("Analyzing bounding boxes...")
    
    widths = []
    heights = []
    areas = []
    aspect_ratios = []
    
    sample_indices = np.random.choice(
        min(len(dataset), sample_size),
        size=min(len(dataset), sample_size),
        replace=False
    )
    
    for idx in sample_indices:
        try:
            sample = dataset[idx]
            boxes = sample['boxes']
            
            for box in boxes:
                if len(box) >= 4:
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    width = x2 - x1
                    height = y2 - y1
                    
                    if width > 0 and height > 0:
                        widths.append(width)
                        heights.append(height)
                        areas.append(width * height)
                        aspect_ratios.append(width / height if height > 0 else 0)
        except Exception as e:
            continue
    
    if not widths:
        return {}
    
    stats = {
        'width': {
            'mean': np.mean(widths),
            'median': np.median(widths),
            'std': np.std(widths),
            'min': np.min(widths),
            'max': np.max(widths),
        },
        'height': {
            'mean': np.mean(heights),
            'median': np.median(heights),
            'std': np.std(heights),
            'min': np.min(heights),
            'max': np.max(heights),
        },
        'area': {
            'mean': np.mean(areas),
            'median': np.median(areas),
            'std': np.std(areas),
            'min': np.min(areas),
            'max': np.max(areas),
        },
        'aspect_ratio': {
            'mean': np.mean(aspect_ratios),
            'median': np.median(aspect_ratios),
            'std': np.std(aspect_ratios),
            'min': np.min(aspect_ratios),
            'max': np.max(aspect_ratios),
        },
        'small_objects': sum(1 for a in areas if a < 32 * 32),
        'medium_objects': sum(1 for a in areas if 32 * 32 <= a < 128 * 128),
        'large_objects': sum(1 for a in areas if a >= 128 * 128),
    }
    
    total_boxes = len(areas)
    if total_boxes > 0:
        stats['small_objects_pct'] = (stats['small_objects'] / total_boxes) * 100
        stats['medium_objects_pct'] = (stats['medium_objects'] / total_boxes) * 100
        stats['large_objects_pct'] = (stats['large_objects'] / total_boxes) * 100
    
    return stats


def analyze_objects_per_image(dataset, sample_size=1000):
    """Analyze number of objects per image.
    
    Args:
        dataset: BDD100KDetectionDataset instance
        sample_size: Number of samples to analyze
        
    Returns:
        Dictionary with objects per image statistics
    """
    print("Analyzing objects per image...")
    
    objects_per_image = []
    
    sample_indices = np.random.choice(
        min(len(dataset), sample_size),
        size=min(len(dataset), sample_size),
        replace=False
    )
    
    for idx in sample_indices:
        try:
            sample = dataset[idx]
            num_objects = len(sample['boxes'])
            objects_per_image.append(num_objects)
        except Exception as e:
            continue
    
    if not objects_per_image:
        return {}
    
    stats = {
        'mean': np.mean(objects_per_image),
        'median': np.median(objects_per_image),
        'std': np.std(objects_per_image),
        'min': np.min(objects_per_image),
        'max': np.max(objects_per_image),
        'zero_objects': sum(1 for x in objects_per_image if x == 0),
        'one_to_five': sum(1 for x in objects_per_image if 1 <= x <= 5),
        'six_to_ten': sum(1 for x in objects_per_image if 6 <= x <= 10),
        'ten_plus': sum(1 for x in objects_per_image if x > 10),
    }
    
    total = len(objects_per_image)
    if total > 0:
        stats['zero_objects_pct'] = (stats['zero_objects'] / total) * 100
        stats['one_to_five_pct'] = (stats['one_to_five'] / total) * 100
        stats['six_to_ten_pct'] = (stats['six_to_ten'] / total) * 100
        stats['ten_plus_pct'] = (stats['ten_plus'] / total) * 100
    
    return stats


def analyze_image_characteristics(dataset, sample_size=100):
    """Analyze image characteristics.
    
    Args:
        dataset: BDD100KDetectionDataset instance
        sample_size: Number of samples to analyze
        
    Returns:
        Dictionary with image statistics
    """
    print("Analyzing image characteristics...")
    
    widths = []
    heights = []
    aspect_ratios = []
    
    sample_indices = np.random.choice(
        min(len(dataset), sample_size),
        size=min(len(dataset), sample_size),
        replace=False
    )
    
    for idx in sample_indices:
        try:
            sample = dataset[idx]
            img = sample['image']
            if hasattr(img, 'shape'):
                h, w = img.shape[:2]
            else:
                w, h = img.size
            
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)
        except Exception as e:
            continue
    
    if not widths:
        return {}
    
    stats = {
        'width': {
            'mean': np.mean(widths),
            'median': np.median(widths),
            'min': np.min(widths),
            'max': np.max(widths),
            'unique': len(set(widths)),
        },
        'height': {
            'mean': np.mean(heights),
            'median': np.median(heights),
            'min': np.min(heights),
            'max': np.max(heights),
            'unique': len(set(heights)),
        },
        'aspect_ratio': {
            'mean': np.mean(aspect_ratios),
            'median': np.median(aspect_ratios),
            'min': np.min(aspect_ratios),
            'max': np.max(aspect_ratios),
        },
    }
    
    return stats


def generate_summary_report(output_path="docs/eda_summary.json"):
    """Generate comprehensive EDA summary report.
    
    Args:
        output_path: Path to save summary JSON
    """
    print("="*60)
    print("Generating EDA Summary Report")
    print("="*60)
    
    # Load dataset
    dataset = BDD100KDetectionDataset(
        images_dir="bdd100k_images_100k/100k",
        labels_dir="bdd100k_det_20_labels",
        split="train",
        transform=None
    )
    
    if len(dataset) == 0:
        print("ERROR: No data loaded! Check dataset paths.")
        return
    
    print(f"\nDataset loaded: {len(dataset)} images")
    
    # Run analyses
    report = {
        'dataset_size': len(dataset),
        'class_distribution': analyze_class_distribution(dataset),
        'bounding_boxes': analyze_bounding_boxes(dataset),
        'objects_per_image': analyze_objects_per_image(dataset),
        'image_characteristics': analyze_image_characteristics(dataset),
    }
    
    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✓ Report saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if report['class_distribution']:
        cd = report['class_distribution']
        print(f"\nClass Distribution:")
        print(f"  Total objects: {cd.get('total_objects', 0)}")
        print(f"  Unique classes: {cd.get('unique_classes', 0)}")
        if cd.get('most_common'):
            print(f"  Most common: {cd['most_common'][0]} ({cd['most_common'][1]} instances)")
        if cd.get('least_common'):
            print(f"  Least common: {cd['least_common'][0]} ({cd['least_common'][1]} instances)")
        print(f"  Imbalance ratio: {cd.get('imbalance_ratio', 0):.2f}")
    
    if report['objects_per_image']:
        opi = report['objects_per_image']
        print(f"\nObjects per Image:")
        print(f"  Mean: {opi.get('mean', 0):.2f}")
        print(f"  Median: {opi.get('median', 0):.2f}")
        print(f"  Range: {opi.get('min', 0)} - {opi.get('max', 0)}")
        print(f"  Images with 0 objects: {opi.get('zero_objects_pct', 0):.1f}%")
    
    if report['bounding_boxes']:
        bb = report['bounding_boxes']
        print(f"\nBounding Boxes:")
        if 'width' in bb:
            print(f"  Average size: {bb['width']['mean']:.1f} x {bb['height']['mean']:.1f}")
            print(f"  Small objects: {bb.get('small_objects_pct', 0):.1f}%")
            print(f"  Large objects: {bb.get('large_objects_pct', 0):.1f}%")
    
    if report['image_characteristics']:
        ic = report['image_characteristics']
        print(f"\nImage Characteristics:")
        if 'width' in ic:
            print(f"  Common size: {ic['width']['mean']:.0f} x {ic['height']['mean']:.0f}")
            print(f"  Unique sizes: {ic['width'].get('unique', 0)}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    generate_summary_report()

