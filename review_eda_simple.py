"""Simple EDA review script that works without PyTorch - reads from JSON files directly"""

import json
from pathlib import Path
from collections import Counter
import numpy as np

def load_labels_from_json(label_path):
    """Load detection labels from JSON file."""
    with open(label_path, 'r') as f:
        data = json.load(f)
    
    objects = []
    attributes = data.get('attributes', {})
    
    for frame in data.get('frames', []):
        for obj in frame.get('objects', []):
            category = obj.get('category', '')
            box2d = obj.get('box2d', {})
            if box2d:
                objects.append({
                    'category': category,
                    'x1': box2d.get('x1', 0),
                    'y1': box2d.get('y1', 0),
                    'x2': box2d.get('x2', 0),
                    'y2': box2d.get('y2', 0)
                })
    
    return objects, attributes

def analyze_dataset_split(images_dir, labels_dir, split="train", max_samples=1000):
    """Analyze a dataset split."""
    images_path = Path(images_dir) / split
    labels_path = Path(labels_dir) / split
    
    if not images_path.exists() or not labels_path.exists():
        print(f"  ✗ Paths don't exist for {split}")
        return None
    
    # Get image files
    image_files = list(images_path.glob("*.jpg"))
    print(f"  Found {len(image_files)} images")
    
    # Get label files
    label_files = list(labels_path.glob("*.json"))
    print(f"  Found {len(label_files)} label files")
    
    # Sample for analysis
    sample_size = min(max_samples, len(label_files))
    sample_indices = np.random.choice(len(label_files), size=sample_size, replace=False)
    
    # Analyze
    class_counts = Counter()
    all_boxes = []
    objects_per_image = []
    attributes_list = []
    
    print(f"  Analyzing {sample_size} samples...")
    for idx in sample_indices:
        label_file = label_files[idx]
        try:
            objects, attrs = load_labels_from_json(label_file)
            
            # Count classes
            for obj in objects:
                class_counts[obj['category']] += 1
            
            # Collect boxes
            for obj in objects:
                width = obj['x2'] - obj['x1']
                height = obj['y2'] - obj['y1']
                if width > 0 and height > 0:
                    all_boxes.append({
                        'width': width,
                        'height': height,
                        'area': width * height,
                        'aspect_ratio': width / height if height > 0 else 0
                    })
            
            objects_per_image.append(len(objects))
            if attrs:
                attributes_list.append(attrs)
        except Exception as e:
            continue
    
    # Calculate statistics
    stats = {
        'total_images': len(image_files),
        'total_labels': len(label_files),
        'samples_analyzed': sample_size,
        'class_distribution': dict(class_counts),
        'total_objects': sum(class_counts.values()),
        'objects_per_image': {
            'mean': np.mean(objects_per_image) if objects_per_image else 0,
            'median': np.median(objects_per_image) if objects_per_image else 0,
            'min': np.min(objects_per_image) if objects_per_image else 0,
            'max': np.max(objects_per_image) if objects_per_image else 0,
        },
        'bounding_boxes': {},
        'attributes': {}
    }
    
    # Bounding box stats
    if all_boxes:
        stats['bounding_boxes'] = {
            'width': {
                'mean': np.mean([b['width'] for b in all_boxes]),
                'median': np.median([b['width'] for b in all_boxes]),
                'min': np.min([b['width'] for b in all_boxes]),
                'max': np.max([b['width'] for b in all_boxes]),
            },
            'height': {
                'mean': np.mean([b['height'] for b in all_boxes]),
                'median': np.median([b['height'] for b in all_boxes]),
                'min': np.min([b['height'] for b in all_boxes]),
                'max': np.max([b['height'] for b in all_boxes]),
            },
            'area': {
                'mean': np.mean([b['area'] for b in all_boxes]),
                'median': np.median([b['area'] for b in all_boxes]),
            },
            'small_objects': sum(1 for b in all_boxes if b['area'] < 32 * 32),
            'medium_objects': sum(1 for b in all_boxes if 32 * 32 <= b['area'] < 128 * 128),
            'large_objects': sum(1 for b in all_boxes if b['area'] >= 128 * 128),
        }
        
        total_boxes = len(all_boxes)
        if total_boxes > 0:
            stats['bounding_boxes']['small_objects_pct'] = (stats['bounding_boxes']['small_objects'] / total_boxes) * 100
            stats['bounding_boxes']['medium_objects_pct'] = (stats['bounding_boxes']['medium_objects'] / total_boxes) * 100
            stats['bounding_boxes']['large_objects_pct'] = (stats['bounding_boxes']['large_objects'] / total_boxes) * 100
    
    # Attributes stats
    if attributes_list:
        weather_counts = Counter([a.get('weather', 'unknown') for a in attributes_list])
        scene_counts = Counter([a.get('scene', 'unknown') for a in attributes_list])
        time_counts = Counter([a.get('timeofday', 'unknown') for a in attributes_list])
        
        stats['attributes'] = {
            'weather': dict(weather_counts),
            'scene': dict(scene_counts),
            'timeofday': dict(time_counts),
        }
    
    return stats

def main():
    print("\n" + "="*60)
    print("EDA Findings Review - Step 1.1")
    print("="*60)
    
    # Dataset paths
    images_dir = "bdd100k_images_100k/100k"
    labels_dir = "bdd100k_det_20_labels"
    
    print(f"\nAnalyzing dataset:")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")
    
    # Analyze each split
    all_stats = {}
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*60}")
        print(f"Analyzing {split.upper()} split...")
        print(f"{'='*60}")
        stats = analyze_dataset_split(images_dir, labels_dir, split, max_samples=1000)
        if stats:
            all_stats[split] = stats
    
    # Save report
    output_path = Path("docs/eda_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("SUMMARY REPORT")
    print(f"{'='*60}")
    
    # Print summary for train split
    if 'train' in all_stats:
        train_stats = all_stats['train']
        
    print(f"\nDataset Size:")
    print(f"  Total images: {train_stats['total_images']}")
    print(f"  Images with labels: {train_stats['total_labels']}")
    print(f"  Samples analyzed: {train_stats['samples_analyzed']}")
        
        print(f"\nClass Distribution:")
        class_dist = train_stats['class_distribution']
        if class_dist:
            sorted_classes = sorted(class_dist.items(), key=lambda x: x[1], reverse=True)
            print(f"  Total objects: {train_stats['total_objects']}")
            print(f"  Unique classes: {len(class_dist)}")
            print(f"\n  Top classes:")
            for class_name, count in sorted_classes[:10]:
                pct = (count / train_stats['total_objects']) * 100
                print(f"    {class_name:20s}: {count:6d} ({pct:5.1f}%)")
            
            if len(sorted_classes) > 1:
                most_common = sorted_classes[0][1]
                least_common = sorted_classes[-1][1]
                if least_common > 0:
                    imbalance = most_common / least_common
                    print(f"\n  Imbalance ratio: {imbalance:.1f}:1")
                    if imbalance > 10:
                        print(f"  ⚠️  WARNING: Severe class imbalance detected!")
        
        print(f"\nObjects per Image:")
        opi = train_stats['objects_per_image']
        print(f"  Mean: {opi['mean']:.2f}")
        print(f"  Median: {opi['median']:.2f}")
        print(f"  Range: {opi['min']} - {opi['max']}")
        
        print(f"\nBounding Box Statistics:")
        if train_stats['bounding_boxes']:
            bb = train_stats['bounding_boxes']
            print(f"  Average size: {bb['width']['mean']:.1f} x {bb['height']['mean']:.1f} pixels")
            print(f"  Size range: {bb['width']['min']:.0f}-{bb['width']['max']:.0f} x {bb['height']['min']:.0f}-{bb['height']['max']:.0f}")
            print(f"  Small objects (<32x32): {bb.get('small_objects_pct', 0):.1f}%")
            print(f"  Medium objects: {bb.get('medium_objects_pct', 0):.1f}%")
            print(f"  Large objects (>128x128): {bb.get('large_objects_pct', 0):.1f}%")
        
        print(f"\nDataset Attributes:")
        if train_stats['attributes']:
            attrs = train_stats['attributes']
            if 'weather' in attrs:
                print(f"  Weather: {list(attrs['weather'].keys())[:5]}")
            if 'scene' in attrs:
                print(f"  Scenes: {list(attrs['scene'].keys())[:5]}")
            if 'timeofday' in attrs:
                print(f"  Time of day: {list(attrs['timeofday'].keys())}")
    
    print(f"\n{'='*60}")
    print("Next Steps:")
    print(f"{'='*60}")
    print("1. Review the summary above")
    print("2. Check docs/eda_summary.json for detailed statistics")
    print("3. Fill in docs/eda_findings.md with your findings")
    print("4. Make decisions based on the analysis")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

