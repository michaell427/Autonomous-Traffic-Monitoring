"""Data validation utilities for BDD100K dataset"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from pathlib import Path
import json


def validate_bbox(bbox: List[float], image_width: int, image_height: int, 
                  min_size: int = 2) -> bool:
    """Validate a bounding box.
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2]
        image_width: Image width
        image_height: Image height
        min_size: Minimum width/height in pixels
        
    Returns:
        True if valid, False otherwise
    """
    if len(bbox) != 4:
        return False
    
    x1, y1, x2, y2 = bbox
    
    # Check for NaN or inf values
    if any(not np.isfinite(val) for val in bbox):
        return False
    
    # Check coordinates are in valid range
    if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
        return False
    
    # Check box has positive area
    width = x2 - x1
    height = y2 - y1
    
    if width <= 0 or height <= 0:
        return False
    
    # Check minimum size
    if width < min_size or height < min_size:
        return False
    
    # Check box is not too small relative to image
    if width < image_width * 0.001 or height < image_height * 0.001:
        return False
    
    return True


def filter_invalid_boxes(boxes: List[List[float]], labels: List[int],
                        image_width: int, image_height: int,
                        min_size: int = 2) -> Tuple[List[List[float]], List[int]]:
    """Filter out invalid bounding boxes.
    
    Args:
        boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
        labels: List of class labels
        image_width: Image width
        image_height: Image height
        min_size: Minimum box size in pixels
        
    Returns:
        Tuple of (filtered_boxes, filtered_labels)
    """
    valid_boxes = []
    valid_labels = []
    
    for box, label in zip(boxes, labels):
        if validate_bbox(box, image_width, image_height, min_size):
            valid_boxes.append(box)
            valid_labels.append(label)
    
    return valid_boxes, valid_labels


def validate_label_file(label_path: Path) -> Dict:
    """Validate a label file and return statistics.
    
    Args:
        label_path: Path to label JSON file
        
    Returns:
        Dictionary with validation statistics
    """
    stats = {
        'file_exists': False,
        'valid_json': False,
        'num_frames': 0,
        'num_objects': 0,
        'num_valid_boxes': 0,
        'num_invalid_boxes': 0,
        'invalid_reasons': {
            'missing_box2d': 0,
            'invalid_coords': 0,
            'empty_box': 0,
            'too_small': 0,
            'out_of_bounds': 0
        },
        'classes': {},
        'errors': []
    }
    
    if not label_path.exists():
        stats['errors'].append(f"Label file does not exist: {label_path}")
        return stats
    
    stats['file_exists'] = True
    
    try:
        with open(label_path, 'r') as f:
            data = json.load(f)
        stats['valid_json'] = True
    except json.JSONDecodeError as e:
        stats['errors'].append(f"Invalid JSON: {e}")
        return stats
    except Exception as e:
        stats['errors'].append(f"Error reading file: {e}")
        return stats
    
    # Process frames and objects
    frames = data.get('frames', [])
    stats['num_frames'] = len(frames)
    
    # Get image dimensions from first frame if available
    image_width = 1280  # Default BDD100K width
    image_height = 720  # Default BDD100K height
    
    if frames:
        first_frame = frames[0]
        if 'attributes' in first_frame:
            attrs = first_frame['attributes']
            if 'resolution' in attrs:
                res = attrs['resolution']
                image_width = res.get('width', image_width)
                image_height = res.get('height', image_height)
    
    for frame in frames:
        objects = frame.get('objects', [])
        stats['num_objects'] += len(objects)
        
        for obj in objects:
            category = obj.get('category', 'unknown')
            
            # Count classes
            if category not in stats['classes']:
                stats['classes'][category] = 0
            stats['classes'][category] += 1
            
            # Validate box2d
            box2d = obj.get('box2d', {})
            if not box2d:
                stats['invalid_reasons']['missing_box2d'] += 1
                stats['num_invalid_boxes'] += 1
                continue
            
            x1 = box2d.get('x1', 0)
            y1 = box2d.get('y1', 0)
            x2 = box2d.get('x2', 0)
            y2 = box2d.get('y2', 0)
            
            # Check for invalid coordinates
            if not all(isinstance(v, (int, float)) and np.isfinite(v) 
                      for v in [x1, y1, x2, y2]):
                stats['invalid_reasons']['invalid_coords'] += 1
                stats['num_invalid_boxes'] += 1
                continue
            
            # Check box validity
            bbox = [x1, y1, x2, y2]
            if validate_bbox(bbox, image_width, image_height):
                stats['num_valid_boxes'] += 1
            else:
                stats['num_invalid_boxes'] += 1
                
                # Determine reason
                width = x2 - x1
                height = y2 - y1
                
                if width <= 0 or height <= 0:
                    stats['invalid_reasons']['empty_box'] += 1
                elif width < 2 or height < 2:
                    stats['invalid_reasons']['too_small'] += 1
                elif x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
                    stats['invalid_reasons']['out_of_bounds'] += 1
                else:
                    stats['invalid_reasons']['invalid_coords'] += 1
    
    return stats


def validate_dataset_directory(labels_dir: Path, split: str = "train",
                               sample_size: Optional[int] = None) -> Dict:
    """Validate all label files in a directory.
    
    Args:
        labels_dir: Path to labels directory
        split: Dataset split
        sample_size: If provided, only validate this many files (for quick check)
        
    Returns:
        Dictionary with aggregate validation statistics
    """
    split_dir = labels_dir / split
    label_files = sorted(list(split_dir.glob("*.json")))
    
    if sample_size:
        label_files = label_files[:sample_size]
    
    total_stats = {
        'total_files': len(label_files),
        'files_processed': 0,
        'files_with_errors': 0,
        'total_frames': 0,
        'total_objects': 0,
        'total_valid_boxes': 0,
        'total_invalid_boxes': 0,
        'invalid_reasons': {
            'missing_box2d': 0,
            'invalid_coords': 0,
            'empty_box': 0,
            'too_small': 0,
            'out_of_bounds': 0
        },
        'classes': {},
        'error_files': []
    }
    
    for label_file in label_files:
        stats = validate_label_file(label_file)
        
        total_stats['files_processed'] += 1
        
        if stats['errors']:
            total_stats['files_with_errors'] += 1
            total_stats['error_files'].append(str(label_file))
        
        total_stats['total_frames'] += stats['num_frames']
        total_stats['total_objects'] += stats['num_objects']
        total_stats['total_valid_boxes'] += stats['num_valid_boxes']
        total_stats['total_invalid_boxes'] += stats['num_invalid_boxes']
        
        for reason, count in stats['invalid_reasons'].items():
            total_stats['invalid_reasons'][reason] += count
        
        for class_name, count in stats['classes'].items():
            if class_name not in total_stats['classes']:
                total_stats['classes'][class_name] = 0
            total_stats['classes'][class_name] += count
    
    return total_stats

