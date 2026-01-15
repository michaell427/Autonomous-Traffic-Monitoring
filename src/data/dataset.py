"""BDD100K Dataset Loader"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from src.data.validation import validate_bbox, filter_invalid_boxes


def collate_fn(batch):
    """Custom collate function for variable-length bounding boxes.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched data with padded boxes and labels
    """
    images = torch.stack([item['image'] for item in batch])
    image_ids = [item['image_id'] for item in batch]
    
    # Get max number of boxes in this batch
    max_boxes = max(len(item['boxes']) for item in batch)
    
    # Pad boxes and labels to same length
    boxes_list = []
    labels_list = []
    
    for item in batch:
        num_boxes = len(item['boxes'])
        
        if num_boxes > 0:
            # Convert boxes to tensor
            boxes = torch.tensor(item['boxes'], dtype=torch.float32)
            labels = torch.tensor(item['labels'], dtype=torch.long)
            
            # Pad with zeros if needed
            if num_boxes < max_boxes:
                padding = torch.zeros(max_boxes - num_boxes, 4, dtype=torch.float32)
                boxes = torch.cat([boxes, padding], dim=0)
                
                # Pad labels with -1 (invalid class)
                label_padding = torch.full((max_boxes - num_boxes,), -1, dtype=torch.long)
                labels = torch.cat([labels, label_padding], dim=0)
        else:
            # No boxes - create empty tensors
            boxes = torch.zeros(max_boxes, 4, dtype=torch.float32)
            labels = torch.full((max_boxes,), -1, dtype=torch.long)
        
        boxes_list.append(boxes)
        labels_list.append(labels)
    
    # Stack into batch tensors
    boxes_batch = torch.stack(boxes_list)
    labels_batch = torch.stack(labels_list)
    
    return {
        'image': images,
        'boxes': boxes_batch,
        'labels': labels_batch,
        'image_id': image_ids
    }


class BDD100KDetectionDataset(Dataset):
    """BDD100K Detection Dataset Loader.
    
    Loads images and detection labels (bounding boxes) from BDD100K dataset.
    """
    
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        split: str = "train",
        transform: Optional[callable] = None,
        target_classes: Optional[List[str]] = None
    ):
        """Initialize BDD100K Detection Dataset.
        
        Args:
            images_dir: Path to images directory (e.g., "bdd100k_images_100k/100k")
            labels_dir: Path to labels directory (e.g., "bdd100k_det_20_labels")
            split: Dataset split ("train", "val", "test")
            transform: Optional transform to apply to images
            target_classes: List of class names to filter (None = all classes)
        """
        self.images_dir = Path(images_dir) / split
        self.labels_dir = Path(labels_dir) / split
        self.split = split
        self.transform = transform
        self.target_classes = target_classes or []
        
        # BDD100K class mapping
        self.class_to_id = {
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
        self.id_to_class = {v: k for k, v in self.class_to_id.items()}
        
        # Load image and label file paths
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        self.label_files = [self.labels_dir / f"{img.stem}.json" 
                           for img in self.image_files]
        
        # Filter out images without labels
        valid_pairs = [(img, lbl) for img, lbl in zip(self.image_files, self.label_files) 
                      if lbl.exists()]
        self.image_files, self.label_files = zip(*valid_pairs) if valid_pairs else ([], [])
        
        print(f"Loaded {len(self.image_files)} {split} images from BDD100K")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get item from dataset.
        
        Returns:
            Dictionary with:
                - image: PIL Image or numpy array
                - boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
                - labels: List of class IDs
                - image_id: Image filename
        """
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image_height, image_width = image.shape[:2]
        
        # Load labels
        label_path = self.label_files[idx]
        boxes, labels = self._load_labels(label_path)
        
        # Filter invalid boxes
        boxes, labels = filter_invalid_boxes(
            boxes, labels, image_width, image_height, min_size=2
        )
        
        # Apply transform if provided
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'image_id': image_path.stem
        }
    
    def _load_labels(self, label_path: Path) -> Tuple[List, List]:
        """Load bounding box labels from JSON file.
        
        Args:
            label_path: Path to label JSON file
            
        Returns:
            Tuple of (boxes, labels) where boxes are [[x1, y1, x2, y2], ...]
            and labels are class IDs
        """
        boxes = []
        labels = []
        
        with open(label_path, 'r') as f:
            data = json.load(f)
        
        for frame in data.get('frames', []):
            for obj in frame.get('objects', []):
                category = obj.get('category', '')
                
                # Filter by target classes if specified
                if self.target_classes and category not in self.target_classes:
                    continue
                
                if category in self.class_to_id:
                    box2d = obj.get('box2d', {})
                    if box2d:
                        x1 = box2d.get('x1', 0)
                        y1 = box2d.get('y1', 0)
                        x2 = box2d.get('x2', 0)
                        y2 = box2d.get('y2', 0)
                        
                        # Add box (validation will happen in __getitem__)
                        boxes.append([x1, y1, x2, y2])
                        labels.append(self.class_to_id[category])
        
        return boxes, labels


class BDD100KSegmentationDataset(Dataset):
    """BDD100K Segmentation Dataset Loader.
    
    Loads images and segmentation masks from BDD100K dataset.
    """
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        split: str = "train",
        transform: Optional[callable] = None
    ):
        """Initialize BDD100K Segmentation Dataset.
        
        Args:
            images_dir: Path to images directory
            masks_dir: Path to segmentation masks directory
            split: Dataset split ("train", "val", "test")
            transform: Optional transform to apply to images and masks
        """
        self.images_dir = Path(images_dir) / split
        self.masks_dir = Path(masks_dir) / split
        self.split = split
        self.transform = transform
        
        # Load image file paths
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        
        # Match with mask files
        self.mask_files = [self.masks_dir / f"{img.stem}.png" 
                          for img in self.image_files]
        
        # Filter out images without masks
        valid_pairs = [(img, mask) for img, mask in zip(self.image_files, self.mask_files) 
                      if mask.exists()]
        self.image_files, self.mask_files = zip(*valid_pairs) if valid_pairs else ([], [])
        
        print(f"Loaded {len(self.image_files)} {split} images with masks from BDD100K")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get item from dataset.
        
        Returns:
            Dictionary with:
                - image: PIL Image or numpy array
                - mask: Segmentation mask
                - image_id: Image filename
        """
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Load mask
        mask_path = self.mask_files[idx]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Apply transform if provided
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return {
            'image': image,
            'mask': mask,
            'image_id': image_path.stem
        }

