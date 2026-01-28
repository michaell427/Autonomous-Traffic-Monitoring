"""Test augmentation pipeline to ensure it preserves label validity"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import BDD100KDetectionDataset
from src.data.augmentation import get_detection_augmentation
from src.utils.config import load_config
from tqdm import tqdm


def test_augmentation_validity(num_samples=100):
    """Test that augmentation preserves label validity.
    
    Args:
        num_samples: Number of samples to test
    """
    config = load_config('configs/data_config.yaml')
    data_cfg = config['data']
    aug_cfg = data_cfg.get('augmentation', {})
    
    # Create dataset
    dataset = BDD100KDetectionDataset(
        images_dir=data_cfg['images_dir'],
        labels_dir=data_cfg['det_labels_dir'],
        split='train',
        transform=None,
        target_classes=config.get('detection_classes', [])
    )
    
    # Create augmentation transform (filter out 'enabled' flag)
    aug_params = {k: v for k, v in aug_cfg.items() if k != 'enabled'}
    transform = get_detection_augmentation(
        image_size=tuple(data_cfg['image_size']),
        training=True,
        **aug_params
    )
    
    print("Testing augmentation validity...")
    print("=" * 60)
    
    errors = []
    valid_count = 0
    
    num_samples = min(num_samples, len(dataset))
    
    for i in tqdm(range(num_samples), desc="Testing samples"):
        sample = dataset[i]
        
        img = sample['image']
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
        
        original_boxes = len(sample['boxes'])
        original_labels = len(sample['labels'])
        
        try:
            transformed = transform(
                image=img.copy(),
                bboxes=sample['boxes'].copy(),
                class_labels=sample['labels'].copy()
            )
            
            augmented_boxes = len(transformed['bboxes'])
            augmented_labels = len(transformed['class_labels'])
            
            # Validate
            if augmented_boxes != augmented_labels:
                errors.append({
                    'idx': i,
                    'error': f"Box/Label mismatch: {augmented_boxes} boxes, {augmented_labels} labels"
                })
                continue
            
            # Check boxes are valid
            for box in transformed['bboxes']:
                if len(box) != 4:
                    errors.append({
                        'idx': i,
                        'error': f"Invalid box format: {box}"
                    })
                    break
                x1, y1, x2, y2 = box
                if x2 <= x1 or y2 <= y1:
                    errors.append({
                        'idx': i,
                        'error': f"Invalid box coordinates: {box}"
                    })
                    break
            else:
                valid_count += 1
                
        except Exception as e:
            errors.append({
                'idx': i,
                'error': f"Exception: {str(e)}"
            })
    
    print(f"\nResults:")
    print(f"  Total samples tested: {num_samples}")
    print(f"  Valid: {valid_count}")
    print(f"  Errors: {len(errors)}")
    
    if errors:
        print(f"\n  First 5 errors:")
        for err in errors[:5]:
            print(f"    Sample {err['idx']}: {err['error']}")
        return False
    else:
        print("\n[SUCCESS] All augmentations valid!")
        return True


if __name__ == "__main__":
    success = test_augmentation_validity(num_samples=100)
    sys.exit(0 if success else 1)

