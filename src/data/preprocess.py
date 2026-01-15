"""Data preprocessing script for Phase 1"""

import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from src.data.dataset import BDD100KDetectionDataset, BDD100KSegmentationDataset, collate_fn
from src.data.augmentation import get_detection_augmentation, get_segmentation_augmentation
from src.utils.config import load_config
from src.data.quality_report import generate_quality_report
import json


def validate_dataset(config: dict, split: str = "train"):
    """Validate dataset and print statistics.
    
    Args:
        config: Data configuration dictionary
        split: Dataset split to validate
    """
    data_cfg = config['data']
    
    print(f"\n{'='*60}")
    print(f"Validating {split} dataset...")
    print(f"{'='*60}\n")
    
    # Validate detection dataset
    print("Detection Dataset:")
    print("-" * 40)
    det_dataset = BDD100KDetectionDataset(
        images_dir=data_cfg['images_dir'],
        labels_dir=data_cfg['det_labels_dir'],
        split=split,
        transform=None,
        target_classes=config.get('detection_classes', [])
    )
    
    if len(det_dataset) > 0:
        sample = det_dataset[0]
        print(f"  ✓ Loaded {len(det_dataset)} images")
        print(f"  ✓ Sample image shape: {sample['image'].shape}")
        print(f"  ✓ Sample has {len(sample['boxes'])} bounding boxes")
        print(f"  ✓ Classes found: {set(sample['labels'])}")
    else:
        print(f"  ✗ No images found in {split} split")
    
    # Validate segmentation dataset if paths exist
    if Path(data_cfg.get('seg_labels_dir', '')).exists():
        print("\nSegmentation Dataset:")
        print("-" * 40)
        try:
            seg_dataset = BDD100KSegmentationDataset(
                images_dir=data_cfg['images_dir'],
                masks_dir=data_cfg['seg_labels_dir'],
                split=split,
                transform=None
            )
            if len(seg_dataset) > 0:
                sample = seg_dataset[0]
                print(f"  ✓ Loaded {len(seg_dataset)} images with masks")
                print(f"  ✓ Sample image shape: {sample['image'].shape}")
                print(f"  ✓ Sample mask shape: {sample['mask'].shape}")
            else:
                print(f"  ✗ No images with masks found in {split} split")
        except Exception as e:
            print(f"  ✗ Error loading segmentation dataset: {e}")
    
    print(f"\n{'='*60}\n")


def test_dataloader(config: dict, split: str = "train"):
    """Test data loading with augmentation.
    
    Args:
        config: Data configuration dictionary
        split: Dataset split to test
    """
    data_cfg = config['data']
    aug_cfg = data_cfg.get('augmentation', {})
    
    print(f"\n{'='*60}")
    print(f"Testing DataLoader with augmentation ({split})...")
    print(f"{'='*60}\n")
    
    # Create augmentation transform
    transform = get_detection_augmentation(
        image_size=tuple(data_cfg['image_size']),
        training=(split == 'train'),
        **aug_cfg
    )
    
    # Create dataset
    dataset = BDD100KDetectionDataset(
        images_dir=data_cfg['images_dir'],
        labels_dir=data_cfg['det_labels_dir'],
        split=split,
        transform=transform,
        target_classes=config.get('detection_classes', [])
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=(split == 'train'),
        num_workers=data_cfg['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Test loading a batch
    print("Loading a batch...")
    try:
        batch = next(iter(dataloader))
        print(f"  ✓ Batch loaded successfully")
        print(f"  ✓ Batch image shape: {batch['image'].shape}")
        print(f"  ✓ Number of samples in batch: {len(batch['image_id'])}")
        print(f"  ✓ Image tensor dtype: {batch['image'].dtype}")
        print(f"  ✓ Image tensor range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
    except Exception as e:
        print(f"  ✗ Error loading batch: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}\n")


def generate_data_report(config: dict, output_path: str = "outputs/data_report.json"):
    """Generate data quality report.
    
    Args:
        config: Data configuration dictionary
        output_path: Path to save report
    """
    print(f"\n{'='*60}")
    print("Generating data quality report...")
    print(f"{'='*60}\n")
    
    report = {
        'splits': {},
        'classes': {},
        'statistics': {}
    }
    
    for split in ['train', 'val', 'test']:
        try:
            dataset = BDD100KDetectionDataset(
                images_dir=config['data']['images_dir'],
                labels_dir=config['data']['det_labels_dir'],
                split=split,
                transform=None,
                target_classes=config.get('detection_classes', [])
            )
            
            # Collect statistics
            total_boxes = 0
            class_counts = {}
            
            for i in range(min(100, len(dataset))):  # Sample first 100 for speed
                sample = dataset[i]
                total_boxes += len(sample['boxes'])
                for label in sample['labels']:
                    class_counts[label] = class_counts.get(label, 0) + 1
            
            report['splits'][split] = {
                'num_images': len(dataset),
                'total_boxes_sampled': total_boxes,
                'avg_boxes_per_image': total_boxes / min(100, len(dataset)) if dataset else 0,
                'class_distribution': class_counts
            }
            
            print(f"{split}: {len(dataset)} images, {total_boxes} boxes (sampled)")
        except Exception as e:
            print(f"Error processing {split}: {e}")
            report['splits'][split] = {'error': str(e)}
    
    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved to {output_path}")
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Preprocess BDD100K dataset")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/data_config.yaml',
        help='Path to data configuration file'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate dataset structure'
    )
    parser.add_argument(
        '--test-loader',
        action='store_true',
        help='Test data loading with augmentation'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate data quality report'
    )
    parser.add_argument(
        '--quality-report',
        action='store_true',
        help='Generate comprehensive data quality report'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all preprocessing steps'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run requested operations
    if args.all or args.validate:
        for split in ['train', 'val', 'test']:
            validate_dataset(config, split)
    
    if args.all or args.test_loader:
        test_dataloader(config, 'train')
    
    if args.all or args.report:
        generate_data_report(config)
    
    if args.all or args.quality_report:
        generate_quality_report(
            config_path=args.config,
            output_dir="outputs/reports",
            splits=['train', 'val', 'test'],
            sample_size=None  # Set to a number for quick check
        )
    
    if not any([args.validate, args.test_loader, args.report, args.quality_report, args.all]):
        print("No operation specified. Use --help for options.")
        print("Suggested: python src/data/preprocess.py --all")


if __name__ == '__main__':
    main()

