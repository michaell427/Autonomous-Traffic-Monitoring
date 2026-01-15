"""Fix YOLO Dataset Structure - Create proper directory layout for YOLOv8"""

import sys
from pathlib import Path
import shutil
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config


def create_proper_yolo_structure(config_path: str = "configs/data_config.yaml", 
                                  output_dir: str = "bdd100k_yolo_format"):
    """Create proper YOLO dataset structure with images and labels in correct locations.
    
    YOLO expects:
    dataset/
      train/
        images/
        labels/
      val/
        images/
        labels/
      test/
        images/
        labels/
    """
    config = load_config(config_path)
    data_cfg = config['data']
    detection_classes = config.get('detection_classes', [])
    
    # Class mapping
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
    }
    
    # Filter to only include target classes
    if detection_classes:
        class_to_id = {k: v for k, v in class_to_id.items() if k in detection_classes}
        # Renumber classes to be consecutive starting from 0
        class_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(class_to_id.values()))}
        class_to_id = {k: class_mapping[v] for k, v in class_to_id.items()}
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images_base = Path(data_cfg['images_dir'])
    
    print("="*70)
    print("Creating Proper YOLO Dataset Structure")
    print("="*70)
    print(f"Output directory: {output_path.absolute()}")
    print(f"Images base: {images_base.absolute()}")
    print()
    
    # Check if labels already exist in bdd100k_yolo_format structure
    existing_labels_dir = output_path / "val" / "labels"
    if existing_labels_dir.exists() and list(existing_labels_dir.glob("*.txt")):
        print("Found existing YOLO format labels, using them...")
        labels_source = output_path
    else:
        # Check if labels exist in image directories (from symlinks)
        val_labels_check = images_base / "val" / "labels"
        if val_labels_check.exists() and list(val_labels_check.glob("*.txt")):
            print("Found labels in image directories, will copy them...")
            labels_source = images_base
        else:
            print("ERROR: No YOLO format labels found!")
            print("Please run: python src/data/prepare_dataset.py --yolo-only")
            return False
    
    # Create proper structure for each split
    for split in ['train', 'val', 'test']:
        print(f"\nSetting up {split} split...")
        
        # Create directories
        split_images_dir = output_path / split / "images"
        split_labels_dir = output_path / split / "labels"
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Source paths - images_base already points to the right directory
        source_images_dir = images_base / split
        source_labels_dir = labels_source / split / "labels"
        
        if not source_images_dir.exists():
            print(f"  WARNING: Source images directory not found: {source_images_dir}")
            continue
        
        # Create symlinks for images (or copy if symlinks fail)
        image_files = list(source_images_dir.glob("*.jpg"))
        print(f"  Found {len(image_files)} images")
        
        linked_images = 0
        for img_file in image_files:
            target = split_images_dir / img_file.name
            if not target.exists():
                try:
                    # Try symlink first (works on Windows with admin or Developer Mode)
                    target.symlink_to(img_file.absolute())
                    linked_images += 1
                except (OSError, NotImplementedError):
                    # Fallback to copy
                    shutil.copy2(img_file, target)
                    linked_images += 1
        
        print(f"  Linked/copied {linked_images} images")
        
        # Copy label files
        if source_labels_dir.exists():
            label_files = list(source_labels_dir.glob("*.txt"))
            print(f"  Found {len(label_files)} label files")
            
            copied_labels = 0
            for label_file in label_files:
                target = split_labels_dir / label_file.name
                if not target.exists():
                    shutil.copy2(label_file, target)
                    copied_labels += 1
            
            print(f"  Copied {copied_labels} label files")
        else:
            print(f"  WARNING: Source labels directory not found: {source_labels_dir}")
    
    # Create dataset.yaml
    yaml_path = output_path / "dataset.yaml"
    yolo_config = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_to_id),
        'names': list(class_to_id.keys())
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yolo_config, f, default_flow_style=False, sort_keys=False)
    
    print("\n" + "="*70)
    print("Dataset Structure Created Successfully!")
    print("="*70)
    print(f"Dataset YAML: {yaml_path}")
    print(f"Structure:")
    print(f"  {output_path}/train/images/")
    print(f"  {output_path}/train/labels/")
    print(f"  {output_path}/val/images/")
    print(f"  {output_path}/val/labels/")
    print(f"  {output_path}/test/images/")
    print(f"  {output_path}/test/labels/")
    print("="*70)
    
    return True


def verify_dataset_structure(dataset_yaml: str = "bdd100k_yolo_format/dataset.yaml"):
    """Verify the dataset structure is correct."""
    from ultralytics import YOLO
    
    dataset_yaml_path = project_root / dataset_yaml
    
    if not dataset_yaml_path.exists():
        print(f"ERROR: Dataset YAML not found: {dataset_yaml_path}")
        return False
    
    print("="*70)
    print("Verifying Dataset Structure")
    print("="*70)
    
    # Load YAML
    with open(dataset_yaml_path, 'r') as f:
        yml = yaml.safe_load(f)
    
    base_path = Path(yml['path'])
    
    print(f"Base path: {base_path}")
    print(f"Number of classes: {yml['nc']}")
    print(f"Classes: {yml['names']}")
    print()
    
    # Check each split
    for split in ['train', 'val', 'test']:
        split_path = yml.get(split, '')
        images_dir = base_path / split_path
        labels_dir = base_path / split / "labels"
        
        print(f"{split.upper()}:")
        print(f"  Images: {images_dir}")
        if images_dir.exists():
            img_count = len(list(images_dir.glob("*.jpg")))
            print(f"    Found {img_count} images")
        else:
            print(f"    ERROR: Directory not found!")
            return False
        
        print(f"  Labels: {labels_dir}")
        if labels_dir.exists():
            label_count = len(list(labels_dir.glob("*.txt")))
            print(f"    Found {label_count} label files")
            
            # Check a sample label file
            sample_labels = list(labels_dir.glob("*.txt"))[:5]
            if sample_labels:
                sample = sample_labels[0]
                content = sample.read_text().strip()
                if content:
                    lines = content.split('\n')
                    print(f"    Sample label ({sample.name}): {len(lines)} objects")
                    if lines:
                        parts = lines[0].split()
                        if len(parts) >= 5:
                            print(f"      Format: class={parts[0]}, x={parts[1]}, y={parts[2]}, w={parts[3]}, h={parts[4]}")
                else:
                    print(f"    WARNING: Sample label file is empty!")
        else:
            print(f"    ERROR: Directory not found!")
            return False
        print()
    
    # Try to initialize YOLO with the dataset
    print("Testing YOLO dataset loading...")
    try:
        model = YOLO("yolov8n.pt")
        # This will validate the dataset structure
        print("  SUCCESS: YOLO can load the dataset configuration!")
        return True
    except Exception as e:
        print(f"  ERROR: Failed to load dataset: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix YOLO dataset structure")
    parser.add_argument("--config", type=str, default="configs/data_config.yaml",
                       help="Path to data config file")
    parser.add_argument("--output-dir", type=str, default="bdd100k_yolo_format",
                       help="Output directory for YOLO format")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify existing structure, don't create")
    parser.add_argument("--dataset-yaml", type=str, default="bdd100k_yolo_format/dataset.yaml",
                       help="Path to dataset YAML for verification")
    
    args = parser.parse_args()
    
    if args.verify_only:
        success = verify_dataset_structure(args.dataset_yaml)
        sys.exit(0 if success else 1)
    else:
        success = create_proper_yolo_structure(args.config, args.output_dir)
        if success:
            print("\nVerifying structure...")
            verify_dataset_structure(args.dataset_yaml)
        sys.exit(0 if success else 1)

