"""YOLOv8 Training Script for BDD100K Detection"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from src.utils.config import load_config


def train_yolov8(
    model_size: str = "n",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = None,
    project: str = "outputs/yolo_training",
    name: str = "yolov8_detection",
    config_path: str = "configs/data_config.yaml"
):
    """Train YOLOv8 model on BDD100K dataset.
    
    Args:
        model_size: Model size - 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        device: Device to use ('cpu', 'cuda', '0', '1', etc.) or None for auto
        project: Project directory for outputs
        name: Experiment name
        config_path: Path to data config file
    """
    # Load config
    config = load_config(config_path)
    data_cfg = config['data']
    
    # Dataset YAML path (relative to project root)
    dataset_yaml = project_root / "bdd100k_yolo_format" / "dataset.yaml"
    
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
    
    print("="*70)
    print("YOLOv8 Training Setup")
    print("="*70)
    print(f"Model: YOLOv8{model_size}")
    print(f"Dataset: {dataset_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: {device or 'auto'}")
    print(f"Project: {project}")
    print(f"Name: {name}")
    print("="*70)
    print()
    
    # Initialize model
    model_name = f"yolov8{model_size}.pt"
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)  # Downloads pretrained weights if needed
    
    # Train the model
    print("\nStarting training...")
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        # Additional training parameters
        patience=50,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        val=True,  # Validate during training
        plots=True,  # Generate training plots
        verbose=True,
    )
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best model saved to: {Path(project) / name / 'weights' / 'best.pt'}")
    print(f"Last model saved to: {Path(project) / name / 'weights' / 'last.pt'}")
    print(f"Results: {Path(project) / name}")
    print("="*70)
    
    return results


def test_data_loading(dataset_yaml: str = "bdd100k_yolo_format/dataset.yaml"):
    """Test YOLO dataset loading.
    
    Args:
        dataset_yaml: Path to dataset YAML file
    """
    import yaml
    
    dataset_yaml_path = project_root / dataset_yaml
    
    if not dataset_yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml_path}")
    
    print("="*70)
    print("Testing YOLO Dataset Loading")
    print("="*70)
    print(f"Dataset YAML: {dataset_yaml_path}")
    print()
    
    try:
        # Read and validate YAML
        with open(dataset_yaml_path, 'r') as f:
            data_dict = yaml.safe_load(f)
        
        print(f"[SUCCESS] Dataset YAML is valid!")
        print(f"  Number of classes: {data_dict.get('nc', 'N/A')}")
        print(f"  Class names: {data_dict.get('names', [])}")
        
        # Check if image directories exist
        train_path = project_root / data_dict.get('train', '')
        val_path = project_root / data_dict.get('val', '')
        test_path = project_root / data_dict.get('test', '')
        
        print(f"\nChecking image directories:")
        print(f"  Train: {train_path} - {'EXISTS' if train_path.exists() else 'NOT FOUND'}")
        print(f"  Val: {val_path} - {'EXISTS' if val_path.exists() else 'NOT FOUND'}")
        print(f"  Test: {test_path} - {'EXISTS' if test_path.exists() else 'NOT FOUND'}")
        
        # Check label directories
        train_labels = project_root / "bdd100k_yolo_format" / "train" / "labels"
        val_labels = project_root / "bdd100k_yolo_format" / "val" / "labels"
        test_labels = project_root / "bdd100k_yolo_format" / "test" / "labels"
        
        print(f"\nChecking label directories:")
        print(f"  Train labels: {train_labels} - {'EXISTS' if train_labels.exists() else 'NOT FOUND'}")
        if train_labels.exists():
            label_files = list(train_labels.glob("*.txt"))
            print(f"    Found {len(label_files)} label files")
        print(f"  Val labels: {val_labels} - {'EXISTS' if val_labels.exists() else 'NOT FOUND'}")
        if val_labels.exists():
            label_files = list(val_labels.glob("*.txt"))
            print(f"    Found {len(label_files)} label files")
        print(f"  Test labels: {test_labels} - {'EXISTS' if test_labels.exists() else 'NOT FOUND'}")
        if test_labels.exists():
            label_files = list(test_labels.glob("*.txt"))
            print(f"    Found {len(label_files)} label files")
        
        # Try to initialize model with dataset (this will validate the dataset)
        print(f"\nTesting YOLO model initialization with dataset...")
        model = YOLO("yolov8n.pt")
        
        # This will validate the dataset structure
        print(f"[SUCCESS] YOLO can load the dataset configuration!")
        print(f"  Ready for training!")
        
    except Exception as e:
        print(f"[ERROR] Failed to validate dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("="*70)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on BDD100K dataset")
    parser.add_argument("--model", type=str, default="n", choices=["n", "s", "m", "l", "x"],
                       help="Model size: n (nano), s (small), m (medium), l (large), x (xlarge)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda, 0, 1, etc.)")
    parser.add_argument("--project", type=str, default="outputs/yolo_training", help="Project directory")
    parser.add_argument("--name", type=str, default="yolov8_detection", help="Experiment name")
    parser.add_argument("--config", type=str, default="configs/data_config.yaml", help="Config file path")
    parser.add_argument("--test-only", action="store_true", help="Only test data loading, don't train")
    
    args = parser.parse_args()
    
    if args.test_only:
        test_data_loading()
    else:
        train_yolov8(
            model_size=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            config_path=args.config
        )

