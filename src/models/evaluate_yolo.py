"""Evaluate YOLOv8 model performance"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
import argparse


def evaluate_model(
    model_path: str,
    dataset_yaml: str = "bdd100k_yolo_format/dataset.yaml",
    split: str = "val",
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7
):
    """Evaluate YOLOv8 model on dataset.
    
    Args:
        model_path: Path to model weights (.pt file)
        dataset_yaml: Path to dataset YAML file
        split: Split to evaluate on ('val' or 'test')
        imgsz: Image size for evaluation
        conf: Confidence threshold
        iou: IoU threshold for NMS
    """
    model_path = Path(model_path)
    dataset_yaml_path = project_root / dataset_yaml
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if not dataset_yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml_path}")
    
    print("="*70)
    print("YOLOv8 Model Evaluation")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_yaml_path}")
    print(f"Split: {split}")
    print(f"Image size: {imgsz}")
    print(f"Confidence threshold: {conf}")
    print(f"IoU threshold: {iou}")
    print("="*70)
    print()
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))
    
    # Evaluate
    print(f"\nEvaluating on {split} split...")
    results = model.val(
        data=str(dataset_yaml_path),
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        plots=True,
        save_json=True,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)
    
    # Print key metrics
    if hasattr(results, 'metrics'):
        metrics = results.metrics
        print("\nKey Metrics:")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall: {metrics.box.mr:.4f}")
    
    print(f"\nResults saved to: {results.save_dir}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model weights (.pt file)")
    parser.add_argument("--dataset", type=str, default="bdd100k_yolo_format/dataset.yaml",
                       help="Path to dataset YAML file")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                       help="Dataset split to evaluate on")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        dataset_yaml=args.dataset,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou
    )

