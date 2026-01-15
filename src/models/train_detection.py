"""Training script for object detection model (YOLOv8)"""

import argparse
from pathlib import Path
from ultralytics import YOLO
from src.utils.config import load_config


def train_yolov8(config_path: str):
    """Train YOLOv8 model for object detection.
    
    Args:
        config_path: Path to training configuration file
    """
    # Load configuration
    config = load_config(config_path)
    model_cfg = config['model']
    training_cfg = config['training']
    data_cfg = load_config(config['data']['config_path'])
    
    print(f"\n{'='*60}")
    print("Training YOLOv8 Detection Model")
    print(f"{'='*60}\n")
    print(f"Model: {model_cfg['version']}")
    print(f"Epochs: {training_cfg['epochs']}")
    print(f"Batch size: {training_cfg['batch_size']}")
    print(f"Learning rate: {training_cfg['learning_rate']}")
    print(f"\n{'='*60}\n")
    
    # Initialize model
    model = YOLO(f"{model_cfg['version']}.pt")  # Load pretrained weights
    
    # Prepare data path (YOLO format)
    # Note: You may need to convert BDD100K to YOLO format or use YOLO's dataset format
    images_dir = Path(data_cfg['data']['images_dir'])
    
    # Train the model
    results = model.train(
        data=str(images_dir.parent),  # YOLO expects dataset.yaml
        epochs=training_cfg['epochs'],
        batch=training_cfg['batch_size'],
        imgsz=640,
        lr0=training_cfg['learning_rate'],
        weight_decay=training_cfg['weight_decay'],
        momentum=training_cfg['momentum'],
        device=config.get('device', 'cuda'),
        project=str(Path(config['outputs']['checkpoint_dir']).parent),
        name='detection',
        exist_ok=True,
        pretrained=model_cfg['pretrained'],
        optimizer=training_cfg['optimizer'],
        verbose=True
    )
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}\n")
    print(f"Best model saved to: {results.save_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 detection model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/detection_config.yaml',
        help='Path to training configuration file'
    )
    
    args = parser.parse_args()
    
    train_yolov8(args.config)


if __name__ == '__main__':
    main()

