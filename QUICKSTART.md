# Quick Start Guide

## Getting Started

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

**Note:** Detectron2 requires separate installation. See [Detectron2 installation guide](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md).

### 2. Verify Dataset Structure

Your BDD100K dataset should be organized as follows:

```
.
+-- bdd100k_images_100k/
|   +-- 100k/
|       +-- train/
|       +-- val/
|       +-- test/
+-- bdd100k_det_20_labels/
|   +-- train/
|   +-- val/
|   +-- test/
+-- bdd100k_drivable_maps/
|   +-- color_labels/
|       +-- train/
+-- bdd100k_seg_maps/
    +-- color_labels/
    +-- labels/
```

### 3. Validate and Preprocess Data (Phase 1)

```bash
# Validate dataset structure
python src/data/preprocess.py --validate

# Test data loading with augmentation
python src/data/preprocess.py --test-loader

# Generate data quality report
python src/data/preprocess.py --report

# Or run all preprocessing steps
python src/data/preprocess.py --all
```

### 4. Train Detection Model (Phase 2)

```bash
# Train YOLOv8 detection model
python src/models/train_detection.py --config configs/detection_config.yaml
```

**Note:** Before training, you may need to convert BDD100K labels to YOLO format, or modify the training script to work directly with BDD100K format.

### 5. Next Steps

- **Phase 2**: Train segmentation model (Mask R-CNN or YOLOv8-seg)
- **Phase 2**: Implement multi-object tracking (DeepSORT/ByteTrack)
- **Phase 3**: Model optimization (quantization, TensorRT)
- **Phase 4**: Evaluation and visualization

## Configuration

Edit configuration files in `configs/`:
- `data_config.yaml` - Dataset paths and augmentation settings
- `detection_config.yaml` - Detection model training parameters

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed
2. **Dataset not found**: Check paths in `configs/data_config.yaml`
3. **CUDA out of memory**: Reduce batch size in config files
4. **Label format issues**: BDD100K labels may need conversion to YOLO format

## Project Structure

```
.
+-- src/
|   +-- data/           # Data loading and preprocessing
|   +-- models/         # Model training scripts
|   +-- tracking/       # Multi-object tracking
|   +-- utils/          # Utility functions
+-- configs/            # Configuration files
+-- outputs/            # Model checkpoints, logs, visualizations
+-- notebooks/          # Jupyter notebooks for exploration
```

## Current Status

[COMPLETE] Project structure setup
[COMPLETE] Data loading and preprocessing pipeline
[COMPLETE] Augmentation pipeline
[COMPLETE] Configuration system
[COMPLETE] Basic training script

[IN PROGRESS]
- Dataset label format conversion (if needed)
- Full training pipeline
- Segmentation model training
- Tracking implementation

