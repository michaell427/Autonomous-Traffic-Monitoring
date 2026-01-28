# Autonomous Traffic Monitoring & Analysis

A comprehensive computer vision system for real-time traffic monitoring that combines object detection, instance segmentation, and multi-object tracking.

## Project Overview

This project implements a production-ready traffic monitoring system that can:
- **Detect** vehicles (cars, trucks, buses, motorcycles) and pedestrians in traffic scenes
- **Segment** road lanes, sidewalks, crosswalks, and infrastructure
- **Track** objects across video frames with consistent IDs

## Architecture

The system consists of three main components:

1. **Object Detection Module** - YOLOv8 for vehicle and pedestrian detection
2. **Instance Segmentation Module** - Mask R-CNN or YOLOv8-seg for lane and infrastructure segmentation
3. **Multi-Object Tracking Module** - DeepSORT/ByteTrack for maintaining object identities across frames

## Project Structure

```
.
+-- src/
|   +-- data/          # Data loading and preprocessing
|   +-- models/        # Model definitions and training
|   +-- tracking/      # Multi-object tracking implementation
|   +-- utils/         # Utility functions
+-- configs/           # Configuration files
+-- notebooks/         # Jupyter notebooks for exploration
+-- outputs/
|   +-- checkpoints/   # Model checkpoints
|   +-- logs/          # Training logs
|   +-- visualizations/# Output visualizations
+-- comprehensive_cv_project.md  # Detailed project plan

```

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 50GB+ free disk space for datasets

### Installation

1. Clone the repository:
```bash
git clone https://github.com/michaell427/Autonomous-Traffic-Monitoring.git
cd Autonomous-Traffic-Monitoring
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Detectron2 (if using Mask R-CNN):
```bash
# Follow instructions at https://github.com/facebookresearch/detectron2
```

### Dataset Setup

This project uses the BDD100K dataset. The dataset structure should be:

```
bdd100k_images_100k/
  +-- 100k/
      +-- train/
      +-- val/
      +-- test/
bdd100k_det_20_labels/
  +-- train/
  +-- val/
  +-- test/
bdd100k_drivable_maps/
  +-- color_labels/
      +-- train/
```

## Usage

### Phase 1: Data Engineering
```bash
python src/data/preprocess.py --config configs/data_config.yaml
```

### Phase 2: Model Training
```bash
# Train detection model
python src/models/train_detection.py --config configs/detection_config.yaml

# Train segmentation model
python src/models/train_segmentation.py --config configs/segmentation_config.yaml
```

### Phase 3: Inference
```bash
python src/inference.py --video path/to/video.mp4 --output outputs/visualizations/
```

## Performance Targets

- Detection mAP > 0.6
- Segmentation mask mAP > 0.5
- Tracking MOTA > 0.7
- Real-time inference (30 FPS)

## Development Status

- [x] Project structure setup
- [ ] Phase 1: Data Engineering & Preparation
- [ ] Phase 2: Model Development
- [ ] Phase 3: Model Optimization & Deployment
- [ ] Phase 4: Evaluation & Analysis

## License

[Add your license here]

## Contributing

[Add contribution guidelines if applicable]

## Contact

[Add contact information if desired]

