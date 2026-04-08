# Autonomous Traffic Monitoring & Analysis

A comprehensive computer vision system for real-time traffic monitoring that combines object detection, instance segmentation, and multi-object tracking.

## Project Overview

This project implements a production-ready traffic monitoring system that can:
- **Detect** vehicles (cars, trucks, buses, motorcycles) and pedestrians in traffic scenes
- **Segment** road lanes, sidewalks, crosswalks, and infrastructure
- **Track** objects across video frames with consistent IDs

## What is implemented today

| Area | Status |
|------|--------|
| BDD100K loading, augmentation, preprocessing CLI | Implemented (`src/data/`) |
| YOLO-format dataset + Ultralytics training | Implemented (`bdd100k_yolo_format/`, `src/models/train_yolo.py`) |
| Alternate training entry (config-driven) | Implemented (`src/models/train_detection.py`, `configs/detection_config.yaml`) |
| YOLO validation / metrics | Implemented (`src/models/evaluate_yolo.py`); log runs in `docs/experiment_log.md` |
| Image / video / webcam inference (detection) | Implemented (`src/inference.py`) |
| Multi-object tracking on frame sequences | Implemented (`--track` on `src/inference.py`, ByteTrack by default; `src/tracking/` wrapper) |
| Upload UI (image/video before/after) | Implemented (`app_upload_before_after.py`, Gradio) |
| Segmentation training, unified det+seg+track pipeline | Not in repo yet (planned) |

## Architecture (target)

1. **Object Detection** — YOLOv8 / Ultralytics (primary path in this repo)
2. **Instance Segmentation** — Mask R-CNN or YOLO-seg (planned)
3. **Multi-Object Tracking** — ByteTrack / BoT-SORT via Ultralytics on video, webcam, or ordered image folders (`--track`)

## Project Structure

```
.
+-- src/
|   +-- data/           # Loaders, preprocess, dataset fixes, EDA helpers
|   +-- models/         # train_yolo, train_detection, evaluate_yolo
|   +-- inference.py    # detection + optional `--track` (ByteTrack / BoT-SORT)
|   +-- tracking/       # `run_tracking()` helper → same as inference --track
|   +-- utils/          # Config loading, shared utilities
+-- configs/            # data_config.yaml, detection_config.yaml
+-- bdd100k_yolo_format/# YOLO layout + dataset.yaml for Ultralytics
+-- outputs/            # Training runs (e.g. outputs/yolo_training/), reports
+-- docs/               # EDA notes and templates
+-- comprehensive_cv_project.md
+-- QUICKSTART.md
+-- PROJECT_ROADMAP.md
```

Add a `notebooks/` folder locally if you use Jupyter; it is not required by the scripts above.

**How the pipeline fits together (data paths, two formats, train → eval → inference):** see [context.md](context.md).

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

**Windows / Detectron2:** `detectron2` is **not** in `requirements.txt` (it is not installable via plain `pip` on most Windows/Python combos). This repo’s YOLO path works without it. If you add Mask R-CNN later, see [requirements-detectron2.txt](requirements-detectron2.txt) and the [official install guide](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md).

**Install hiccups:**
- **`WinError 32` / file in use** while pip installs (often `dvc`): close other Python processes and the IDE’s terminal using that venv, then run `pip install -r requirements.txt` again.
- **OpenCV GUI** (`demo_before_after.py` window fails): `albumentations` may pull `opencv-python-headless`. For an on-screen window, run `pip uninstall opencv-python-headless -y` and keep `opencv-python`, or use `--save-dir` on the demo script.
- **Upload web UI** (`app_upload_before_after.py`) needs Gradio: `pip install -r requirements-webui.txt`.

### Dataset layout (matches `configs/data_config.yaml`)

Paths are relative to the project root. Adjust `configs/data_config.yaml` if your folders differ.

```
bdd100k_images_100k/
  100k/
    train/
    val/
    test/
bdd100k_labels/
  100k/                 # detection labels (JSON), used by the PyTorch dataset path in config
bdd100k_drivable_maps/
  color_labels/
    train/
bdd100k_seg_maps/       # optional, for segmentation experiments
```

For **Ultralytics YOLO** training, you also need the prepared layout under `bdd100k_yolo_format/` (see `bdd100k_yolo_format/dataset.yaml`). Use the scripts under `src/data/` (for example `prepare_dataset.py`, `fix_yolo_*`) if you need to regenerate or repair that tree.

## Usage

### Data validation and reports

You must pass an operation flag; `--config` alone only loads config and prints a hint.

```bash
# Run validate + test loader + JSON report (typical first check)
python src/data/preprocess.py --config configs/data_config.yaml --all

# Full quality report helper
python src/data/preprocess.py --config configs/data_config.yaml --quality-report
```

### Train detection (recommended: BDD100K YOLO layout)

```bash
# Sanity-check YOLO dataset paths only
python src/models/train_yolo.py --test-only

# Train (example)
python src/models/train_yolo.py --model n --epochs 100 --batch 16 --project outputs/yolo_training --name my_run
```

### Train detection (alternate: `detection_config.yaml`)

```bash
python src/models/train_detection.py --config configs/detection_config.yaml
```

### Evaluate a checkpoint

```bash
python src/models/evaluate_yolo.py --model path/to/best.pt --dataset bdd100k_yolo_format/dataset.yaml --split val
```

### Run inference (image, video, or webcam)

From the project root, pass your `best.pt` and a path to an image/video/folder, or `0` for the default webcam:

```bash
python src/inference.py --weights outputs/yolo_training/<run_name>/weights/best.pt --source path/to/video.mp4
python src/inference.py --weights yolov8n.pt --source bdd100k_yolo_format/val/images
python src/inference.py --weights yolov8n.pt --source 0 --show
```

Annotated media is saved under `outputs/inference/<name>/` (override with `--output-dir` and `--name`). Use `--stream` for long videos to reduce memory use.

**Tracking** needs a **sequence of frames** (video file, webcam `0`, or a folder of images). Per-frame detection alone is not tracking; `--track` assigns persistent IDs across frames using Ultralytics’ tracker (default **ByteTrack**):

```bash
python src/inference.py --weights outputs/yolo_training/<run_name>/weights/best.pt --source path/to/video.mp4 --track --name tracked
python src/inference.py --weights yolov8n.pt --source 0 --track --show
# BoT-SORT instead of ByteTrack:
python src/inference.py --weights yolov8n.pt --source clip.mp4 --track --tracker botsort.yaml
```

### Optional helper scripts (repo root)

- `quick_yolo_check.py` — quick inference smoke test on a few val images
- `visualize_yolo_predictions.py` — saves annotated images (edit weights path inside the script if needed)
- `demo_before_after.py` — **Space** toggles raw vs detections, **arrow keys** prev/next image (**Q** quit); needs **`opencv-python`** with GUI (not `opencv-python-headless`). If windows fail, use `--save-dir outputs/...` to write before/after/pair JPEGs instead
- `app_upload_before_after.py` — local upload app (image/video) with before/after preview in the browser; install `gradio` first (`pip install -r requirements-webui.txt`)

### Not available yet

- End-to-end **segmentation** training (`train_segmentation.py`, `segmentation_config.yaml`) and a single **unified** det+seg+track pipeline are **not** wired up yet. See `PROJECT_ROADMAP.md`.

## Performance Targets

- Detection mAP > 0.6
- Segmentation mask mAP > 0.5
- Tracking MOTA > 0.7
- Real-time inference (30 FPS)

## Development Status

- [x] Project structure and configs
- [x] Data pipeline (BDD100K loaders, preprocess CLI, augmentation)
- [x] YOLO-format dataset + Ultralytics train / eval scripts
- [ ] Segmentation training integrated in repo
- [x] Multi-object tracking on video / webcam / image folders (`src/inference.py --track`; `src/tracking.run_tracking`)
- [x] Before/after toggle demo (`demo_before_after.py`; OpenCV, local display only)
- [x] Upload before/after UI (`app_upload_before_after.py`; Gradio)
- [x] Detection inference CLI (`src/inference.py` — image, video, webcam)
- [ ] Deployment / optimization pass

For a step-by-step plan, see [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md). For commands in order, see [QUICKSTART.md](QUICKSTART.md).

## License

[Add your license here]

## Contributing

[Add contribution guidelines if applicable]

## Contact

[Add contact information if desired]
