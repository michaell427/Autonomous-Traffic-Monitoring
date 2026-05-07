# Autonomous Traffic Monitoring & Analysis

A comprehensive computer vision system for real-time traffic monitoring that combines object detection, **semantic drivable-area segmentation**, and multi-object tracking. Instance segmentation (e.g., per-lane masks) remains a future extension.

## Project Overview

This project implements a production-ready traffic monitoring system that can:
- **Detect** vehicles (cars, trucks, buses, motorcycles) and pedestrians in traffic scenes
- **Segment** BDD100K **drivable area** (direct / alternative / background) at the pixel level
- **Track** objects across video frames with consistent IDs (via Ultralytics + ByteTrack / BoT-SORT)

## What is implemented today

| Area | Status |
|------|--------|
| BDD100K loading, augmentation, preprocessing CLI | Implemented (`src/data/`) |
| YOLO-format dataset + Ultralytics training | Implemented (`bdd100k_yolo_format/`, `src/models/train_yolo.py`) |
| Alternate training entry (config-driven) | Implemented (`src/models/train_detection.py`, `configs/detection_config.yaml`) |
| YOLO validation / metrics | Implemented (`src/models/evaluate_yolo.py`); log runs in `docs/experiment_log.md` |
| Image / video / webcam inference (detection) | Implemented (`src/inference.py`) |
| Multi-object tracking on frame sequences | Implemented (`--track` on `src/inference.py`, ByteTrack by default; `src/tracking/` wrapper) |
| Desktop demo (queue, before/after, play, optional track IDs) | Implemented (`demo_upload_window.py`, Tk + Canvas; `opencv-python-headless` OK for I/O) |
| Upload UI (image/video before/after) | Implemented (`app_upload_before_after.py`, Gradio) |
| Drivable semantic segmentation (BDD100K color masks) | Implemented (`src/models/train_drivable_seg.py`, `evaluate_drivable_seg.py`, `src/inference_drivable.py`, `src/data/drivable_dataset.py`, `configs/drivable_seg_config.yaml`) |
| Instance segmentation (lanes/infrastructure per instance), unified det+seg+track pipeline | Not wired as one product yet (planned) |

## Architecture (target)

1. **Object Detection** — YOLOv8 / Ultralytics (primary path in this repo)
2. **Semantic drivable segmentation** — DeepLabV3–ResNet50 (`torchvision`) on BDD100K `color_labels` PNGs
3. **Instance Segmentation** — Mask R-CNN or YOLO-seg (planned)
4. **Multi-Object Tracking** — ByteTrack / BoT-SORT via Ultralytics on video, webcam, or ordered image folders (`--track`)

## Project Structure

```
.
+-- src/
|   +-- data/           # Loaders, preprocess, dataset fixes, EDA helpers, drivable_dataset
|   +-- models/         # train_yolo, train_detection, evaluate_yolo, train/eval drivable seg
|   +-- inference.py    # detection + optional `--track` (ByteTrack / BoT-SORT)
|   +-- inference_drivable.py  # drivable segmentation on images / folders
|   +-- tracking/       # `run_tracking()` helper → same as inference --track
|   +-- utils/          # Config loading, shared utilities
+-- configs/            # data_config.yaml, detection_config.yaml, drivable_seg_config.yaml
+-- bdd100k_yolo_format/# YOLO layout + dataset.yaml for Ultralytics
+-- outputs/            # Training runs (e.g. outputs/yolo_training/), reports
+-- docs/               # EDA notes and templates
+-- comprehensive_cv_project.md
+-- QUICKSTART.md
+-- PROJECT_ROADMAP.md
```

Add a `notebooks/` folder locally if you use Jupyter; it is not required by the scripts above.

Large **local** downloads (e.g. **BDDA** / BDD-Attention video, full BDD100K trees) are listed in **`.gitignore`** so they are not committed by mistake; keep them on disk or track with **DVC** if you want versioned data.

**How the pipeline fits together (data paths, JSON vs YOLO layout, train → eval → inference):** see [context.md](context.md) if you keep a local copy. Extended drivable-segmentation notes may live in [docs/segmentation_drivable_handoff.md](docs/segmentation_drivable_handoff.md) if present; the training/inference code paths are under `src/` and `configs/drivable_seg_config.yaml`.

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
    val/
bdd100k_seg_maps/
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

### Evaluate a detection checkpoint

```bash
python src/models/evaluate_yolo.py --model path/to/best.pt --dataset bdd100k_yolo_format/dataset.yaml --split val
```

### Train drivable semantic segmentation (BDD100K)

Requires RGB frames under `bdd100k_images_100k/100k/<split>/` and matching **color** PNG masks under `bdd100k_drivable_maps/color_labels/<split>/` (see `configs/drivable_seg_config.yaml`). Paths are validated against the config when you run training.

```bash
python src/models/train_drivable_seg.py --config configs/drivable_seg_config.yaml
# Optional overrides: --epochs, --batch, --lr, --device cuda, --project, --name
```

### Evaluate a drivable segmentation checkpoint

```bash
python src/models/evaluate_drivable_seg.py --config configs/drivable_seg_config.yaml --weights path/to/best.pth --device cuda
```

### Run drivable inference (images or folder of images)

Checkpoints are `.pth` from `train_drivable_seg` (they embed `image_size` and `num_classes` for the script).

```bash
python src/inference_drivable.py --weights outputs/drivable_seg/<run>/best.pth --source path/to/image.jpg --name my_run
python src/inference_drivable.py --weights outputs/drivable_seg/<run>/best.pth --source bdd100k_images_100k/100k/val --overlay
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
- `demo_upload_window.py` — **Tk** app: add multiple images/videos, before/after, frame/media navigation, **Play/Pause**, optional **Tracking IDs** (`--tracker`); preview is **not real-time** on weak CPUs—fine for demos
- `app_upload_before_after.py` — local upload app (image/video) with before/after preview in the browser; install `gradio` first (`pip install -r requirements-webui.txt`)

### Not available yet

- A single **unified** det + drivable-seg + track **product** pipeline (one command / one runtime graph) is **not** wired up yet; detection/tracking and drivable segmentation are separate entry points today.
- Broader **instance segmentation** (Mask R-CNN / YOLO-seg) for lanes and objects as instances is **not** integrated. See `PROJECT_ROADMAP.md`.

## Performance Targets

- Detection mAP > 0.6
- Segmentation mask mAP > 0.5
- Tracking MOTA > 0.7
- Real-time inference (30 FPS)

## Development Status

- [x] Project structure and configs
- [x] Data pipeline (BDD100K loaders, preprocess CLI, augmentation)
- [x] YOLO-format dataset + Ultralytics train / eval scripts
- [x] Drivable semantic segmentation (DeepLabV3, train / eval / `inference_drivable.py`)
- [x] Multi-object tracking on video / webcam / image folders (`src/inference.py --track`; `src/tracking.run_tracking`)
- [x] Before/after toggle demo (`demo_before_after.py`; OpenCV, local display only)
- [x] Desktop upload / before-after / optional tracking demo (`demo_upload_window.py`; Tk)
- [x] Upload before/after UI (`app_upload_before_after.py`; Gradio)
- [x] Detection inference CLI (`src/inference.py` — image, video, webcam)
- [ ] Deployment / optimization pass

For a step-by-step plan, see [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md). For commands in order, see [QUICKSTART.md](QUICKSTART.md).
