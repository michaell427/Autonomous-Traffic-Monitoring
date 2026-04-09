# Quick Start Guide

For the **logical flow** of the repo (JSON vs YOLO layout, what runs when), read [context.md](context.md).

## Getting Started

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

**Notes:**
- **Detectron2** is not in `requirements.txt` (optional segmentation; see `requirements-detectron2.txt`). Core installs should complete with YOLO only.
- **YOLO** uses [Ultralytics](https://github.com/ultralytics/ultralytics) (`ultralytics` in `requirements.txt`).
- **Upload web UI** uses Gradio. Install when needed: `pip install -r requirements-webui.txt`.

### 2. Verify Dataset Structure

**PyTorch / preprocess pipeline** — paths should match `configs/data_config.yaml` (defaults below):

```
.
+-- bdd100k_images_100k/
|   +-- 100k/
|       +-- train/
|       +-- val/
|       +-- test/
+-- bdd100k_labels/
|   +-- 100k/              # detection annotations (JSON) — see data_config.yaml
+-- bdd100k_drivable_maps/
|   +-- color_labels/
|       +-- train/
+-- bdd100k_seg_maps/      # optional, for future segmentation
    +-- color_labels/
    +-- labels/
```

Some BDD100K releases use a different label folder name (for example `bdd100k_det_20_labels`). If so, update `det_labels_dir` in `configs/data_config.yaml` to point at your JSON labels root.

**Ultralytics YOLO training** uses the converted layout:

```
bdd100k_yolo_format/
  dataset.yaml
  train/  images/  labels/
  val/    images/  labels/
  test/   images/  labels/
```

If `bdd100k_yolo_format/dataset.yaml` is missing, use or adapt the preparation scripts under `src/data/` (for example `prepare_dataset.py`, `fix_yolo_dataset.py`, `fix_yolo_labels.py`).

### 3. Validate and Preprocess Data (Phase 1)

Always pass an operation flag; `--config` alone only loads the YAML and suggests `--all`.

```bash
# Validate dataset structure (train / val / test)
python src/data/preprocess.py --config configs/data_config.yaml --validate

# Test data loading with augmentation
python src/data/preprocess.py --config configs/data_config.yaml --test-loader

# Generate data quality report (JSON sample stats)
python src/data/preprocess.py --config configs/data_config.yaml --report

# Comprehensive quality report under outputs/reports/
python src/data/preprocess.py --config configs/data_config.yaml --quality-report

# Or run validate + test-loader + report together
python src/data/preprocess.py --config configs/data_config.yaml --all
```

### 4. Train Detection Model (Phase 2)

**Primary path (BDD100K in YOLO format):**

```bash
# Verify dataset.yaml and folders before training
python src/models/train_yolo.py --test-only

python src/models/train_yolo.py --model n --epochs 100 --batch 16 --project outputs/yolo_training --name yolov8_detection
```

Useful flags: `--device cuda`, `--imgsz 640`, `--config configs/data_config.yaml` (for shared path settings). Weights and plots are written under `outputs/yolo_training/<name>/`.

**Alternate path (single config file for hyperparameters):**

```bash
python src/models/train_detection.py --config configs/detection_config.yaml
```

### 5. Evaluate and Quick Checks

```bash
python src/models/evaluate_yolo.py --model outputs/yolo_training/<run_name>/weights/best.pt --dataset bdd100k_yolo_format/dataset.yaml --split val
```

Copy mAP and related numbers into [`docs/experiment_log.md`](docs/experiment_log.md) so experiments stay traceable.

### 5b. Run inference (image, video, webcam)

```bash
python src/inference.py --weights outputs/yolo_training/<run_name>/weights/best.pt --source path/to/video.mp4
python src/inference.py --weights yolov8n.pt --source bdd100k_yolo_format/val/images --name val_preview
python src/inference.py --weights yolov8n.pt --source 0 --show
```

Outputs default to `outputs/inference/<name>/`. Add `--stream` for long videos.

**Tracking** (video, webcam, or folder of frames): add `--track` (ByteTrack by default). Example:

```bash
python src/inference.py --weights outputs/yolo_training/<run_name>/weights/best.pt --source path/to/video.mp4 --track --name mot_run
```

Optional smoke tests in the repo root (edit paths inside if your run name differs):

- `python quick_yolo_check.py`
- `python visualize_yolo_predictions.py`
- `python demo_before_after.py --weights yolov8n.pt --source bdd100k_yolo_format/val/images` — Space toggles before/after, **Left/Right** prev/next image (needs **`opencv-python`** with GUI, not headless). If GUI fails: `--save-dir outputs/demo_ba` writes JPEGs; or `pip uninstall opencv-python-headless -y` then `pip install opencv-python`
- `python demo_upload_window.py --weights yolov8n.pt` — **Tk** window: queue images/videos, before/after, play/step frames, optional **Tracking IDs**; `--file path` can be repeated to pre-load. Headless OpenCV is OK; UI may be slow on long videos
- `python app_upload_before_after.py --weights yolov8n.pt` — upload image/video in browser and see before/after (install Gradio first)

### 6. Next Steps (roadmap)

- **Drivable / semantic segmentation** — full implementer spec: [docs/segmentation_drivable_handoff.md](docs/segmentation_drivable_handoff.md) (not wired in repo yet)
- Deeper MOT (metrics, custom tracker configs, Re-ID) beyond Ultralytics `--track`
- Faster interactive preview (threading / smaller resize / ONNX) for `demo_upload_window.py` if demos need it
- Model optimization (quantization, TensorRT) and one **unified** det+seg+track pipeline

See [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) for the full plan.

## Configuration

- `configs/data_config.yaml` — dataset paths, augmentation, batch size, workers
- `configs/detection_config.yaml` — model and training settings for `train_detection.py`
- `bdd100k_yolo_format/dataset.yaml` — Ultralytics data definition for `train_yolo.py` / `evaluate_yolo.py`

## Troubleshooting

1. **Import errors** — Install dependencies; ensure the project root is the current working directory when running `src/...` scripts.
2. **Dataset not found** — Check paths in `configs/data_config.yaml` and `bdd100k_yolo_format/dataset.yaml`.
3. **CUDA out of memory** — Lower `--batch` or `batch_size` in configs.
4. **Label format** — JSON labels for the PyTorch dataset vs `.txt` YOLO labels under `bdd100k_yolo_format/`; use `src/data/` helpers if conversion or fixes are needed.

## Project Structure

```
.
+-- src/
|   +-- data/
|   +-- models/         # train_yolo.py, train_detection.py, evaluate_yolo.py
|   +-- inference.py    # detection; add --track for MOT on video/webcam/folder
|   +-- tracking/       # run_tracking() → same as inference --track
|   +-- utils/
+-- configs/
+-- bdd100k_yolo_format/
+-- outputs/            # training runs, reports, visualizations
+-- docs/               # EDA notes
```

## Current Status

**Done:** Project layout, configs, BDD100K loading and preprocess CLI, augmentation, YOLO-format dataset integration, Ultralytics training and evaluation, detection + tracking inference (`--track`).

**In progress / next:** Segmentation training in-repo, MOT benchmarks / custom trackers, unified multi-task pipeline and deployment.
