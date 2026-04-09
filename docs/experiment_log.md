# Detection experiments (YOLO)

Use this file as the **single place** to record what you trained and what numbers you got. Pull metrics from Ultralytics outputs and from `evaluate_yolo.py` so README / roadmap / resume stay aligned with reality.

## How to capture metrics

1. **After training:** Ultralytics writes under `outputs/yolo_training/<run_name>/` — see `results.csv`, plots, and `weights/best.pt`.
2. **Structured eval:** From the repo root:
   ```bash
   python src/models/evaluate_yolo.py --model outputs/yolo_training/<run_name>/weights/best.pt --dataset bdd100k_yolo_format/dataset.yaml --split val
   ```
   The script prints mAP50, mAP50-95, precision, recall and saves under the run’s val folder (see console for `save_dir`).
3. **Copy** the printed metrics (and optional notes) into the table below.

## Run log

| Date | Run name (`--name`) | Model | Epochs | Batch | imgsz | Weights (`best.pt`) | mAP50 | mAP50-95 | Precision | Recall | Notes |
|------|---------------------|-------|--------|-------|-------|----------------------|-------|----------|-----------|--------|-------|
| | | e.g. yolov8n | | | 640 | `outputs/yolo_training/.../best.pt` | | | | | |

Add more rows as you run new experiments. If you change class list or dataset YAML, note it in **Notes**.

## Targets (from roadmap)

- Detection mAP (overall) > **0.6** — track your best mAP50-95 here and in project docs when you hit milestones.

## Drivable segmentation (DeepLabV3)

Train: `python src/models/train_drivable_seg.py --config configs/drivable_seg_config.yaml` — best weights `outputs/drivable_seg/<run_name>/best.pth`. Optional: append a row with `--log-experiment`.

| Date (UTC) | Run name | val mIoU | Checkpoint | Notes |
|------------|----------|----------|------------|-------|
| | | | | |

## Related paths

- Data config: `configs/data_config.yaml`
- Drivable seg config: `configs/drivable_seg_config.yaml`
- Detection training config (alternate entry): `configs/detection_config.yaml`
- YOLO data YAML: `bdd100k_yolo_format/dataset.yaml`
- Qualitative checks on new video or images: `python src/inference.py --weights ... --source ...` (saved under `outputs/inference/` by default)
