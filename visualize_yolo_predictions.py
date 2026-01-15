"""Visualize YOLO predictions on a small set of validation images.

This script:
- Loads the trained baseline YOLOv8 model (best.pt)
- Runs inference on 10 validation images
- Saves annotated images to outputs/yolo_vis/baseline_yolov8n_v2
"""

from pathlib import Path

from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parent

    # Use trained baseline weights if available
    weights_path = (
        project_root
        / "outputs"
        / "yolo_training"
        / "yolov8n_fixeddata"
        / "weights"
        / "best.pt"
    )

    if not weights_path.exists():
        raise FileNotFoundError(f"Trained weights not found: {weights_path}")

    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    # Take 10 images from val split of the YOLO-structured dataset
    val_images_dir = project_root / "bdd100k_yolo_format" / "val" / "images"
    if not val_images_dir.exists():
        raise FileNotFoundError(f"Val images directory not found: {val_images_dir}")

    image_paths = sorted(val_images_dir.glob("*.jpg"))[:10]
    if not image_paths:
        raise RuntimeError(f"No .jpg images found in {val_images_dir}")

    out_dir = project_root / "outputs" / "yolo_vis" / "yolov8n_fixeddata"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running visualization on {len(image_paths)} images...")
    print(f"Saving annotated images to: {out_dir}")

    model.predict(
        source=[str(p) for p in image_paths],
        imgsz=640,
        conf=0.25,
        save=True,
        save_txt=False,
        project=str(out_dir),
        name="predictions",
        exist_ok=True,
        verbose=False,
    )

    print("Done. Check the annotated images in:", out_dir / "predictions")


if __name__ == "__main__":
    main()


