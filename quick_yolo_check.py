"""Quick sanity check: run YOLO on 10 validation images.

This does a fast forward-pass only (no metrics), just to confirm that:
- The trained model loads correctly
- The new YOLO dataset structure is valid
- We can run inference on a small batch of images without errors
"""

from pathlib import Path

from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parent

    # Use the best baseline weights if available; otherwise fall back to yolov8n.pt
    trained_weights = (
        project_root
        / "outputs"
        / "yolo_training"
        / "baseline_yolov8n_v2"
        / "weights"
        / "best.pt"
    )

    if trained_weights.exists():
        print(f"Loading trained model: {trained_weights}")
        model = YOLO(str(trained_weights))
    else:
        print("Trained weights not found, falling back to yolov8n.pt")
        model = YOLO("yolov8n.pt")

    # Take 10 images from the new YOLO-format val/images directory
    val_images_dir = project_root / "bdd100k_yolo_format" / "val" / "images"
    if not val_images_dir.exists():
        raise FileNotFoundError(f"Val images directory not found: {val_images_dir}")

    image_paths = sorted(val_images_dir.glob("*.jpg"))[:10]
    if not image_paths:
        raise RuntimeError(f"No .jpg images found in {val_images_dir}")

    print(f"Running inference on {len(image_paths)} images...")

    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=640,
        conf=0.25,
        save=False,
        verbose=False,
    )

    # Basic confirmation output
    print("Inference completed successfully.")
    print(f"Got {len(results)} result objects.")
    print("First image path:", image_paths[0])


if __name__ == "__main__":
    main()


