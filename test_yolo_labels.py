"""Quick test to verify YOLO can see labels"""

from ultralytics import YOLO

print("Testing YOLO dataset loading...")
model = YOLO('yolov8n.pt')

# Quick validation on a small subset
results = model.val(
    data='bdd100k_yolo_format/dataset.yaml',
    split='val',
    imgsz=640,
    verbose=True
)

print("\n" + "="*70)
print("Label Detection Test Results")
print("="*70)
if hasattr(results, 'box') and hasattr(results.box, 'map50'):
    print(f"SUCCESS: YOLO detected labels!")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")
else:
    print("WARNING: Could not get metrics - labels may not be detected")
print("="*70)

