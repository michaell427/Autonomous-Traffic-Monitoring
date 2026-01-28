# Fix for DataLoader batching issue
# Copy this code into your notebook cell that tests DataLoader

from torch.utils.data import DataLoader
from src.data.dataset import BDD100KDetectionDataset, collate_fn  # Import collate_fn
from src.data.augmentation import get_detection_augmentation

# Create transform
transform = get_detection_augmentation(
    image_size=(640, 640),
    training=True
)

# Create dataset
dataset = BDD100KDetectionDataset(
    images_dir=str(IMAGES_DIR),
    labels_dir=str(LABELS_DIR),
    split="train",
    transform=transform,
    target_classes=None
)

# Create DataLoader with custom collate function
# IMPORTANT: Add collate_fn=collate_fn to handle variable-length boxes
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    collate_fn=collate_fn  # <-- ADD THIS LINE
)

print(f"✓ DataLoader created")
print(f"  Dataset size: {len(dataset)}")
print(f"  Batch size: 4")
print(f"  Number of batches: {len(dataloader)}")

# Try loading a batch
batch = next(iter(dataloader))
print(f"\n✓ Batch loaded successfully:")
print(f"  Batch image shape: {batch['image'].shape}")
print(f"  Batch boxes shape: {batch['boxes'].shape}")
print(f"  Batch labels shape: {batch['labels'].shape}")
print(f"  Number of samples in batch: {len(batch['image_id'])}")
print(f"  Image tensor dtype: {batch['image'].dtype}")

if hasattr(batch['image'], 'min'):
    print(f"  Image tensor range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")

print(f"\n  Note: Boxes are padded to same length per batch")
print(f"  Labels with -1 indicate padding (ignore in loss calculation)")

