"""Fix YOLO label paths by creating labels directories next to images"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
import os


def create_label_symlinks(config_path: str = "configs/data_config.yaml"):
    """Create symlinks/junctions for labels next to image directories."""
    
    config = load_config(config_path)
    data_cfg = config['data']
    images_base = Path(data_cfg['images_dir'])
    labels_base = Path("bdd100k_yolo_format")
    
    print("="*70)
    print("Creating Label Directories for YOLO")
    print("="*70)
    print(f"Images base: {images_base}")
    print(f"Labels base: {labels_base}")
    print()
    
    for split in ['train', 'val', 'test']:
        image_dir = images_base / split
        labels_source = labels_base / split / "labels"
        labels_target = image_dir / "labels"
        
        print(f"\n{split.upper()} split:")
        print(f"  Image directory: {image_dir}")
        print(f"  Labels source: {labels_source}")
        print(f"  Labels target: {labels_target}")
        
        if not image_dir.exists():
            print(f"  [ERROR] Image directory does not exist: {image_dir}")
            continue
        
        if not labels_source.exists():
            print(f"  [ERROR] Labels source does not exist: {labels_source}")
            continue
        
        # Remove existing labels directory if it exists (could be a broken symlink)
        if labels_target.exists() or labels_target.is_symlink():
            print(f"  Removing existing: {labels_target}")
            if labels_target.is_symlink():
                labels_target.unlink()
            elif labels_target.is_dir():
                import shutil
                shutil.rmtree(labels_target)
        
        # Try to create symlink (Windows junction or symbolic link)
        try:
            # On Windows, use junction for directories (works without admin)
            if sys.platform == 'win32':
                import subprocess
                # Create directory junction (Windows)
                subprocess.run(
                    ['mklink', '/J', str(labels_target), str(labels_source.resolve())],
                    shell=True,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"  [SUCCESS] Created junction: {labels_target} -> {labels_source}")
            else:
                # Unix-like: use symlink
                labels_target.symlink_to(labels_source.resolve(), target_is_directory=True)
                print(f"  [SUCCESS] Created symlink: {labels_target} -> {labels_source}")
        except (subprocess.CalledProcessError, OSError) as e:
            # If symlink/junction fails, copy the labels (slower but works)
            print(f"  [WARNING] Could not create symlink/junction: {e}")
            print(f"  [INFO] Copying labels instead (this may take a while)...")
            import shutil
            shutil.copytree(labels_source, labels_target)
            print(f"  [SUCCESS] Copied labels to: {labels_target}")
        
        # Verify
        if labels_target.exists():
            label_files = list(labels_target.glob("*.txt"))
            print(f"  [VERIFY] Found {len(label_files)} label files")
        else:
            print(f"  [ERROR] Labels target does not exist after creation!")
    
    print("\n" + "="*70)
    print("Label linking complete!")
    print("="*70)


if __name__ == "__main__":
    create_label_symlinks()

