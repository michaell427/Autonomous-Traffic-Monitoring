"""Step 1.2: Test and fix data pipeline issues"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("="*60)
    print("Testing Imports")
    print("="*60)
    
    try:
        from src.data.dataset import BDD100KDetectionDataset
        print("✓ BDD100KDetectionDataset imported")
    except Exception as e:
        print(f"✗ Error importing BDD100KDetectionDataset: {e}")
        return False
    
    try:
        from src.data.augmentation import get_detection_augmentation
        print("✓ get_detection_augmentation imported")
    except Exception as e:
        print(f"✗ Error importing augmentation: {e}")
        return False
    
    try:
        from src.utils.config import load_config
        print("✓ load_config imported")
    except Exception as e:
        print(f"✗ Error importing config: {e}")
        return False
    
    print()
    return True


def test_paths():
    """Test that dataset paths exist."""
    print("="*60)
    print("Testing Dataset Paths")
    print("="*60)
    
    paths = {
        'images': Path("bdd100k_images_100k/100k"),
        'det_labels': Path("bdd100k_det_20_labels"),
        'full_labels': Path("bdd100k_labels/100k"),
    }
    
    all_exist = True
    for name, path in paths.items():
        if path.exists():
            print(f"✓ {name}: {path}")
            # Check for subdirectories
            if path.is_dir():
                subdirs = [d for d in path.iterdir() if d.is_dir()]
                if subdirs:
                    print(f"  Subdirectories: {[d.name for d in subdirs]}")
        else:
            print(f"✗ {name}: {path} - NOT FOUND")
            all_exist = False
    
    print()
    return all_exist


def test_label_files():
    """Test that label files can be found."""
    print("="*60)
    print("Testing Label Files")
    print("="*60)
    
    # Check det_20_labels
    det_labels_dir = Path("bdd100k_det_20_labels")
    if det_labels_dir.exists():
        for split in ['train', 'val', 'test']:
            split_dir = det_labels_dir / split
            if split_dir.exists():
                json_files = list(split_dir.glob("*.json"))
                print(f"✓ {split}: {len(json_files)} JSON files")
                if len(json_files) > 0:
                    # Check first file structure
                    try:
                        import json
                        with open(json_files[0], 'r') as f:
                            data = json.load(f)
                        print(f"  Sample file structure: OK")
                    except Exception as e:
                        print(f"  ✗ Error reading sample file: {e}")
            else:
                print(f"✗ {split}: Directory not found")
    else:
        print("✗ bdd100k_det_20_labels directory not found")
    
    print()
    

def test_dataset_loading():
    """Test loading dataset without transforms."""
    print("="*60)
    print("Testing Dataset Loading (No Transforms)")
    print("="*60)
    
    try:
        from src.data.dataset import BDD100KDetectionDataset
        
        # Try loading train split
        print("Loading train split...")
        dataset = BDD100KDetectionDataset(
            images_dir="bdd100k_images_100k/100k",
            labels_dir="bdd100k_det_20_labels",
            split="train",
            transform=None
        )
        
        if len(dataset) == 0:
            print("✗ No data loaded! Check paths.")
            return False
        
        print(f"✓ Loaded {len(dataset)} images")
        
        # Try loading a sample
        try:
            sample = dataset[0]
            print(f"✓ Sample loaded successfully")
            print(f"  Image shape: {sample['image'].shape}")
            print(f"  Number of boxes: {len(sample['boxes'])}")
            print(f"  Number of labels: {len(sample['labels'])}")
            print(f"  Image ID: {sample['image_id']}")
            
            if len(sample['boxes']) > 0:
                print(f"  First box: {sample['boxes'][0]}")
                print(f"  First label: {sample['labels'][0]}")
            
            return True
        except Exception as e:
            print(f"✗ Error loading sample: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"✗ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_augmentation():
    """Test augmentation pipeline."""
    print("="*60)
    print("Testing Augmentation Pipeline")
    print("="*60)
    
    try:
        from src.data.dataset import BDD100KDetectionDataset
        from src.data.augmentation import get_detection_augmentation
        
        # Create augmentation
        transform = get_detection_augmentation(
            image_size=(640, 640),
            training=True
        )
        print("✓ Augmentation transform created")
        
        # Create dataset with augmentation
        dataset = BDD100KDetectionDataset(
            images_dir="bdd100k_images_100k/100k",
            labels_dir="bdd100k_det_20_labels",
            split="train",
            transform=transform
        )
        
        if len(dataset) == 0:
            print("✗ No data to test augmentation on")
            return False
        
        # Try loading a sample
        try:
            sample = dataset[0]
            print(f"✓ Sample with augmentation loaded")
            print(f"  Image tensor shape: {sample['image'].shape}")
            print(f"  Image tensor dtype: {sample['image'].dtype}")
            
            # Check if it's a tensor
            if hasattr(sample['image'], 'min') and hasattr(sample['image'], 'max'):
                print(f"  Image tensor range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
            
            return True
        except Exception as e:
            print(f"✗ Error loading sample with augmentation: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"✗ Error testing augmentation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration loading."""
    print("="*60)
    print("Testing Configuration")
    print("="*60)
    
    try:
        from src.utils.config import load_config
        
        config = load_config("configs/data_config.yaml")
        print("✓ Configuration loaded")
        print(f"  Image size: {config['data']['image_size']}")
        print(f"  Batch size: {config['data']['batch_size']}")
        print(f"  Augmentation enabled: {config['data']['augmentation']['enabled']}")
        return True
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Step 1.2: Data Pipeline Testing")
    print("="*60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Paths", test_paths()))
    test_label_files()  # Just prints info, doesn't return pass/fail
    results.append(("Dataset Loading", test_dataset_loading()))
    results.append(("Augmentation", test_augmentation()))
    results.append(("Configuration", test_config()))
    
    # Summary
    print("="*60)
    print("Test Summary")
    print("="*60)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ All tests passed! Data pipeline is working.")
        print("You can proceed to Step 1.3: Data Quality Improvements")
    else:
        print("⚠️  Some tests failed. Fix the issues above before proceeding.")
        print("\nCommon issues:")
        print("  1. Missing label files - check bdd100k_det_20_labels directory")
        print("  2. Path mismatches - verify paths in configs/data_config.yaml")
        print("  3. Missing dependencies - install requirements.txt")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

