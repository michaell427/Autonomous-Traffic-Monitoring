"""Generate data quality report for BDD100K dataset"""

import argparse
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

from src.data.validation import validate_dataset_directory
from src.utils.config import load_config


def generate_quality_report(config_path: str = "configs/data_config.yaml",
                           output_dir: str = "outputs/reports",
                           splits: list = None,
                           sample_size: int = None):
    """Generate comprehensive data quality report.
    
    Args:
        config_path: Path to data configuration file
        output_dir: Directory to save report
        splits: List of splits to analyze (default: ['train', 'val', 'test'])
        sample_size: If provided, only analyze this many files per split (for quick check)
    """
    if splits is None:
        splits = ['train', 'val', 'test']
    
    config = load_config(config_path)
    data_cfg = config['data']
    labels_dir = Path(data_cfg['det_labels_dir'])
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"data_quality_report_{timestamp}.json"
    summary_file = output_dir / f"data_quality_summary_{timestamp}.txt"
    
    print("="*70)
    print("BDD100K Data Quality Report")
    print("="*70)
    print(f"Labels directory: {labels_dir}")
    print(f"Analyzing splits: {splits}")
    if sample_size:
        print(f"Sample size per split: {sample_size} files")
    print("="*70)
    print()
    
    all_results = {}
    summary_lines = []
    
    for split in splits:
        print(f"\nAnalyzing {split} split...")
        print("-" * 70)
        
        stats = validate_dataset_directory(
            labels_dir, split=split, sample_size=sample_size
        )
        
        all_results[split] = stats
        
        # Print summary
        print(f"\n{split.upper()} Split Results:")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Files processed: {stats['files_processed']}")
        print(f"  Files with errors: {stats['files_with_errors']}")
        print(f"  Total frames: {stats['total_frames']}")
        print(f"  Total objects: {stats['total_objects']}")
        print(f"  Valid boxes: {stats['total_valid_boxes']}")
        print(f"  Invalid boxes: {stats['total_invalid_boxes']}")
        
        total_boxes = stats['total_valid_boxes'] + stats['total_invalid_boxes']
        if total_boxes > 0:
            valid_ratio = stats['total_valid_boxes'] / total_boxes
            print(f"  Validation rate: {valid_ratio:.2%}")
        
        print(f"\n  Invalid box reasons:")
        for reason, count in stats['invalid_reasons'].items():
            if count > 0:
                print(f"    {reason}: {count}")
        
        print(f"\n  Class distribution:")
        sorted_classes = sorted(stats['classes'].items(), key=lambda x: x[1], reverse=True)
        for class_name, count in sorted_classes[:10]:  # Top 10
            print(f"    {class_name}: {count}")
        
        if stats['error_files']:
            print(f"\n  Files with errors: {len(stats['error_files'])}")
            if len(stats['error_files']) <= 10:
                for error_file in stats['error_files']:
                    print(f"    - {error_file}")
            else:
                print(f"    (Showing first 10 of {len(stats['error_files'])} error files)")
                for error_file in stats['error_files'][:10]:
                    print(f"    - {error_file}")
        
        # Add to summary
        summary_lines.append(f"\n{split.upper()} Split:")
        summary_lines.append(f"  Files: {stats['files_processed']}/{stats['total_files']}")
        summary_lines.append(f"  Valid boxes: {stats['total_valid_boxes']}")
        summary_lines.append(f"  Invalid boxes: {stats['total_invalid_boxes']}")
        if stats['total_valid_boxes'] + stats['total_invalid_boxes'] > 0:
            valid_ratio = stats['total_valid_boxes'] / (stats['total_valid_boxes'] + stats['total_invalid_boxes'])
            summary_lines.append(f"  Validation rate: {valid_ratio:.2%}")
    
    # Save detailed JSON report
    with open(report_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nDetailed report saved to: {report_file}")
    
    # Save summary text report
    with open(summary_file, 'w') as f:
        f.write("BDD100K Data Quality Report Summary\n")
        f.write("="*70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Labels directory: {labels_dir}\n")
        f.write("="*70 + "\n")
        for line in summary_lines:
            f.write(line + "\n")
    print(f"Summary report saved to: {summary_file}")
    
    print("\n" + "="*70)
    print("Report generation complete!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data quality report")
    parser.add_argument("--config", type=str, default="configs/data_config.yaml",
                       help="Path to data configuration file")
    parser.add_argument("--output", type=str, default="outputs/reports",
                       help="Output directory for reports")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                       help="Dataset splits to analyze")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Sample size per split (for quick check)")
    
    args = parser.parse_args()
    
    generate_quality_report(
        config_path=args.config,
        output_dir=args.output,
        splits=args.splits,
        sample_size=args.sample_size
    )

