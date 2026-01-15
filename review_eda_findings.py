"""Interactive script to review and document EDA findings"""

import json
from pathlib import Path
from src.data.analyze_eda_findings import generate_summary_report

def main():
    print("\n" + "="*60)
    print("EDA Findings Review - Step 1.1")
    print("="*60)
    
    print("\nThis script will:")
    print("1. Analyze your dataset")
    print("2. Generate a summary report")
    print("3. Help you document findings in docs/eda_findings.md")
    
    input("\nPress Enter to start analysis...")
    
    # Generate summary
    generate_summary_report()
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Review the summary above")
    print("2. Check docs/eda_summary.json for detailed statistics")
    print("3. Fill in docs/eda_findings.md with your findings")
    print("4. Make decisions based on the analysis")
    print("\nKey things to look for:")
    print("  - Class imbalance (ratio > 10:1 is concerning)")
    print("  - Small objects percentage (may need special handling)")
    print("  - Images with 0 objects (should be minimal)")
    print("  - Image size consistency (affects training)")
    print("="*60)

if __name__ == "__main__":
    main()

