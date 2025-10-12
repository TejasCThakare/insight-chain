#!/usr/bin/env python3
"""
Complete data pipeline: download + prepare.
Usage: python scripts/run_data_pipeline.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.download_datasets import download_datasets
from src.data.prepare_data import prepare_training_data


def main():
    print("🔗 Insight-Chain Data Pipeline")
    print("="*60)
    
    # Step 1: Download datasets
    print("\n📦 STEP 1: Downloading datasets...")
    success = download_datasets()
    
    if not success:
        print("\n❌ Failed to download datasets")
        return 1
    
    # Step 2: Prepare training data
    print("\n🔄 STEP 2: Preparing training data...")
    success = prepare_training_data()
    
    if not success:
        print("\n❌ Failed to prepare data")
        return 1
    
    print("\n✅ Data pipeline complete!")
    print("📊 Ready for training!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
