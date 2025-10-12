"""
Download free datasets with reasoning annotations.
Usage: python -m src.data.download_datasets
"""

import os
from datasets import load_dataset
from pathlib import Path
import yaml


def download_datasets(config_path: str = "configs/data_config.yaml"):
    """Download free datasets with reasoning."""
    
    print("📦 Downloading FREE datasets with reasoning...")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['data']
    
    output_path = Path(config['raw_dir'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    num_aokvqa = config['num_aokvqa_samples']
    num_science = config['num_scienceqa_samples']
    
    # 1. A-OKVQA - Has reasoning rationales
    print(f"\n1️⃣ Downloading A-OKVQA ({num_aokvqa} samples)...")
    try:
        aokvqa = load_dataset("HuggingFaceM4/A-OKVQA", split=f"train[:{num_aokvqa}]")
        save_path = output_path / "aokvqa"
        aokvqa.save_to_disk(str(save_path))
        print(f"   ✅ Saved {len(aokvqa)} samples to {save_path}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # 2. ScienceQA - Has explanations
    print(f"\n2️⃣ Downloading ScienceQA ({num_science} samples)...")
    try:
        scienceqa = load_dataset("derek-thomas/ScienceQA", split=f"train[:{num_science}]")
        save_path = output_path / "scienceqa"
        scienceqa.save_to_disk(str(save_path))
        print(f"   ✅ Saved {len(scienceqa)} samples to {save_path}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    print("\n✅ All datasets downloaded!")
    print(f"💰 Total cost: $0.00")
    print(f"📊 Total samples: ~{num_aokvqa + num_science}")
    
    return True


if __name__ == "__main__":
    download_datasets()
