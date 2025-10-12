"""
Prepare training data from downloaded datasets.
Usage: python -m src.data.prepare_data
"""

import json
import os
from pathlib import Path
from datasets import load_from_disk
from PIL import Image
import random
from typing import List, Dict
import yaml


def prepare_training_data(config_path: str = "configs/data_config.yaml"):
    """Prepare training data from raw datasets."""
    
    print("🔄 Preparing training data...")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['data']
    
    # Setup paths
    raw_path = Path(config['raw_dir'])
    output_path = Path(config['processed_dir'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    images_path = output_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)
    
    train_samples: List[Dict] = []
    
    # Load A-OKVQA
    print("\n📂 Processing A-OKVQA...")
    aokvqa_path = raw_path / "aokvqa"
    if aokvqa_path.exists():
        aokvqa = load_from_disk(str(aokvqa_path))
        
        for idx, sample in enumerate(aokvqa):
            # Need at least 2 reasoning steps
            if len(sample['rationales']) >= 2:
                # Save image
                img_path = images_path / f"aokvqa_{idx}.jpg"
                sample['image'].save(str(img_path))
                
                # Format reasoning chain
                reasoning_steps = sample['rationales'][:5]
                reasoning_chain = "\n".join([
                    f"Step {i+1}: {step}" 
                    for i, step in enumerate(reasoning_steps)
                ])
                
                # Get answer
                if len(sample['direct_answers']) > 0:
                    answer = sample['direct_answers'][0]
                else:
                    answer = sample['choices'][sample['correct_choice_idx']]
                
                train_samples.append({
                    "id": f"aokvqa_{idx}",
                    "image_path": str(img_path),
                    "question": sample['question'],
                    "reasoning_chain": reasoning_chain,
                    "final_answer": answer
                })
        
        print(f"   ✅ Processed {len([s for s in train_samples if 'aokvqa' in s['id']])} A-OKVQA samples")
    
    # Load ScienceQA
    print("\n📂 Processing ScienceQA...")
    scienceqa_path = raw_path / "scienceqa"
    if scienceqa_path.exists():
        scienceqa = load_from_disk(str(scienceqa_path))
        
        for idx, sample in enumerate(scienceqa):
            if sample['hint'] and len(sample['hint']) > 20:
                # Save image
                img_path = images_path / f"science_{idx}.jpg"
                sample['image'].save(str(img_path))
                
                # Format as reasoning chain
                reasoning_chain = (
                    f"Step 1: Observing the image and question context.\n"
                    f"Step 2: {sample['hint']}\n"
                    f"Step 3: Analyzing the given choices.\n"
                    f"Step 4: {sample['solution']}"
                )
                
                answer = sample['choices'][sample['answer']]
                
                train_samples.append({
                    "id": f"science_{idx}",
                    "image_path": str(img_path),
                    "question": sample['question'],
                    "reasoning_chain": reasoning_chain,
                    "final_answer": answer
                })
        
        print(f"   ✅ Processed {len([s for s in train_samples if 'science' in s['id']])} ScienceQA samples")
    
    # Split train/val
    random.seed(config['random_seed'])
    random.shuffle(train_samples)
    
    split_idx = int(config['train_split'] * len(train_samples))
    train_split_data = train_samples[:split_idx]
    val_split_data = train_samples[split_idx:]
    
    # Save
    train_file = output_path / "train.json"
    val_file = output_path / "val.json"
    
    with open(train_file, "w") as f:
        json.dump(train_split_data, f, indent=2)
    
    with open(val_file, "w") as f:
        json.dump(val_split_data, f, indent=2)
    
    print(f"\n✅ Data preparation complete!")
    print(f"📊 Train: {len(train_split_data)} samples → {train_file}")
    print(f"📊 Val: {len(val_split_data)} samples → {val_file}")
    print(f"💰 Total cost: $0.00")
    
    return True


if __name__ == "__main__":
    prepare_training_data()
