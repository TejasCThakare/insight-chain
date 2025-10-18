"""
Train reasoning agent.
Usage: python -m src.training.train_reasoning
"""

import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
import yaml
from pathlib import Path

from src.data.dataset import ReasoningDataset
from src.training.utils import setup_model_for_training


def train_reasoning_agent(
    model_config_path: str = "configs/model_config.yaml",
    training_config_path: str = "configs/training_config.yaml",
    data_config_path: str = "configs/data_config.yaml"
):
    """Train the reasoning agent."""
    
    print("🚀 Training Reasoning Agent")
    print("="*60)
    
    # Load configs
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)['model']
    
    with open(training_config_path, 'r') as f:
        train_config = yaml.safe_load(f)['training']
    
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)['data']
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ No GPU detected! Training will be very slow.")
    
    # Load model WITHOUT quantization (FP16 is enough!)
    print(f"\n📥 Loading {model_config['base_model']}...")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_config['base_model'],
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(
        model_config['base_model'],
        trust_remote_code=True
    )
    
    # Setup LoRA
    print("\n🔧 Setting up LoRA training...")
    model = setup_model_for_training(model, model_config)
    model.print_trainable_parameters()
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_file = Path(data_config['processed_dir']) / "train.json"
    val_file = Path(data_config['processed_dir']) / "val.json"
    
    train_dataset = ReasoningDataset(
        str(train_file),
        processor,
        max_length=train_config['max_seq_length']
    )
    
    val_dataset = ReasoningDataset(
        str(val_file),
        processor,
        max_length=train_config['max_seq_length']
    )
    
    # Training arguments
    print("\n⚙️ Configuring training...")
    training_args = TrainingArguments(
        output_dir=train_config['output_dir'],
        num_train_epochs=train_config['num_train_epochs'],
        per_device_train_batch_size=train_config['per_device_train_batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        per_device_eval_batch_size=train_config['per_device_eval_batch_size'],
        learning_rate=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
        warmup_steps=train_config['warmup_steps'],
        max_grad_norm=train_config['max_grad_norm'],
        fp16=train_config['fp16'],
        logging_steps=train_config['logging_steps'],
        save_steps=train_config['save_steps'],
        eval_steps=train_config['eval_steps'],
        save_total_limit=train_config['save_total_limit'],
        load_best_model_at_end=train_config['load_best_model_at_end'],
        gradient_checkpointing=train_config['gradient_checkpointing'],
        optim=train_config['optim'],
        report_to=train_config['report_to'],
        dataloader_num_workers=train_config.get('dataloader_num_workers', 0),
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train!
    print("\n🏃 Starting training...")
    print("="*60)
    trainer.train()
    
    # Save final model
    print("\n💾 Saving final model...")
    output_path = Path(train_config['output_dir']) / "final"
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(str(output_path))
    processor.save_pretrained(str(output_path))
    
    print(f"\n✅ Training complete!")
    print(f"📦 Model saved to: {output_path}")
    
    return model, processor


if __name__ == "__main__":
    train_reasoning_agent()
