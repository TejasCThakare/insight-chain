"""Training utilities."""

import torch
from peft import LoraConfig, get_peft_model


def setup_model_for_training(model, model_config: dict):
    """Prepare model for LoRA training (FP16, no quantization)."""
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Setup LoRA
    lora_config = LoraConfig(
        r=model_config['lora_r'],
        lora_alpha=model_config['lora_alpha'],
        target_modules=model_config['target_modules'],
        lora_dropout=model_config['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    return model


def get_quantization_config(model_config: dict):
    """Get quantization config (no longer used - returns None)."""
    return None
