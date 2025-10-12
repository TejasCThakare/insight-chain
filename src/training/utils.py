"""Training utilities."""

import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def setup_model_for_training(model, model_config: dict):
    """Prepare model for QLoRA training."""
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
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
    """Get BitsAndBytes quantization config."""
    
    return BitsAndBytesConfig(
        load_in_4bit=model_config['use_4bit'],
        bnb_4bit_quant_type=model_config['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=getattr(torch, model_config['bnb_4bit_compute_dtype']),
        bnb_4bit_use_double_quant=model_config.get('bnb_4bit_use_double_quant', True)
    )
