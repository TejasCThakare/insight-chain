"""
PyTorch Dataset classes for vision-language reasoning.
"""

import json
from pathlib import Path
from typing import Dict
import torch
from torch.utils.data import Dataset
from PIL import Image


class ReasoningDataset(Dataset):
    """Dataset for training reasoning agents."""
    
    def __init__(self, data_file: str, processor, max_length: int = 1024):
        """
        Args:
            data_file: Path to JSON file with training data
            processor: Qwen2-VL processor
            max_length: Maximum sequence length
        """
        self.processor = processor
        self.max_length = max_length
        
        # Load data
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} samples from {data_file}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Load image
        image = Image.open(item['image_path']).convert('RGB')
        
        # Format conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Question: {item['question']}\n\nProvide detailed step-by-step reasoning."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": item['reasoning_chain']}
                ]
            }
        ]
        
        # Process with Qwen2-VL processor
        text = self.processor.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Squeeze batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # Create labels for training
        inputs["labels"] = inputs["input_ids"].clone()
        
        return inputs


class SummaryDataset(Dataset):
    """Dataset for training summary agents."""
    
    def __init__(self, data_file: str, processor, max_length: int = 1024):
        self.processor = processor
        self.max_length = max_length
        
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} samples from {data_file}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        image = Image.open(item['image_path']).convert('RGB')
        
        # Different prompt for summary agent
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Question: {item['question']}\n\nReasoning:\n{item['reasoning_chain']}\n\nEvaluate this reasoning and provide the final answer."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"The reasoning is coherent and well-structured. Final answer: {item['final_answer']}"}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        
        return inputs
