"""Reasoning agent model wrapper."""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
from typing import Dict, Any


class ReasoningAgent:
    """Wrapper for reasoning agent model."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize reasoning agent.
        
        Args:
            model_path: Path to trained model
            device: Device to load model on
        """
        print(f"Loading reasoning agent from {model_path}...")
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device
        )
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = next(self.model.parameters()).device
        
        print(f"✅ Reasoning agent loaded on {self.device}")
    
    def generate_reasoning(
        self, 
        image: Image.Image, 
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate reasoning chain for a question about an image.
        
        Args:
            image: PIL Image
            question: Question about the image
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated reasoning chain as string
        """
        # Format conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Question: {question}\n\nProvide detailed step-by-step reasoning."}
                ]
            }
        ]
        
        # Process
        text_prompt = self.processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        
        # Decode
        result = self.processor.decode(output[0], skip_special_tokens=True)
        
        # Extract only assistant response
        if "assistant" in result.lower():
            result = result.split("assistant")[-1].strip()
        
        return result
