"""Summary agent model wrapper."""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
from typing import Dict, Any


class SummaryAgent:
    """Wrapper for summary agent model."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize summary agent.
        
        Args:
            model_path: Path to trained model
            device: Device to load model on
        """
        print(f"Loading summary agent from {model_path}...")
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device
        )
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = next(self.model.parameters()).device
        
        print(f" Summary agent loaded on {self.device}")
    
    def extract_answer(
        self,
        image: Image.Image,
        question: str,
        reasoning_chain: str,
        max_new_tokens: int = 128,
        temperature: float = 0.3
    ) -> str:
        """
        Evaluate reasoning and extract final answer.
        
        Args:
            image: PIL Image
            question: Original question
            reasoning_chain: Generated reasoning chain
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower for more deterministic)
            
        Returns:
            Final answer as string
        """
        # Format conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Question: {question}\n\nReasoning:\n{reasoning_chain}\n\nEvaluate this reasoning and provide the final answer."}
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
                do_sample=True
            )
        
        # Decode
        result = self.processor.decode(output[0], skip_special_tokens=True)
        
        # Extract only assistant response
        if "assistant" in result.lower():
            result = result.split("assistant")[-1].strip()
        
        return result
