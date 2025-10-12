"""Multi-agent inference pipeline combining reasoning and summary agents."""

import torch
from PIL import Image
from pathlib import Path
from typing import Dict, Optional
import yaml

from .reasoning_agent import ReasoningAgent
from .summary_agent import SummaryAgent


class MultiAgentPipeline:
    """Complete inference pipeline with reasoning and summary agents."""
    
    def __init__(
        self,
        reasoning_model_path: str,
        summary_model_path: str,
        config_path: str = "configs/model_config.yaml",
        device: str = "auto"
    ):
        """
        Initialize multi-agent pipeline.
        
        Args:
            reasoning_model_path: Path to reasoning agent
            summary_model_path: Path to summary agent
            config_path: Path to model config
            device: Device to load models on
        """
        print("🔗 Initializing Multi-Agent Pipeline")
        print("="*60)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['model']
        
        # Initialize agents
        self.reasoning_agent = ReasoningAgent(reasoning_model_path, device)
        self.summary_agent = SummaryAgent(summary_model_path, device)
        
        print("\n✅ Pipeline ready!")
    
    def generate(
        self,
        image: Image.Image,
        question: str,
        return_reasoning: bool = True
    ) -> Dict[str, str]:
        """
        Run complete inference pipeline.
        
        Args:
            image: PIL Image
            question: Question about the image
            return_reasoning: Whether to return reasoning chain
            
        Returns:
            Dictionary with reasoning_chain and final_answer
        """
        # Step 1: Generate reasoning chain
        print("\n🧠 Generating reasoning chain...")
        reasoning_chain = self.reasoning_agent.generate_reasoning(
            image=image,
            question=question,
            max_new_tokens=self.config.get('max_new_tokens', 512),
            temperature=self.config.get('temperature', 0.7),
            top_p=self.config.get('top_p', 0.9)
        )
        
        # Step 2: Extract final answer
        print("\n📝 Extracting final answer...")
        final_answer = self.summary_agent.extract_answer(
            image=image,
            question=question,
            reasoning_chain=reasoning_chain
        )
        
        result = {
            "question": question,
            "final_answer": final_answer
        }
        
        if return_reasoning:
            result["reasoning_chain"] = reasoning_chain
        
        return result
    
    def generate_from_path(
        self,
        image_path: str,
        question: str,
        return_reasoning: bool = True
    ) -> Dict[str, str]:
        """
        Run inference from image path.
        
        Args:
            image_path: Path to image file
            question: Question about the image
            return_reasoning: Whether to return reasoning chain
            
        Returns:
            Dictionary with reasoning_chain and final_answer
        """
        image = Image.open(image_path).convert('RGB')
        return self.generate(image, question, return_reasoning)
