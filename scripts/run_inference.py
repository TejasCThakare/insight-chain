#!/usr/bin/env python3
"""
Run inference with trained models.
Usage: python scripts/run_inference.py --image path/to/image.jpg --question "What is this?"
"""

import argparse
import sys
from pathlib import Path
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_inference(image_path: str, question: str, model_path: str = "models/reasoning_agent/final"):
    """Run inference on a single image."""
    
    print("🔗 Insight-Chain Inference")
    print("="*60)
    
    # Load model
    print(f"\n📥 Loading model from {model_path}...")
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained(model_path)
    
    # Load image
    print(f"\n📷 Loading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    # DIFFERENT PROMPTS FOR DIFFERENT AGENTS
    if "summary" in model_path:
        prompt_text = f"Question: {question}\n\nProvide a concise single-sentence answer."
        print("📝 Using SUMMARY prompt")
    else:
        prompt_text = f"Question: {question}\n\nProvide detailed step-by-step reasoning."
        print("📝 Using REASONING prompt")
    
    # Create conversation
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]
    
    # Process
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[image], return_tensors="pt").to(device)
    
    # Generate
    print("\n🧠 Generating...")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True
        )
    
    result = processor.decode(output[0], skip_special_tokens=True)
    
    # Print result
    print("\n" + "="*60)
    print("📝 RESULT:")
    print("="*60)
    print(result)
    print("="*60)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Insight-Chain Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--question", type=str, required=True, help="Question about the image")
    parser.add_argument("--model", type=str, default="models/reasoning_agent/final", help="Path to model")
    
    args = parser.parse_args()
    
    run_inference(args.image, args.question, args.model)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
