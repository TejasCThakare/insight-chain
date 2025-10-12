"""Evaluation script for trained models."""

import json
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
from PIL import Image

from src.models.pipeline import MultiAgentPipeline
from .metrics import calculate_accuracy, calculate_exact_match


def evaluate_model(
    pipeline: MultiAgentPipeline,
    data_file: str,
    output_file: str,
    max_samples: int = None
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        pipeline: MultiAgentPipeline instance
        data_file: Path to JSON file with test data
        output_file: Path to save results
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"📊 Evaluating on {data_file}...")
    
    # Load test data
    with open(data_file, 'r') as f:
        test_data = json.load(f)
    
    if max_samples:
        test_data = test_data[:max_samples]
    
    print(f"📝 Evaluating {len(test_data)} samples...")
    
    results = []
    correct = 0
    exact_matches = 0
    
    for item in tqdm(test_data, desc="Evaluating"):
        try:
            # Load image
            image = Image.open(item['image_path']).convert('RGB')
            
            # Run inference
            output = pipeline.generate(
                image=image,
                question=item['question'],
                return_reasoning=True
            )
            
            # Get ground truth
            ground_truth = item['final_answer'].lower().strip()
            predicted = output['final_answer'].lower().strip()
            
            # Check if correct (contains answer)
            is_correct = ground_truth in predicted or predicted in ground_truth
            
            # Check exact match
            is_exact_match = ground_truth == predicted
            
            if is_correct:
                correct += 1
            if is_exact_match:
                exact_matches += 1
            
            # Store result
            results.append({
                "id": item['id'],
                "question": item['question'],
                "ground_truth": item['final_answer'],
                "reasoning_chain": output['reasoning_chain'],
                "predicted_answer": output['final_answer'],
                "is_correct": is_correct,
                "is_exact_match": is_exact_match
            })
            
        except Exception as e:
            print(f"\n⚠️ Error on {item['id']}: {e}")
            continue
    
    # Calculate metrics
    total = len(results)
    accuracy = (correct / total) * 100 if total > 0 else 0
    exact_match_score = (exact_matches / total) * 100 if total > 0 else 0
    
    metrics = {
        "total_samples": total,
        "correct": correct,
        "exact_matches": exact_matches,
        "accuracy": accuracy,
        "exact_match_score": exact_match_score
    }
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            "metrics": metrics,
            "results": results
        }, f, indent=2)
    
    print(f"\n✅ Evaluation complete!")
    print(f"📊 Accuracy: {accuracy:.2f}%")
    print(f"📊 Exact Match: {exact_match_score:.2f}%")
    print(f"💾 Results saved to: {output_file}")
    
    return metrics
