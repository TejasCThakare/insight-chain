"""Visualization utilities for evaluation results."""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import pandas as pd


def plot_accuracy_comparison(
    results_files: Dict[str, str],
    output_file: str = "results/accuracy_comparison.png"
):
    """
    Plot accuracy comparison across different models/methods.
    
    Args:
        results_files: Dict mapping model names to result JSON files
        output_file: Path to save plot
    """
    model_names = []
    accuracies = []
    exact_matches = []
    
    for model_name, file_path in results_files.items():
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        model_names.append(model_name)
        accuracies.append(data['metrics']['accuracy'])
        exact_matches.append(data['metrics']['exact_match_score'])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(model_names))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], accuracies, width, label='Accuracy', color='steelblue')
    ax.bar([i + width/2 for i in x], exact_matches, width, label='Exact Match', color='coral')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score (%)')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to {output_file}")
    
    plt.close()


def create_results_table(
    results_file: str,
    output_file: str = "results/results_table.md",
    num_examples: int = 10
):
    """
    Create markdown table with example results.
    
    Args:
        results_file: Path to results JSON
        output_file: Path to save markdown table
        num_examples: Number of examples to include
    """
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results'][:num_examples]
    
    # Create markdown table
    md_content = "# Evaluation Results\n\n"
    md_content += "## Metrics\n\n"
    md_content += f"- **Accuracy**: {data['metrics']['accuracy']:.2f}%\n"
    md_content += f"- **Exact Match**: {data['metrics']['exact_match_score']:.2f}%\n"
    md_content += f"- **Total Samples**: {data['metrics']['total_samples']}\n\n"
    
    md_content += "## Example Results\n\n"
    md_content += "| Question | Ground Truth | Prediction | Correct |\n"
    md_content += "|----------|--------------|------------|----------|\n"
    
    for result in results:
        question = result['question'][:50] + "..." if len(result['question']) > 50 else result['question']
        gt = result['ground_truth'][:30] + "..." if len(result['ground_truth']) > 30 else result['ground_truth']
        pred = result['predicted_answer'][:30] + "..." if len(result['predicted_answer']) > 30 else result['predicted_answer']
        correct = "✅" if result['is_correct'] else "❌"
        
        md_content += f"| {question} | {gt} | {pred} | {correct} |\n"
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(md_content)
    
    print(f"✅ Table saved to {output_file}")
