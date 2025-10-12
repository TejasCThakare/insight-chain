"""Metrics computation for evaluation."""

from typing import List, Dict
import numpy as np


def calculate_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Calculate accuracy (contains match).
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
        
    Returns:
        Accuracy as percentage
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    
    correct = 0
    for pred, truth in zip(predictions, ground_truths):
        pred_lower = pred.lower().strip()
        truth_lower = truth.lower().strip()
        
        if truth_lower in pred_lower or pred_lower in truth_lower:
            correct += 1
    
    return (correct / len(predictions)) * 100 if predictions else 0


def calculate_exact_match(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Calculate exact match score.
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
        
    Returns:
        Exact match score as percentage
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    
    exact_matches = 0
    for pred, truth in zip(predictions, ground_truths):
        if pred.lower().strip() == truth.lower().strip():
            exact_matches += 1
    
    return (exact_matches / len(predictions)) * 100 if predictions else 0


def calculate_f1_score(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Calculate F1 score at token level.
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
        
    Returns:
        Average F1 score
    """
    f1_scores = []
    
    for pred, truth in zip(predictions, ground_truths):
        pred_tokens = set(pred.lower().split())
        truth_tokens = set(truth.lower().split())
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            f1_scores.append(0.0)
            continue
        
        common = pred_tokens & truth_tokens
        
        if len(common) == 0:
            f1_scores.append(0.0)
            continue
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1)
    
    return np.mean(f1_scores) * 100 if f1_scores else 0
