"""Evaluation module for Insight-Chain."""

from .evaluate import evaluate_model
from .metrics import calculate_accuracy, calculate_exact_match

__all__ = ['evaluate_model', 'calculate_accuracy', 'calculate_exact_match']
