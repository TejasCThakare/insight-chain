"""Training module for Insight-Chain."""

from .train_reasoning import train_reasoning_agent
from .train_summary import train_summary_agent

__all__ = ['train_reasoning_agent', 'train_summary_agent']
