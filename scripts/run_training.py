#!/usr/bin/env python3
"""
Main training script.
Usage: 
    python scripts/run_training.py --mode reasoning
    python scripts/run_training.py --mode summary
    python scripts/run_training.py --mode full
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train_reasoning import train_reasoning_agent
from src.training.train_summary import train_summary_agent


def main():
    parser = argparse.ArgumentParser(description="Insight-Chain Training")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["reasoning", "summary", "full"],
        default="reasoning",
        help="Training mode"
    )
    
    args = parser.parse_args()
    
    print("🔗 Insight-Chain Training")
    print("="*60)
    
    # Train reasoning agent
    if args.mode in ["reasoning", "full"]:
        print("\n Training Reasoning Agent...")
        train_reasoning_agent()
    
    # Train summary agent
    if args.mode in ["summary", "full"]:
        print("\n Training Summary Agent...")
        train_summary_agent()
    
    print("\n Training complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
