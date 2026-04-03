"""
main.py — entry point for the Murder Mystery game.

Usage:
    python main.py
"""

import sys
from pathlib import Path

# Ensure project root is on the path when running directly
sys.path.insert(0, str(Path(__file__).parent))

from src.game import run_game

if __name__ == "__main__":
    run_game()
