"""
conftest.py — Shared pytest fixtures and path bootstrapping.

Adds the project root to sys.path so `from rag.xxx import ...` works
regardless of where pytest is invoked from.
"""
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
