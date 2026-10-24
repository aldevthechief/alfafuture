"""Pytest root for the whole suite: the repo root and every tests/ subfolder on sys.path
(several suites import helper test modules by name, e.g. test_hybrid_news_search)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
for folder in [HERE] + [d for d in HERE.rglob('*') if d.is_dir() and '__pycache__' not in d.name]:
    sys.path.insert(0, str(folder))
