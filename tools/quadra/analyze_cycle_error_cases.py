"""Compatibility entry point; implementation: publication/analysis/analyze_cycle_error_cases.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.compat import forward
forward(__name__, 'publication.analysis.analyze_cycle_error_cases')
