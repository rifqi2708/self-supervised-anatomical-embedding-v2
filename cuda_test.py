"""Compatibility entry point; implementation: junk/cuda_test.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from shared.compat import forward
forward(__name__, 'junk.cuda_test')
