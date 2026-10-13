"""Compatibility entry point; implementation: publication/masks/segmentation/core.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared.compat import forward
forward(__name__, 'publication.masks.segmentation.core')
