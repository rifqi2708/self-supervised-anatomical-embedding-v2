"""Compatibility entry point; implementation: miscellaneous/ai/sam_torch.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.compat import forward
forward(__name__, 'miscellaneous.ai.sam_torch')
