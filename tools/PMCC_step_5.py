"""Compatibility entry point; implementation: miscellaneous/PMCC/PMCC_step_5.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.compat import forward
forward(__name__, 'miscellaneous.PMCC.PMCC_step_5')
