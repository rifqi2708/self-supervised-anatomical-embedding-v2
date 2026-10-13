"""Compatibility entry point; implementation: miscellaneous/PMCC/pmcc_rmse.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.compat import forward
forward(__name__, 'miscellaneous.PMCC.pmcc_rmse')
