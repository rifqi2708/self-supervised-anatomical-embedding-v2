"""Compatibility entry point; implementation: master_project/scripts/train_loop_v2.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from shared.compat import forward
forward(__name__, 'master_project.scripts.train_loop_v2')
