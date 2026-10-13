"""Compatibility entry point; implementation: master_project/scripts/get_tcia_data.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.compat import forward
forward(__name__, 'master_project.scripts.get_tcia_data')
