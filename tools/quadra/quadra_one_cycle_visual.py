# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
"""Compatibility entry point; implementation: publication/uae/quadra_one_cycle_visual.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.compat import forward
forward(__name__, 'publication.uae.quadra_one_cycle_visual')
