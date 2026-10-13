"""Compatibility entry point; implementation: publication/uae/organ_group_lattice_alignment.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.compat import forward
forward(__name__, 'publication.uae.organ_group_lattice_alignment')
