"""Compatibility entry point; implementation: publication/uae/reviewed_uae_diagnostics.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.compat import forward
forward(__name__, 'publication.uae.reviewed_uae_diagnostics')
