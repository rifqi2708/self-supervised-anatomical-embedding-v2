"""Compatibility entry point; implementation: publication/investigations/tiled_embeddings/validate_uaes_streaming.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.compat import forward
forward(__name__, 'publication.investigations.tiled_embeddings.validate_uaes_streaming')
