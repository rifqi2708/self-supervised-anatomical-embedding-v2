# Copyright (c) OpenMMLab. All rights reserved.
"""Compatibility entry point; implementation: shared/test_sam.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.compat import forward
forward(__name__, 'shared.test_sam')
