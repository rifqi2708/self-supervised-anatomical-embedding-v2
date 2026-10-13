# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
"""Compatibility entry point; implementation: miscellaneous/misc/lymphnode_preprocess_crop_multi_process.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.compat import forward
forward(__name__, 'miscellaneous.misc.lymphnode_preprocess_crop_multi_process')
