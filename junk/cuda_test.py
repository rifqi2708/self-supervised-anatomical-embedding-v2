"""Approved retained review-queue placement; preserve original bytes, notices and recovery context.

Purpose: Approved retained review-queue placement; preserve original bytes, notices and recovery context.
Inputs: caller arguments and existing script settings.
Outputs: return values or existing export destinations.
Dependencies: os, torch.
Role: shared helper.
Usage: the README in this workflow directory; existing CLI/settings remain applicable.
"""

# Direct-file entry points resolve imports from the repository root.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = 'junk'
import os
import torch
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Use CPU