"""Compatibility entry point; implementation: publication/investigations/crop_validation/organ_group_workflow_decision.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.compat import forward
forward(__name__, 'publication.investigations.crop_validation.organ_group_workflow_decision')
