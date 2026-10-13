"""Compatibility entry point; implementation: publication/registration/registration_organ_group_cohort_report.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.compat import forward
forward(__name__, 'publication.registration.registration_organ_group_cohort_report')
