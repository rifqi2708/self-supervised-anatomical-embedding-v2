"""Compatibility entry point; implementation: miscellaneous/misc/metric/compute_landmark_distance.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.compat import forward
forward(__name__, 'miscellaneous.misc.metric.compute_landmark_distance')
