"""TotalSegmentator generation/resegmentation and supporting configuration.

Purpose: TotalSegmentator generation/resegmentation and supporting configuration.
Inputs: caller arguments and existing script settings.
Outputs: return values or existing export destinations.
Dependencies: publication.
Role: main runner.
Usage: the README in this workflow directory; existing CLI/settings remain applicable.
"""

# Direct-file entry points resolve imports from the repository root.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    __package__ = 'publication.masks.segmentation'
from publication.masks.segmentation.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
