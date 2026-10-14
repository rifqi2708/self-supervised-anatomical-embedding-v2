"""Canonical signature-member and explicit frozen-checkout source lookup.

This registry does not expose obsolete modules in the current source tree.
Original member names are stable signature keys, not compatibility entrypoints.
Dataset/checkpoint/evidence locations are outside this source-only registry.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = {'tools/quadra/aligned_organ_group_cohort.py': 'publication/uae/aligned_organ_group_cohort.py',
 'tools/quadra/disposable_pod.py': 'publication/operations/disposable_pod.py',
 'tools/quadra/environment.py': 'publication/operations/environment.py',
 'tools/quadra/environment/requirements-preprocess.txt': 'publication/operations/requirements-preprocess.txt',
 'tools/quadra/environment/requirements-registration.txt': 'publication/operations/requirements-registration.txt',
 'tools/quadra/memory_configuration_screen.py': 'publication/investigations/memory/memory_configuration_screen.py',
 'tools/quadra/organ_group_lattice_alignment.py': 'publication/uae/organ_group_lattice_alignment.py',
 'tools/quadra/registration_organ_group.py': 'publication/registration/registration_organ_group.py',
 'tools/quadra/registration_organ_group_cohort.py': 'publication/registration/registration_organ_group_cohort.py',
 'tools/quadra/registration_point_transform.py': 'publication/registration/registration_point_transform.py',
 'tools/quadra/registration_runtime.py': 'publication/registration/registration_runtime.py',
 'tools/quadra/reviewed_evidence.py': 'publication/shared/reviewed_evidence.py',
 'tools/quadra/reviewed_matching_contract.py': 'publication/shared/reviewed_matching_contract.py',
 'tools/quadra/reviewed_registration.py': 'publication/registration/reviewed_registration.py',
 'tools/quadra/reviewed_uae_cuda.py': 'publication/uae/reviewed_uae_cuda.py',
 'tools/quadra/reviewed_uae_diagnostics.py': 'publication/uae/reviewed_uae_diagnostics.py',
 'tools/quadra/reviewed_uae_export.py': 'publication/uae/reviewed_uae_export.py',
 'tools/quadra/reviewed_uae_matching.py': 'publication/uae/reviewed_uae_matching.py',
 'tools/quadra/streaming_cycle_error.py': 'publication/uae/streaming_cycle_error.py',
 'tools/quadra/streaming_embedding.py': 'publication/uae/streaming_embedding.py',
 'tools/quadra/uaes_matching.py': 'publication/uae/uaes_matching.py'}

def source_path(original, root=None):
    """Use canonical source; permit old layouts only for explicit checkouts."""
    base = ROOT if root is None else Path(root)
    canonical = base / SOURCE_PATHS.get(original, original)
    if root is not None and not canonical.exists():
        return base / original
    return canonical

def checkout_module(original, root):
    """Select a canonical or frozen legacy module in a specified checkout.

    A fixture-only checkout with neither member defaults to the canonical name;
    real execution still fails normally if the selected source is unavailable.
    """
    relative = original.replace('.', '/') + '.py'
    canonical = SOURCE_PATHS.get(relative, relative)
    base = Path(root)
    if (base/canonical).is_file() or not (base/relative).is_file():
        return canonical[:-3].replace('/', '.')
    return original
