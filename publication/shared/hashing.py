"""Checksum helper shared by query-point detector methods.

Inputs: file path and read-block size. Outputs: SHA-256 hexadecimal digest.
Dependencies: hashlib and pathlib. Role: shared helper.
Usage: imported by detector adapters; no scientific execution.
"""
import hashlib
from pathlib import Path

def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
