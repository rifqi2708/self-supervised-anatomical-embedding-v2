"""Embedding cache persistence and lookup helpers for Quadra workflows."""

import json
import os
from datetime import datetime

import torch


INDEX_SCHEMA_VERSION = 1
EMBEDDING_SCHEMA_VERSION = 1


def is_nifti_file(name):
    return isinstance(name, str) and (name.endswith(".nii.gz") or name.endswith(".nii"))


def strip_nii_suffix(filename):
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename


def embedding_relpath_for_image_relpath(image_relpath):
    return f"{strip_nii_suffix(os.path.normpath(image_relpath))}.pt"


def normalized_path_variants(path):
    if not isinstance(path, str) or not path:
        return set()
    variants = {os.path.normpath(path)}
    variants.add(os.path.normpath(os.path.abspath(path)))
    return variants


def discover_image_files(input_root):
    input_root = os.path.abspath(input_root)
    images_root = os.path.join(input_root, "images")
    if not os.path.isdir(images_root):
        raise FileNotFoundError(f"Images root not found: {images_root}")

    image_files = []
    for dirpath, dirnames, filenames in os.walk(images_root):
        dirnames.sort()
        filenames.sort()
        for name in filenames:
            if not is_nifti_file(name):
                continue
            image_abs = os.path.join(dirpath, name)
            image_rel = os.path.relpath(image_abs, input_root)
            image_files.append((os.path.normpath(image_rel), os.path.abspath(image_abs)))

    image_files.sort(key=lambda pair: pair[0])
    if not image_files:
        raise RuntimeError(f"No image files found under: {images_root}")
    return image_files


def _has_tensor(payload):
    if torch.is_tensor(payload):
        return True
    if isinstance(payload, dict):
        return any(_has_tensor(v) for v in payload.values())
    if isinstance(payload, (list, tuple)):
        return any(_has_tensor(v) for v in payload)
    return False


def validate_embedding_payload(embedding, source=None):
    if not _has_tensor(embedding):
        source_str = f" in '{source}'" if source else ""
        raise ValueError(f"Embedding payload{source_str} does not contain any tensors.")
    if isinstance(embedding, (list, tuple)) and len(embedding) < 2:
        source_str = f" in '{source}'" if source else ""
        raise ValueError(
            f"Embedding payload{source_str} must contain at least fine and coarse levels (len >= 2)."
        )


def _to_cpu_recursive(payload):
    if torch.is_tensor(payload):
        return payload.detach().cpu()
    if isinstance(payload, dict):
        return {k: _to_cpu_recursive(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_to_cpu_recursive(v) for v in payload]
    if isinstance(payload, tuple):
        return tuple(_to_cpu_recursive(v) for v in payload)
    return payload


def _to_device_recursive(payload, device):
    if torch.is_tensor(payload):
        non_blocking = bool(getattr(device, "type", None) == "cuda")
        return payload.to(device=device, non_blocking=non_blocking)
    if isinstance(payload, dict):
        return {k: _to_device_recursive(v, device) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_to_device_recursive(v, device) for v in payload]
    if isinstance(payload, tuple):
        return tuple(_to_device_recursive(v, device) for v in payload)
    return payload


def resolve_runtime_device(device=None):
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested embedding device '{resolved}' but CUDA is not available.")
    return resolved


def save_embedding_file(path, embedding, metadata=None):
    validate_embedding_payload(embedding)
    payload = {
        "schema_version": EMBEDDING_SCHEMA_VERSION,
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "embedding": _to_cpu_recursive(embedding),
    }
    if isinstance(metadata, dict):
        payload.update(metadata)

    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    torch.save(payload, path)


def load_embedding_file(path, device=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embedding file not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Failed to load embedding file '{path}': {exc}") from exc

    if isinstance(payload, dict) and "embedding" in payload:
        embedding = payload["embedding"]
    else:
        embedding = payload

    validate_embedding_payload(embedding, source=path)

    if device is not None:
        embedding = _to_device_recursive(embedding, device)
    return embedding


def write_embedding_index(index_path, index_payload):
    parent_dir = os.path.dirname(index_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_payload, f, indent=2)


def load_embedding_index(index_path):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Embedding index not found: {index_path}")
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    if not isinstance(index, dict):
        raise ValueError("Embedding index must be a JSON object.")
    records = index.get("images")
    if not isinstance(records, list) or not records:
        raise ValueError("Embedding index must contain a non-empty 'images' list.")
    return index


def build_embedding_lookup(index_payload):
    lookup = {}
    input_root = index_payload.get("input_root")
    embedding_root = index_payload.get("embedding_root")

    for record in index_payload.get("images", []):
        source_image = record.get("source_image")
        source_image_abs = record.get("source_image_abs")
        embedding_relpath = record.get("embedding_relpath")
        embedding_abs = record.get("embedding_abs")

        if embedding_abs:
            resolved_embedding = os.path.normpath(embedding_abs)
        elif embedding_root and embedding_relpath:
            resolved_embedding = os.path.normpath(os.path.join(embedding_root, embedding_relpath))
        else:
            raise ValueError(f"Invalid embedding index record (missing embedding path): {record}")

        def register(path, root=None):
            for variant in normalized_path_variants(path):
                lookup[variant] = resolved_embedding
            if isinstance(path, str) and path and root and not os.path.isabs(path):
                joined = os.path.join(root, path)
                for variant in normalized_path_variants(joined):
                    lookup[variant] = resolved_embedding

        register(source_image, root=input_root)
        register(source_image_abs)

    if not lookup:
        raise ValueError("No valid path mappings found in embedding index.")
    return lookup


def resolve_embedding_path(image_path, embedding_lookup, embedding_index_file):
    for key in normalized_path_variants(image_path):
        if key in embedding_lookup:
            return embedding_lookup[key]

    raise KeyError(
        f"Could not find embedding for image '{image_path}' in embedding index '{embedding_index_file}'. "
        "Re-run tools/quadra/precompute_quadra_embeddings.py for this dataset and verify image paths match."
    )
