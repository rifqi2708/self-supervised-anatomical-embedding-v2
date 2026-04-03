import argparse
import os
import sys
import time
from datetime import datetime

import torch

sys.path.append("..")
sys.path.append(".")

if torch.cuda.is_available():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    print("Using GPU")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("Using CPU")

try:
    from embedding_cache import (
        INDEX_SCHEMA_VERSION,
        discover_image_files,
        embedding_relpath_for_image_relpath,
        resolve_runtime_device,
        save_embedding_file,
        write_embedding_index,
    )
except ImportError:
    from tools.embedding_cache import (
        INDEX_SCHEMA_VERSION,
        discover_image_files,
        embedding_relpath_for_image_relpath,
        resolve_runtime_device,
        save_embedding_file,
        write_embedding_index,
    )
from interfaces import get_embedding, init
from utils import read_image


os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))  # go to root dir of this project

DEFAULT_INPUT_ROOT = "data/quadra_dataset_males_cropped"
DEFAULT_EMBEDDING_ROOT = "data/quadra_dataset_males_cropped_embeddings"
DEFAULT_CONFIG_FILE = "configs/sam/sam_NIHLN.py"
DEFAULT_CHECKPOINT_FILE = "checkpoints/SAM.pth"


def run_precompute(
    input_root=DEFAULT_INPUT_ROOT,
    embedding_root=DEFAULT_EMBEDDING_ROOT,
    embedding_index_file=None,
    config_file=DEFAULT_CONFIG_FILE,
    checkpoint_file=DEFAULT_CHECKPOINT_FILE,
    is_mri=False,
    overwrite=False,
    embedding_device=None,
):
    input_root = os.path.abspath(input_root)
    embedding_root = os.path.abspath(embedding_root)
    config_file = os.path.abspath(config_file) if not os.path.isabs(config_file) else config_file
    checkpoint_file = os.path.abspath(checkpoint_file) if not os.path.isabs(checkpoint_file) else checkpoint_file

    if embedding_index_file is None:
        embedding_index_file = os.path.join(embedding_root, "embeddings_index.json")
    embedding_index_file = (
        os.path.abspath(embedding_index_file)
        if not os.path.isabs(embedding_index_file)
        else embedding_index_file
    )

    runtime_device = resolve_runtime_device(embedding_device)

    print(f"Input root: {input_root}")
    print(f"Embedding root: {embedding_root}")
    print(f"Embedding index: {embedding_index_file}")
    print(f"Embedding runtime device: {runtime_device}")
    print(f"Overwrite existing embeddings: {overwrite}")

    image_files = discover_image_files(input_root)
    print(f"Found {len(image_files)} image files under '{os.path.join(input_root, 'images')}'.")

    t0 = time.time()
    model = init(config_file, checkpoint_file)
    t1 = time.time()
    print(f"Model loading time: {t1 - t0:.3f}s")

    records = []
    computed = 0
    reused = 0

    for idx, (image_rel, image_abs) in enumerate(image_files, start=1):
        embedding_rel = embedding_relpath_for_image_relpath(image_rel)
        embedding_abs = os.path.join(embedding_root, embedding_rel)

        print(f"[{idx:04d}/{len(image_files):04d}] {image_rel}")
        if os.path.exists(embedding_abs) and not overwrite:
            reused += 1
            print(f"  Reusing existing embedding: {embedding_abs}")
        else:
            image_info, normed_im, _ = read_image(image_abs, mask_path=None, is_MRI=is_mri)
            _ = image_info
            embedding = get_embedding(normed_im, model)
            save_embedding_file(
                embedding_abs,
                embedding,
                metadata={
                    "source_image": image_rel,
                    "source_image_abs": image_abs,
                },
            )
            computed += 1
            print(f"  Saved embedding: {embedding_abs}")

        records.append(
            {
                "source_image": image_rel,
                "source_image_abs": image_abs,
                "embedding_relpath": os.path.normpath(embedding_rel),
                "embedding_abs": os.path.abspath(embedding_abs),
            }
        )

    payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "dataset": "quadra_dataset_males_cropped",
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_root": input_root,
        "embedding_root": embedding_root,
        "config_file": config_file,
        "checkpoint_file": checkpoint_file,
        "embedding_device": str(runtime_device),
        "num_images": len(records),
        "num_computed": int(computed),
        "num_reused": int(reused),
        "images": records,
    }
    write_embedding_index(embedding_index_file, payload)

    t2 = time.time()
    print(f"\nEmbedding precompute complete in {t2 - t1:.3f}s")
    print(f"Computed: {computed}, Reused: {reused}, Total indexed: {len(records)}")
    print(f"Embedding index written: {embedding_index_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute and cache SAM embeddings for QUADRA cropped images."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default=DEFAULT_INPUT_ROOT,
        help="Dataset root containing images/ and masks/ (cropped layout).",
    )
    parser.add_argument(
        "--embedding-root",
        type=str,
        default=DEFAULT_EMBEDDING_ROOT,
        help="Root directory where embedding .pt files are stored.",
    )
    parser.add_argument(
        "--embedding-index-file",
        type=str,
        default=None,
        help="Optional path to embeddings index JSON. Defaults to <embedding-root>/embeddings_index.json.",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=DEFAULT_CONFIG_FILE,
        help="Model config file.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=DEFAULT_CHECKPOINT_FILE,
        help="Model checkpoint file.",
    )
    parser.add_argument(
        "--is-mri",
        action="store_true",
        help="Enable MRI intensity processing path in read_image.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite existing embedding files.",
    )
    parser.add_argument(
        "--embedding-device",
        type=str,
        default=None,
        help="Runtime device for loaded embeddings later (metadata only). Default resolves to cuda:0 if available, else cpu.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_precompute(
        input_root=args.input_root,
        embedding_root=args.embedding_root,
        embedding_index_file=args.embedding_index_file,
        config_file=args.config_file,
        checkpoint_file=args.checkpoint_file,
        is_mri=args.is_mri,
        overwrite=args.overwrite,
        embedding_device=args.embedding_device,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
