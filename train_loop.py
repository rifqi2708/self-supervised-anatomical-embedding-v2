import csv
import os
import random
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch
import torch.optim as optim
from mmcv.parallel.data_container import DataContainer
from mmcv.runner import load_checkpoint
from mmcv.utils.config import ConfigDict

from sam.datasets import build_dataloader
from sam.datasets.dataset3dsam import Dataset3dsam
from sam.models.frameworks.sam import Sam

TRAIN_DATA_DIR = "data/quadra_fine_tune/train"
TRAIN_INDEX_FILE = "data/quadra_fine_tune/train_filename.csv"
VAL_DATA_DIR = "data/quadra_fine_tune/val"
VAL_INDEX_FILE = "data/quadra_fine_tune/val_filename.csv"

PRETRAINED_CHECKPOINT = "checkpoints/SAM.pth"
OUTPUT_DIR = "checkpoints/quadra_fine_tune_train_loop"
NUM_EPOCHS = 60
LEARNING_RATE = 3e-5
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 1e-4
MIN_LEARNING_RATE = 1e-6
EARLY_STOPPING_PATIENCE = 20
CHECKPOINT_SAVE_INTERVAL = 10
SAMPLES_PER_GPU = 8
WORKERS_PER_GPU = 8
SEED = 42
LOSS_HISTORY_CSV = "loss_history.csv"
TRAINING_SUMMARY_FILE = "training_summary.txt"


def build_train_cfg() -> ConfigDict:
    return ConfigDict(
        {
            "pre_select_pos_number": 2000,
            "after_select_pos_number": 100,
            "pre_select_neg_number": 2000,
            "after_select_neg_number": 500,
            "positive_distance": 2.0,
            "ignore_distance": 20.0,
            "coarse_positive_distance": 25.0,
            "coarse_ignore_distance": 5.0,
            "coarse_z_thres": 6.0,
            "coarse_pre_select_neg_number": 250,
            "coarse_after_select_neg_number": 200,
            "coarse_global_select_number": 1000,
            "temperature": 0.5,
        }
    )


def build_train_pipeline() -> list:
    view1_pipeline = [
        {"type": "ExtraAttrs", "tag": "view1"},
        {"type": "Crop"},
        {"type": "Resample"},
        {"type": "Crop", "switch": "fix"},
        {"type": "RescaleIntensity"},
        {"type": "RandomNoise3d"},
        {"type": "GenerateMeshGrid"},
        {"type": "GenerateMetaInfo"},
        {"type": "DefaultFormatBundle3d"},
        {
            "type": "Collect3d",
            "keys": ["img", "meshgrid", "valid"],
            "meta_keys": ("filename", "tag", "crop_info"),
        },
    ]
    view2_pipeline = [
        {"type": "ExtraAttrs", "tag": "view2"},
        {"type": "Crop"},
        {"type": "Resample"},
        {"type": "Crop", "switch": "fix"},
        {"type": "RescaleIntensity"},
        {"type": "RandomNoise3d"},
        {"type": "GenerateMeshGrid"},
        {"type": "GenerateMetaInfo"},
        {"type": "DefaultFormatBundle3d"},
        {
            "type": "Collect3d",
            "keys": ["img", "meshgrid", "valid"],
            "meta_keys": ("filename", "tag", "crop_info"),
        },
    ]
    return [
        {"type": "LoadTioImage"},
        {"type": "CropBackground"},
        {"type": "ComputeAugParam_sample"},
        {"type": "MultiBranch", "view1": view1_pipeline, "view2": view2_pipeline},
    ]


def build_val_pipeline() -> list:
    view1_pipeline = [
        {"type": "ExtraAttrs", "tag": "view1"},
        {"type": "Resample"},
        {"type": "Crop", "switch": "fix"},
        {"type": "RescaleIntensity"},
        {"type": "GenerateMeshGrid"},
        {"type": "GenerateMetaInfo"},
        {"type": "DefaultFormatBundle3d"},
        {
            "type": "Collect3d",
            "keys": ["img", "meshgrid", "valid"],
            "meta_keys": ("filename", "tag"),
        },
    ]
    view2_pipeline = [
        {"type": "ExtraAttrs", "tag": "view2"},
        {"type": "Resample"},
        {"type": "Crop", "switch": "fix"},
        {"type": "RescaleIntensity"},
        {"type": "GenerateMeshGrid"},
        {"type": "GenerateMetaInfo"},
        {"type": "DefaultFormatBundle3d"},
        {
            "type": "Collect3d",
            "keys": ["img", "meshgrid", "valid"],
            "meta_keys": ("filename", "tag"),
        },
    ]
    return [
        {"type": "LoadTioImage"},
        {"type": "CropBackground"},
        {"type": "MultiBranch", "view1": view1_pipeline, "view2": view2_pipeline},
    ]


def build_model(device: torch.device) -> Sam:
    backbone = {
        "type": "ResNet3d",
        "pretrained2d": True,
        "pretrained": "torchvision://resnet18",
        "depth": 18,
        "in_channels": 1,
        "spatial_strides": (2, 2, 2, 2),
        "temporal_strides": (1, 1, 1, 2),
        "conv1_kernel": (3, 7, 7),
        "conv1_stride_t": 1,
        "conv1_stride_s": 1,
        "pool1_stride_t": 1,
        "with_pool1": False,
        "with_pool2": True,
        "conv_cfg": {"type": "Conv3d"},
        "inflate": ((0, 0), (0, 0), (1, 1), (1, 1)),
        "norm_eval": False,
        "zero_init_residual": False,
    }
    neck = {
        "type": "FPN3d",
        "end_level": 3,
        "in_channels": [64, 128, 256],
        "out_channels": 128,
        "num_outs": 3,
        "conv_cfg": {"type": "Conv3d"},
    }
    read_out_head = {
        "type": "FPN3d",
        "end_level": 1,
        "in_channels": [512],
        "out_channels": 128,
        "num_outs": 1,
        "conv_cfg": {"type": "Conv3d"},
    }
    train_cfg = build_train_cfg()
    test_cfg = {"save_path": "/data/results/result-dlt/", "output_embedding": True}

    model = Sam(
        backbone=backbone,
        neck=neck,
        read_out_head=read_out_head,
        train_cfg=train_cfg,
        test_cfg=test_cfg,
    )

    load_checkpoint(model, PRETRAINED_CHECKPOINT, map_location="cuda")
    model = model.to(device).float()
    return model


def build_datasets() -> Tuple[Dataset3dsam, Dataset3dsam]:
    train_dataset = Dataset3dsam(
        data_dir=TRAIN_DATA_DIR,
        index_file=TRAIN_INDEX_FILE,
        pipeline=build_train_pipeline(),
    )
    val_dataset = Dataset3dsam(
        data_dir=VAL_DATA_DIR,
        index_file=VAL_INDEX_FILE,
        pipeline=build_val_pipeline(),
    )
    return train_dataset, val_dataset


def build_dataloaders(
    train_dataset: Dataset3dsam, val_dataset: Dataset3dsam
) -> Tuple[Any, Any]:
    train_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=SAMPLES_PER_GPU,
        workers_per_gpu=WORKERS_PER_GPU,
        num_gpus=1,
        dist=False,
        shuffle=True,
        seed=SEED,
        persistent_workers=True,
    )
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=SAMPLES_PER_GPU,
        workers_per_gpu=WORKERS_PER_GPU,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=SEED,
        persistent_workers=True,
    )
    return train_loader, val_loader


def unbox(dc: Any) -> Any:
    return dc.data if isinstance(dc, DataContainer) else dc


def _extract_tensor_batch(value: Any, name: str, device: torch.device) -> torch.Tensor:
    data = unbox(value)
    if isinstance(data, (list, tuple)):
        if len(data) != 1:
            raise ValueError(f"Unexpected container length for '{name}': {len(data)}")
        data = data[0]
    if not torch.is_tensor(data):
        raise TypeError(f"Expected tensor for '{name}', got {type(data)}")
    return data.to(device=device, dtype=torch.float32)


def _flatten_img_metas(value: Any) -> List[Dict[str, Any]]:
    data = unbox(value)
    flattened: List[Dict[str, Any]] = []

    def _flatten(item: Any) -> None:
        if isinstance(item, dict):
            flattened.append(item)
            return
        if isinstance(item, (list, tuple)):
            for sub in item:
                _flatten(sub)
            return
        raise TypeError(f"Unexpected img_metas item type: {type(item)}")

    _flatten(data)
    return flattened


def prepare_loader_batch(
    batch: Mapping[str, Any], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
    required_keys = {"img", "meshgrid", "valid", "img_metas"}
    missing = required_keys.difference(set(batch.keys()))
    if missing:
        raise KeyError(f"Missing required keys in loader batch: {sorted(missing)}")

    batch_img = _extract_tensor_batch(batch["img"], "img", device)
    batch_meshgrid = _extract_tensor_batch(batch["meshgrid"], "meshgrid", device)
    batch_valid = _extract_tensor_batch(batch["valid"], "valid", device)
    batch_metas = _flatten_img_metas(batch["img_metas"])

    if batch_img.shape[0] % 2 != 0:
        raise ValueError(
            f"Batch first dimension must be even for SAM pair loss, got {batch_img.shape[0]}"
        )
    if batch_meshgrid.shape[0] != batch_img.shape[0] or batch_valid.shape[0] != batch_img.shape[0]:
        raise ValueError("img, meshgrid, and valid must share the same batch dimension")
    if len(batch_metas) != batch_img.shape[0]:
        raise ValueError(
            "Number of img_metas entries must match batch size: "
            f"{len(batch_metas)} vs {batch_img.shape[0]}"
        )

    return batch_img, batch_meshgrid, batch_valid, batch_metas


def loss_to_scalar(losses: Dict[str, torch.Tensor]) -> torch.Tensor:
    return sum(loss.mean() for loss in losses.values())


def train_one_epoch(
    model: Sam,
    loader: Any,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0
    num_steps = 0

    for step, batch in enumerate(loader, start=1):
        batch_img, batch_meshgrid, batch_valid, batch_metas = prepare_loader_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        losses = model.forward_train(
            img=batch_img,
            img_metas=batch_metas,
            meshgrid=batch_meshgrid,
            valid=batch_valid,
        )
        total_loss = loss_to_scalar(losses)
        total_loss.backward()
        optimizer.step()

        loss_value = total_loss.item()
        running_loss += loss_value
        num_steps += 1
        print(
            f"[train] epoch={epoch:03d} step={step:04d}/{len(loader):04d} "
            f"loss={loss_value:.6f}"
        )

    return running_loss / max(num_steps, 1)


def validate_one_epoch(
    model: Sam, loader: Any, device: torch.device, epoch: int
) -> float:
    model.eval()
    running_loss = 0.0
    num_steps = 0

    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            batch_img, batch_meshgrid, batch_valid, batch_metas = prepare_loader_batch(
                batch, device
            )
            losses = model.forward_train(
                img=batch_img,
                img_metas=batch_metas,
                meshgrid=batch_meshgrid,
                valid=batch_valid,
            )
            total_loss = loss_to_scalar(losses)
            loss_value = total_loss.item()
            running_loss += loss_value
            num_steps += 1
            print(
                f"[val]   epoch={epoch:03d} step={step:04d}/{len(loader):04d} "
                f"loss={loss_value:.6f}"
            )

    return running_loss / max(num_steps, 1)


def save_checkpoint(
    path: str,
    epoch: int,
    model: Sam,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    train_loss: float,
    val_loss: float,
) -> None:
    checkpoint = {
        "meta": {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        },
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(checkpoint, path)


def write_loss_history_csv(path: str, history: List[Dict[str, Any]]) -> None:
    fieldnames = ["epoch", "lr", "train_loss", "val_loss", "is_best"]
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            writer.writerow(
                {
                    "epoch": record["epoch"],
                    "lr": f"{record['lr']:.8f}",
                    "train_loss": f"{record['train_loss']:.6f}",
                    "val_loss": f"{record['val_loss']:.6f}",
                    "is_best": str(record["is_best"]).lower(),
                }
            )


def build_fine_tune_summary(
    train_dataset_size: int,
    val_dataset_size: int,
    train_steps_per_epoch: int,
    val_steps_per_epoch: int,
) -> List[str]:
    pipeline_types = [step["type"] for step in build_train_pipeline()]
    train_cfg = build_train_cfg()

    summary_lines = [
        "Fine-Tune Parameters",
        f"Pretrained checkpoint: {PRETRAINED_CHECKPOINT}",
        f"Train data dir: {TRAIN_DATA_DIR}",
        f"Train index file: {TRAIN_INDEX_FILE}",
        f"Val data dir: {VAL_DATA_DIR}",
        f"Val index file: {VAL_INDEX_FILE}",
        f"Output dir: {OUTPUT_DIR}",
        f"Train dataset size: {train_dataset_size}",
        f"Val dataset size: {val_dataset_size}",
        f"Train steps per epoch: {train_steps_per_epoch}",
        f"Val steps per epoch: {val_steps_per_epoch}",
        f"Num epochs: {NUM_EPOCHS}",
        f"Learning rate: {LEARNING_RATE:.8f}",
        f"Adam betas: {ADAM_BETAS}",
        f"Adam eps: {ADAM_EPS}",
        f"Weight decay: {WEIGHT_DECAY}",
        f"Min learning rate: {MIN_LEARNING_RATE:.8f}",
        f"Early stopping patience: {EARLY_STOPPING_PATIENCE}",
        f"Checkpoint save interval: {CHECKPOINT_SAVE_INTERVAL}",
        f"Samples per GPU: {SAMPLES_PER_GPU}",
        f"Workers per GPU: {WORKERS_PER_GPU}",
        f"Seed: {SEED}",
        f"Pipeline: {' -> '.join(pipeline_types)}",
        "SAM train_cfg:",
    ]
    summary_lines.extend(f"  - {key}: {value}" for key, value in train_cfg.items())
    return summary_lines


def write_training_summary(
    path: str,
    fine_tune_summary_lines: List[str],
    best_record: Dict[str, Any],
    last_record: Dict[str, Any],
    best_checkpoint_path: str,
    last_checkpoint_path: str,
) -> None:
    summary_lines = fine_tune_summary_lines + [
        "",
        "Training Summary",
        f"Best epoch: {best_record['epoch']}",
        f"Best train loss: {best_record['train_loss']:.6f}",
        f"Best val loss: {best_record['val_loss']:.6f}",
        f"Best learning rate: {best_record['lr']:.8f}",
        f"Best checkpoint: {best_checkpoint_path}",
        "",
        f"Last epoch: {last_record['epoch']}",
        f"Last train loss: {last_record['train_loss']:.6f}",
        f"Last val loss: {last_record['val_loss']:.6f}",
        f"Last learning rate: {last_record['lr']:.8f}",
        f"Last checkpoint: {last_checkpoint_path}",
    ]

    with open(path, "w") as summary_file:
        summary_file.write("\n".join(summary_lines) + "\n")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script, but no GPU was found.")
    device = torch.device("cuda")
    print(f"[setup] device={device}")
    print(f"[setup] samples_per_gpu={SAMPLES_PER_GPU} workers_per_gpu={WORKERS_PER_GPU}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    train_dataset, val_dataset = build_datasets()
    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty.")
    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty.")
    print(f"[setup] train samples={len(train_dataset)}")
    print(f"[setup] val samples={len(val_dataset)}")
    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset)
    print(f"[setup] train steps/epoch={len(train_loader)}")
    print(f"[setup] val steps/epoch={len(val_loader)}")
    fine_tune_summary_lines = build_fine_tune_summary(
        train_dataset_size=len(train_dataset),
        val_dataset_size=len(val_dataset),
        train_steps_per_epoch=len(train_loader),
        val_steps_per_epoch=len(val_loader),
    )
    for line in fine_tune_summary_lines:
        print(f"[setup] {line}")

    first_train_batch = next(iter(train_loader))
    train_img, train_mesh, train_valid, train_metas = prepare_loader_batch(first_train_batch, device)
    print(
        "[setup] train first batch shapes "
        f"img={tuple(train_img.shape)} meshgrid={tuple(train_mesh.shape)} "
        f"valid={tuple(train_valid.shape)} img_metas={len(train_metas)}"
    )
    del first_train_batch, train_img, train_mesh, train_valid, train_metas

    first_val_batch = next(iter(val_loader))
    val_img, val_mesh, val_valid, val_metas = prepare_loader_batch(first_val_batch, device)
    print(
        "[setup] val first batch shapes "
        f"img={tuple(val_img.shape)} meshgrid={tuple(val_mesh.shape)} "
        f"valid={tuple(val_valid.shape)} img_metas={len(val_metas)}"
    )
    del first_val_batch, val_img, val_mesh, val_valid, val_metas

    model = build_model(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=MIN_LEARNING_RATE,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    no_improve_epochs = 0
    best_record: Dict[str, Any] | None = None
    history: List[Dict[str, Any]] = []
    loss_history_path = os.path.join(OUTPUT_DIR, LOSS_HISTORY_CSV)
    summary_path = os.path.join(OUTPUT_DIR, TRAINING_SUMMARY_FILE)
    best_path = os.path.join(OUTPUT_DIR, "best.pth")
    last_path = os.path.join(OUTPUT_DIR, "last.pth")

    for epoch in range(1, NUM_EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n[epoch] {epoch}/{NUM_EPOCHS}")
        print(f"[lr] epoch={epoch:03d} lr={current_lr:.8f}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate_one_epoch(model, val_loader, device, epoch)

        print(
            f"[summary] epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )

        save_checkpoint(last_path, epoch, model, optimizer, scheduler, train_loss, val_loss)
        print(f"[checkpoint] saved last -> {last_path}")
        if epoch % CHECKPOINT_SAVE_INTERVAL == 0:
            interval_checkpoint_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch:03d}.pth")
            save_checkpoint(
                interval_checkpoint_path,
                epoch,
                model,
                optimizer,
                scheduler,
                train_loss,
                val_loss,
            )
            print(f"[checkpoint] saved interval -> {interval_checkpoint_path}")

        epoch_record = {
            "epoch": epoch,
            "lr": float(current_lr),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "is_best": False,
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            save_checkpoint(best_path, epoch, model, optimizer, scheduler, train_loss, val_loss)
            epoch_record["is_best"] = True
            best_record = dict(epoch_record)
            print(f"[checkpoint] new best val_loss={best_val_loss:.6f} -> {best_path}")
        else:
            no_improve_epochs += 1
            print(
                f"[early-stop] no improvement for {no_improve_epochs}/"
                f"{EARLY_STOPPING_PATIENCE} epoch(s)"
            )

        history.append(epoch_record)
        write_loss_history_csv(loss_history_path, history)

        if best_record is None:
            best_record = dict(epoch_record)
        write_training_summary(
            summary_path,
            fine_tune_summary_lines,
            best_record,
            history[-1],
            best_path,
            last_path,
        )

        if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
            print(
                f"[early-stop] stopping at epoch {epoch} "
                f"(best_val_loss={best_val_loss:.6f})"
            )
            break

        scheduler.step()

    if not history or best_record is None:
        raise RuntimeError("Training finished without any recorded epochs.")

    last_record = history[-1]
    print("\n[final-summary]")
    print(
        f"best_epoch={best_record['epoch']:03d} "
        f"train_loss={best_record['train_loss']:.6f} "
        f"val_loss={best_record['val_loss']:.6f} "
        f"checkpoint={best_path}"
    )
    print(
        f"last_epoch={last_record['epoch']:03d} "
        f"train_loss={last_record['train_loss']:.6f} "
        f"val_loss={last_record['val_loss']:.6f} "
        f"checkpoint={last_path}"
    )
    print(f"loss_history_csv={loss_history_path}")
    print(f"training_summary={summary_path}")


if __name__ == "__main__":
    main()
