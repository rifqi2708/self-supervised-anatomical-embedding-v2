import os
from typing import Any, Dict, Tuple

import torch
import torch.optim as optim
from mmcv.parallel.data_container import DataContainer
from mmcv.runner import load_checkpoint
from mmcv.utils.config import ConfigDict

from sam.datasets.dataset3dsam import Dataset3dsam
from sam.models.frameworks.sam import Sam

TRAIN_DATA_DIR = "data/quadra_fine_tune/train"
TRAIN_INDEX_FILE = "data/quadra_fine_tune/train_filename.csv"
VAL_DATA_DIR = "data/quadra_fine_tune/val"
VAL_INDEX_FILE = "data/quadra_fine_tune/val_filename.csv"

PRETRAINED_CHECKPOINT = "checkpoints/SAM.pth"
OUTPUT_DIR = "checkpoints/quadra_fine_tune_train_loop"
NUM_EPOCHS = 60
LEARNING_RATE = 5e-5
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 1e-4
MIN_LEARNING_RATE = 1e-6
EARLY_STOPPING_PATIENCE = 10


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
    train_cfg = ConfigDict(
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
        pipeline=build_train_pipeline(),
    )
    return train_dataset, val_dataset


def unbox(dc: Any) -> Any:
    return dc.data if isinstance(dc, DataContainer) else dc


def prepare_pair_batch(
    sample_pair: Tuple[Dict[str, Any], Dict[str, Any]], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
    data1, data2 = sample_pair

    batch_img = torch.stack(
        [unbox(data1["img"]), unbox(data2["img"])], dim=0
    ).to(device=device, dtype=torch.float32)
    batch_meshgrid = torch.stack(
        [unbox(data1["meshgrid"]), unbox(data2["meshgrid"])], dim=0
    ).to(device=device, dtype=torch.float32)
    batch_valid = torch.stack(
        [unbox(data1["valid"]), unbox(data2["valid"])], dim=0
    ).to(device=device, dtype=torch.float32)
    batch_metas = [unbox(data1["img_metas"]), unbox(data2["img_metas"])]
    return batch_img, batch_meshgrid, batch_valid, batch_metas


def loss_to_scalar(losses: Dict[str, torch.Tensor]) -> torch.Tensor:
    return sum(loss.mean() for loss in losses.values())


def train_one_epoch(
    model: Sam,
    dataset: Dataset3dsam,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0
    num_steps = 0

    for step, sample_pair in enumerate(dataset, start=1):
        batch_img, batch_meshgrid, batch_valid, batch_metas = prepare_pair_batch(
            sample_pair, device
        )

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
            f"[train] epoch={epoch:03d} step={step:04d}/{len(dataset):04d} "
            f"loss={loss_value:.6f}"
        )

    return running_loss / max(num_steps, 1)


def validate_one_epoch(
    model: Sam, dataset: Dataset3dsam, device: torch.device, epoch: int
) -> float:
    model.eval()
    running_loss = 0.0
    num_steps = 0

    with torch.no_grad():
        for step, sample_pair in enumerate(dataset, start=1):
            batch_img, batch_meshgrid, batch_valid, batch_metas = prepare_pair_batch(
                sample_pair, device
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
                f"[val]   epoch={epoch:03d} step={step:04d}/{len(dataset):04d} "
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
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, path)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script, but no GPU was found.")
    device = torch.device("cuda")
    print(f"[setup] device={device}")

    train_dataset, val_dataset = build_datasets()
    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty.")
    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty.")
    print(f"[setup] train samples={len(train_dataset)}")
    print(f"[setup] val samples={len(val_dataset)}")

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

    for epoch in range(1, NUM_EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n[epoch] {epoch}/{NUM_EPOCHS}")
        print(f"[lr] epoch={epoch:03d} lr={current_lr:.8f}")
        train_loss = train_one_epoch(model, train_dataset, optimizer, device, epoch)
        val_loss = validate_one_epoch(model, val_dataset, device, epoch)

        print(
            f"[summary] epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )

        last_path = os.path.join(OUTPUT_DIR, "last.pth")
        save_checkpoint(last_path, epoch, model, optimizer, scheduler, train_loss, val_loss)
        print(f"[checkpoint] saved last -> {last_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            best_path = os.path.join(OUTPUT_DIR, "best.pth")
            save_checkpoint(best_path, epoch, model, optimizer, scheduler, train_loss, val_loss)
            print(f"[checkpoint] new best val_loss={best_val_loss:.6f} -> {best_path}")
        else:
            no_improve_epochs += 1
            print(
                f"[early-stop] no improvement for {no_improve_epochs}/"
                f"{EARLY_STOPPING_PATIENCE} epoch(s)"
            )

        if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
            print(
                f"[early-stop] stopping at epoch {epoch} "
                f"(best_val_loss={best_val_loss:.6f})"
            )
            break

        scheduler.step()


if __name__ == "__main__":
    main()
