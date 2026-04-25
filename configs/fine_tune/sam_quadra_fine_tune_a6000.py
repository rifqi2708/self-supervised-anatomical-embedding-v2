_base_ = "../sam/sam_NIHLN.py"  # Inherit original SAM NIH-LN config as the base template.

load_from = "checkpoints/SAM.pth"  # Initialize from pretrained SAM weights (fine-tuning, not scratch training).
resume_from = None  # Do not force resume from a previous run unless overridden at runtime.

optimizer = dict(  # Override optimizer settings for fine-tuning stability on a small dataset.
    type="SGD",  # Keep optimizer family consistent with original SAM training.
    lr=0.002,  # Lower learning rate for fine-tuning (base config uses a higher pretraining LR).
    momentum=0.9,  # Keep momentum from original training setup.
    weight_decay=0.0001,  # Keep weight decay from original training setup.
)  # End optimizer overrides.

runner = dict(  # Override training length to a safer fine-tuning schedule.
    type="EpochBasedRunner",  # Keep runner type aligned with original repo setup.
    max_iters=2000,  # Stop after 2,000 iterations by default (instead of inherited 20,000).
)  # End runner override.

checkpoint_config = dict(  # Control checkpoint save cadence.
    by_epoch=False,  # Save checkpoints by iteration count (not by epoch count).
    interval=100,  # Save every 100 iterations for tighter monitoring during fine-tuning.
    max_keep_ckpts=20,  # Keep at most 20 checkpoints to limit disk growth.
)  # End checkpoint config.

log_config = dict(  # Control logging frequency during training.
    interval=10,  # Write logs every 10 iterations.
    hooks=[dict(type="TextLoggerHook")],  # Use plain text logger hook (same logging style as base runtime).
)  # End log config.

model = dict(  # Override only test/export behavior; keep architecture and loss setup from base config.
    test_cfg=dict(  # Settings used by test/post-validation scripts.
        save_path="work_dirs/sam_quadra_fine_tune_a6000/post_val_embeddings/",  # Where exported val embeddings will be written.
        output_embedding=False,  # False means save embeddings to disk (.pkl) instead of returning tensors in memory.
    )  # End test_cfg overrides.
)  # End model overrides.

data = dict(  # DataLoader and dataset split configuration for fine-tuning.
    samples_per_gpu=8,  # Balanced batch size target for RTX A6000 (adjust at runtime if needed).
    workers_per_gpu=8,  # Number of dataloader workers per GPU.
    train=dict(  # Training split.
        data_dir="data/quadra_fine_tune/train",  # Root folder that contains training NIfTI files.
        index_file="data/quadra_fine_tune/train_filename.csv",  # CSV index listing training relative file paths.
        # Pipeline is inherited from the base config.
    ),  # End train split config.
    val=dict(  # Validation split metadata (kept for completeness, even with --no-validate training).
        data_dir="data/quadra_fine_tune/val",  # Root folder that contains validation NIfTI files.
        index_file="data/quadra_fine_tune/val_filename.csv",  # CSV index listing validation relative file paths.
        # Pipeline is inherited from the base config.
    ),  # End val split config.
    test=dict(  # Post-validation uses cfg.data.test via tools/test_sam.py.
        data_dir="data/quadra_fine_tune/val",  # Point test split to val data for post-validation export.
        index_file="data/quadra_fine_tune/val_filename.csv",  # Reuse val CSV for post-validation runs.
        # Pipeline is inherited from the base config.
    ),  # End test split config.
)  # End data config.
