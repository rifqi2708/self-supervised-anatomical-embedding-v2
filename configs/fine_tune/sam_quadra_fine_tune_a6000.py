_base_ = "../sam/sam_NIHLN.py"

# Fine-tune from pretrained SAM weights instead of training from scratch.
load_from = "checkpoints/SAM.pth"
resume_from = None

# Keep the same model architecture/training settings from the base config,
# and only override test export behavior for post-validation.
model = dict(
    test_cfg=dict(
        save_path="work_dirs/sam_quadra_fine_tune_a6000/post_val_embeddings/",
        output_embedding=False,
    )
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        data_dir="data/quadra_fine_tune/train",
        index_file="data/quadra_fine_tune/train_filename.csv",
        pipeline=train_pipeline,
    ),
    val=dict(
        data_dir="data/quadra_fine_tune/val",
        index_file="data/quadra_fine_tune/val_filename.csv",
        pipeline=test_pipeline,
    ),
    # Post-validation uses tools/test_sam.py against cfg.data.test.
    test=dict(
        data_dir="data/quadra_fine_tune/val",
        index_file="data/quadra_fine_tune/val_filename.csv",
        pipeline=test_pipeline,
    ),
)
