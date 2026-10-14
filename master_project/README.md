# Earlier master project workflow

Training/fine-tuning scripts, dedicated fine-tuning configuration and notebooks are grouped here. The current publication workflow uses pretrained models; this folder is not a publication fine-tuning requirement.

- `scripts/`: earlier training loops, evaluation and single-cycle runners.
- `fine_tune/`: existing training and post-validation launchers.
- `configs/`: dedicated Quadra fine-tuning settings inheriting unchanged root model configurations.
- `notebooks/`: training reconstruction and loss-history inspection.
- `shared/`: the reviewed earlier monitor helper; cross-project interfaces stay in root `shared/`.

Run from the repository root in the original scientific environment. Dataset, checkpoint and work-directory defaults are unchanged. Invoke the canonical paths in this layout. Consult root [training reference](../resources/training.md) and preserved script settings before running anything.
