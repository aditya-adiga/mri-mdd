"""Tests for the standalone evaluation script."""

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from mdd_reprogramming.config import parse_args
from mdd_reprogramming.evaluate import main as eval_main, run_evaluation
from mdd_reprogramming.model import MDDReprogrammingModel

VOLUME_SHAPE = (121, 145, 121)


@pytest.fixture()
def eval_setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create synthetic data and a saved baseline checkpoint.

    Returns:
        Tuple of (data_dir, labels_csv, checkpoint_path).
    """
    nifti_dir = tmp_path / "nifti"
    nifti_dir.mkdir()

    n_subjects = 10
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "label"])
        for i in range(n_subjects):
            sid = f"sub_{i:03d}"
            label = 1 if i < 6 else 0
            writer.writerow([sid, label])
            data = np.random.randn(*VOLUME_SHAPE).astype(np.float32)
            img = nib.Nifti1Image(data, affine=np.eye(4))
            nib.save(img, nifti_dir / f"{sid}.nii.gz")

    # Create and save a baseline model checkpoint
    model = MDDReprogrammingModel(
        patch_size=16, d_model=128, d_llm=1024, baseline=True,
    )
    ckpt_path = tmp_path / "test_checkpoint.pt"
    torch.save(model.state_dict(), ckpt_path)

    return nifti_dir, csv_path, ckpt_path


class TestEvalConfig:
    """Tests for evaluation argument parsing."""

    def test_checkpoint_arg_parsed(self) -> None:
        """--checkpoint argument is parsed correctly."""
        cfg = parse_args([
            "--checkpoint", "/tmp/model.pt",
            "--baseline",
            "--no_wandb",
        ])
        assert cfg.checkpoint == Path("/tmp/model.pt")
        assert cfg.baseline is True

    def test_checkpoint_default_none(self) -> None:
        """Checkpoint defaults to None."""
        cfg = parse_args([])
        assert cfg.checkpoint is None

    def test_missing_checkpoint_raises(self) -> None:
        """run_evaluation raises ValueError when --checkpoint is missing."""
        with pytest.raises(ValueError, match="--checkpoint is required"):
            run_evaluation(["--no_wandb"])


class TestRunEvaluation:
    """Tests for the run_evaluation function."""

    def test_returns_all_metrics(self, eval_setup: tuple[Path, Path, Path]) -> None:
        """Evaluation returns all expected metric keys."""
        nifti_dir, csv_path, ckpt_path = eval_setup
        metrics = eval_main([
            "--data_dir", str(nifti_dir),
            "--labels_csv", str(csv_path),
            "--checkpoint", str(ckpt_path),
            "--baseline",
            "--no_wandb",
            "--batch_size", "5",
        ])
        expected_keys = {"Accuracy", "Sensitivity", "Specificity", "F1", "AUROC", "AUPRC"}
        assert set(metrics.keys()) == expected_keys

    def test_metrics_in_valid_range(self, eval_setup: tuple[Path, Path, Path]) -> None:
        """All metrics are between 0 and 1."""
        nifti_dir, csv_path, ckpt_path = eval_setup
        metrics = eval_main([
            "--data_dir", str(nifti_dir),
            "--labels_csv", str(csv_path),
            "--checkpoint", str(ckpt_path),
            "--baseline",
            "--no_wandb",
            "--batch_size", "5",
        ])
        for name, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{name}={value} out of [0,1] range"

    def test_reproducible_with_seed(self, eval_setup: tuple[Path, Path, Path]) -> None:
        """Same seed produces same test split and metrics."""
        nifti_dir, csv_path, ckpt_path = eval_setup
        common_args = [
            "--data_dir", str(nifti_dir),
            "--labels_csv", str(csv_path),
            "--checkpoint", str(ckpt_path),
            "--baseline",
            "--no_wandb",
            "--batch_size", "5",
            "--seed", "42",
        ]
        metrics_1 = eval_main(common_args)
        metrics_2 = eval_main(common_args)
        for key in metrics_1:
            assert metrics_1[key] == metrics_2[key], f"{key} differs across runs"
