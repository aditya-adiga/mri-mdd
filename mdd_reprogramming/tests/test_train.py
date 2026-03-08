"""Tests for the training pipeline.

All tests use synthetic data and mock the HuggingFace model download.
"""

import csv
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch
import torch.nn as nn

from mdd_reprogramming.train import (
    compute_metrics,
    evaluate,
    main,
    train_one_epoch,
)

VOLUME_SHAPE = (121, 145, 121)


@pytest.fixture()
def synthetic_data_dir(tmp_path: Path) -> tuple[Path, Path]:
    """Create synthetic NIfTI files and labels CSV.

    Returns:
        Tuple of (data_dir, labels_csv_path).
    """
    nifti_dir = tmp_path / "nifti"
    nifti_dir.mkdir()

    csv_path = tmp_path / "labels.csv"
    n_subjects = 10

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

    return nifti_dir, csv_path


class TestComputeMetrics:
    """Tests for the compute_metrics function."""

    def test_perfect_predictions(self) -> None:
        """Perfect predictions yield all metrics = 1.0."""
        labels = np.array([0, 0, 1, 1])
        preds = np.array([0, 0, 1, 1])
        probs = np.array([0.1, 0.2, 0.9, 0.8])
        metrics = compute_metrics(labels, preds, probs)
        assert metrics["Accuracy"] == 1.0
        assert metrics["Sensitivity"] == 1.0
        assert metrics["Specificity"] == 1.0
        assert metrics["F1"] == 1.0
        assert metrics["AUROC"] == 1.0
        assert metrics["AUPRC"] == 1.0

    def test_returns_all_keys(self) -> None:
        """Metrics dict contains all expected keys."""
        labels = np.array([0, 1])
        preds = np.array([1, 0])
        probs = np.array([0.6, 0.4])
        metrics = compute_metrics(labels, preds, probs)
        expected_keys = {"Accuracy", "Sensitivity", "Specificity", "F1", "AUROC", "AUPRC"}
        assert set(metrics.keys()) == expected_keys


class TestTrainOneEpoch:
    """Tests for the train_one_epoch function."""

    def test_returns_float(self) -> None:
        """train_one_epoch returns a float loss value."""
        # Simple model
        model = nn.Linear(4, 2)
        loader = [
            (torch.randn(2, 4), torch.tensor([0, 1])),
            (torch.randn(2, 4), torch.tensor([1, 0])),
        ]
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        device = torch.device("cpu")

        # Wrap loader items to device
        class FakeLoader:
            def __iter__(self):
                return iter(loader)

        avg_loss = train_one_epoch(model, FakeLoader(), loss_fn, optimizer, device)
        assert isinstance(avg_loss, float)
        assert avg_loss > 0


class TestSmokeTraining:
    """Smoke tests for the full training pipeline."""

    @patch("mdd_reprogramming.model.AutoModel.from_pretrained")
    def test_smoke_baseline_2_epochs(
        self,
        mock_from_pretrained: MagicMock,
        synthetic_data_dir: tuple[Path, Path],
        tmp_path: Path,
    ) -> None:
        """Smoke test: 2 epochs on 10 synthetic samples in baseline mode completes."""
        nifti_dir, csv_path = synthetic_data_dir
        ckpt_dir = tmp_path / "checkpoints"

        main([
            "--data_dir", str(nifti_dir),
            "--labels_csv", str(csv_path),
            "--checkpoint_dir", str(ckpt_dir),
            "--epochs", "2",
            "--batch_size", "2",
            "--n_folds", "2",
            "--baseline",
            "--no_wandb",
            "--seed", "42",
        ])

        # Verify checkpoints were saved in timestamped run dir
        assert ckpt_dir.exists()
        run_dirs = sorted(ckpt_dir.iterdir())
        assert len(run_dirs) == 1
        latest_run = run_dirs[0]
        fold_dirs = list(latest_run.glob("fold_*"))
        assert len(fold_dirs) >= 1
        assert (fold_dirs[0] / "best.pt").exists()

    @patch("mdd_reprogramming.model.AutoModel.from_pretrained")
    def test_optimizer_only_trainable_params(
        self,
        mock_from_pretrained: MagicMock,
    ) -> None:
        """Optimizer only contains parameters with requires_grad=True."""
        from mdd_reprogramming.model import MDDReprogrammingModel

        # Create a fake LLM
        fake_llm = MagicMock(spec=nn.Module)
        fake_llm.parameters.return_value = [
            nn.Parameter(torch.randn(4, 4), requires_grad=True)
        ]
        fake_llm.eval = MagicMock(return_value=fake_llm)
        mock_from_pretrained.return_value = fake_llm

        model = MDDReprogrammingModel(
            patch_size=16, d_model=128, d_llm=1024, baseline=False,
        )

        # Simulate what train.py does
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=0.001)

        # All params in optimizer should have requires_grad=True
        for group in optimizer.param_groups:
            for p in group["params"]:
                assert p.requires_grad is True

        # LLM params should be frozen
        for p in model.llm.parameters():
            assert p.requires_grad is False

    @patch("mdd_reprogramming.model.AutoModel.from_pretrained")
    def test_checkpoint_saved(
        self,
        mock_from_pretrained: MagicMock,
        synthetic_data_dir: tuple[Path, Path],
        tmp_path: Path,
    ) -> None:
        """A checkpoint file is saved after training."""
        nifti_dir, csv_path = synthetic_data_dir
        ckpt_dir = tmp_path / "checkpoints"

        main([
            "--data_dir", str(nifti_dir),
            "--labels_csv", str(csv_path),
            "--checkpoint_dir", str(ckpt_dir),
            "--epochs", "1",
            "--batch_size", "5",
            "--n_folds", "2",
            "--baseline",
            "--no_wandb",
            "--seed", "42",
        ])

        run_dirs = sorted(ckpt_dir.iterdir())
        latest_run = run_dirs[0]
        fold_dirs = sorted(latest_run.glob("fold_*"))
        assert len(fold_dirs) == 2

        # Verify checkpoint can be loaded
        state_dict = torch.load(fold_dirs[0] / "best.pt", map_location="cpu", weights_only=True)
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

        # Verify config.json was saved
        assert (latest_run / "config.json").exists()
