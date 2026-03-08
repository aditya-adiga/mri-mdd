"""Tests for the dataset module."""

import csv
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from mdd_reprogramming.dataset import MRIDataset, get_weighted_sampler

VOLUME_SHAPE = (121, 145, 121)


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> MRIDataset:
    """Create a synthetic dataset with 6 NIfTI files and a labels CSV.

    3 MDD (label=1) and 3 HC (label=0).
    """
    nifti_dir = tmp_path / "nifti"
    nifti_dir.mkdir()

    subjects = []
    for i in range(6):
        subject_id = f"sub_{i:03d}"
        label = 1 if i < 3 else 0
        subjects.append((subject_id, label))

        data = np.random.randn(*VOLUME_SHAPE).astype(np.float32) * 100 + 500
        img = nib.Nifti1Image(data, affine=np.eye(4))
        nib.save(img, nifti_dir / f"{subject_id}.nii.gz")

    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "label"])
        for sid, lbl in subjects:
            writer.writerow([sid, lbl])

    return MRIDataset(data_dir=nifti_dir, labels_csv=csv_path)


class TestMRIDataset:
    """Tests for MRIDataset class."""

    def test_length(self, synthetic_dataset: MRIDataset) -> None:
        """Dataset length matches number of valid samples."""
        assert len(synthetic_dataset) == 6

    def test_tensor_shape(self, synthetic_dataset: MRIDataset) -> None:
        """Returned tensor has shape (1, 121, 145, 121)."""
        tensor, _ = synthetic_dataset[0]
        assert tensor.shape == (1, *VOLUME_SHAPE)

    def test_tensor_dtype(self, synthetic_dataset: MRIDataset) -> None:
        """Returned tensor is float32."""
        tensor, _ = synthetic_dataset[0]
        assert tensor.dtype == torch.float32

    def test_label_dtype(self, synthetic_dataset: MRIDataset) -> None:
        """Label is torch.long scalar."""
        _, label = synthetic_dataset[0]
        assert label.dtype == torch.long
        assert label.dim() == 0

    def test_normalization_mean(self, synthetic_dataset: MRIDataset) -> None:
        """Voxel intensities have approximately zero mean after normalization."""
        tensor, _ = synthetic_dataset[0]
        assert abs(tensor.mean().item()) < 1e-5

    def test_normalization_std(self, synthetic_dataset: MRIDataset) -> None:
        """Voxel intensities have approximately unit std after normalization."""
        tensor, _ = synthetic_dataset[0]
        assert abs(tensor.std().item() - 1.0) < 1e-3

    def test_labels_property(self, synthetic_dataset: MRIDataset) -> None:
        """Labels property returns correct list."""
        labels = synthetic_dataset.labels
        assert len(labels) == 6
        assert set(labels) == {0, 1}

    def test_subject_ids_filter(self, tmp_path: Path) -> None:
        """Dataset filters to only requested subject IDs."""
        nifti_dir = tmp_path / "nifti_filter"
        nifti_dir.mkdir()

        csv_path = tmp_path / "labels_filter.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["subject_id", "label"])
            for i in range(4):
                sid = f"sub_{i:03d}"
                writer.writerow([sid, i % 2])
                data = np.random.randn(*VOLUME_SHAPE).astype(np.float32)
                img = nib.Nifti1Image(data, affine=np.eye(4))
                nib.save(img, nifti_dir / f"{sid}.nii.gz")

        ds = MRIDataset(
            data_dir=nifti_dir,
            labels_csv=csv_path,
            subject_ids=["sub_000", "sub_002"],
        )
        assert len(ds) == 2

    def test_missing_file_handled_gracefully(self, tmp_path: Path) -> None:
        """Missing NIfTI file is skipped with a warning, not an error."""
        nifti_dir = tmp_path / "nifti_missing"
        nifti_dir.mkdir()

        # Create CSV referencing a file that doesn't exist
        csv_path = tmp_path / "labels_missing.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["subject_id", "label"])
            writer.writerow(["nonexistent_sub", 0])
            writer.writerow(["existing_sub", 1])

        # Only create one file
        data = np.random.randn(*VOLUME_SHAPE).astype(np.float32)
        img = nib.Nifti1Image(data, affine=np.eye(4))
        nib.save(img, nifti_dir / "existing_sub.nii.gz")

        ds = MRIDataset(data_dir=nifti_dir, labels_csv=csv_path)
        assert len(ds) == 1  # Only the existing file

    def test_nii_extension_support(self, tmp_path: Path) -> None:
        """Dataset loads both .nii and .nii.gz files."""
        nifti_dir = tmp_path / "nifti_ext"
        nifti_dir.mkdir()

        csv_path = tmp_path / "labels_ext.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["subject_id", "label"])
            writer.writerow(["sub_gz", 0])
            writer.writerow(["sub_plain", 1])

        for name, ext in [("sub_gz", ".nii.gz"), ("sub_plain", ".nii")]:
            data = np.random.randn(*VOLUME_SHAPE).astype(np.float32)
            img = nib.Nifti1Image(data, affine=np.eye(4))
            nib.save(img, nifti_dir / f"{name}{ext}")

        ds = MRIDataset(data_dir=nifti_dir, labels_csv=csv_path)
        assert len(ds) == 2


class TestWeightedSampler:
    """Tests for get_weighted_sampler function."""

    def test_sampler_length(self, synthetic_dataset: MRIDataset) -> None:
        """Sampler num_samples matches dataset length."""
        sampler = get_weighted_sampler(synthetic_dataset)
        assert sampler.num_samples == len(synthetic_dataset)

    def test_balanced_weights(self, tmp_path: Path) -> None:
        """Class weights are correctly computed as N_total / (2 * N_class)."""
        nifti_dir = tmp_path / "nifti_bal"
        nifti_dir.mkdir()

        csv_path = tmp_path / "labels_bal.csv"
        # 8 total: 6 MDD (label=1), 2 HC (label=0)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["subject_id", "label"])
            for i in range(8):
                sid = f"sub_{i:03d}"
                label = 1 if i < 6 else 0
                writer.writerow([sid, label])
                data = np.random.randn(*VOLUME_SHAPE).astype(np.float32)
                img = nib.Nifti1Image(data, affine=np.eye(4))
                nib.save(img, nifti_dir / f"{sid}.nii.gz")

        ds = MRIDataset(data_dir=nifti_dir, labels_csv=csv_path)
        sampler = get_weighted_sampler(ds)

        weights = list(sampler.weights.numpy())
        # HC (label=0): 8 / (2*2) = 2.0, MDD (label=1): 8 / (2*6) ≈ 0.667
        expected_hc = 8.0 / (2 * 2)
        expected_mdd = 8.0 / (2 * 6)

        labels = ds.labels
        for w, lbl in zip(weights, labels):
            if lbl == 0:
                assert abs(w - expected_hc) < 1e-6
            else:
                assert abs(w - expected_mdd) < 1e-6
