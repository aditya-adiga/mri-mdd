"""Dataset module for loading and preprocessing 3D MRI volumes.

Provides MRIDataset for loading NIfTI files and a helper for
constructing a WeightedRandomSampler to handle class imbalance.
"""

import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


class MRIDataset(Dataset):
    """PyTorch Dataset for 3D grey matter MRI volumes.

    Loads NIfTI files, normalizes voxel intensities to zero mean and unit
    variance per volume, and returns (tensor, label) pairs.

    Args:
        data_dir: Path to directory containing NIfTI files.
        labels_csv: Path to CSV with columns 'subject_id' and 'label'.
        subject_ids: Optional list of subject IDs to include (for splits).
    """

    def __init__(
        self,
        data_dir: Path,
        labels_csv: Path,
        subject_ids: list[str] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        labels_df = pd.read_csv(labels_csv, dtype={"subject_id": str})

        if subject_ids is not None:
            labels_df = labels_df[labels_df["subject_id"].isin(subject_ids)]

        # Filter out rows whose NIfTI files are missing
        self.samples: list[tuple[Path, int]] = []
        for _, row in labels_df.iterrows():
            file_path = self._find_nifti(row["subject_id"])
            if file_path is not None:
                self.samples.append((file_path, int(row["label"])))
            else:
                logger.warning(
                    "NIfTI file not found for subject %s, skipping",
                    row["subject_id"],
                )

        logger.info(
            "MRIDataset initialised with %d samples from %s",
            len(self.samples),
            self.data_dir,
        )

    def _find_nifti(self, subject_id: str) -> Path | None:
        """Locate a NIfTI file for a given subject ID.

        Args:
            subject_id: The subject identifier to search for.

        Returns:
            Path to the NIfTI file, or None if not found.
        """
        for ext in (".nii.gz", ".nii"):
            candidate = self.data_dir / f"{subject_id}{ext}"
            if candidate.exists():
                return candidate
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and return a single MRI volume with its label.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple of (tensor, label) where tensor has shape (1, 121, 145, 121)
            and label is a torch.long scalar.
        """
        file_path, label = self.samples[idx]
        img = nib.load(file_path)
        data = img.get_fdata(dtype=np.float32)

        # Normalize brain voxels to zero mean and unit variance
        # Background (zero) voxels are left at zero to avoid
        # dominating the signal (~66% of volume is background)
        brain_mask = data > 0
        if brain_mask.any():
            brain_vals = data[brain_mask]
            mean = brain_vals.mean()
            std = brain_vals.std()
            if std > 0:
                data[brain_mask] = (brain_vals - mean) / std
            else:
                data[brain_mask] = brain_vals - mean

        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        tensor = torch.from_numpy(data).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return tensor, label_tensor

    @property
    def labels(self) -> list[int]:
        """Return list of all labels in the dataset."""
        return [label for _, label in self.samples]


def get_weighted_sampler(dataset: MRIDataset) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler for class-imbalanced datasets.

    Computes per-class weights as N_total / (2 * N_class) and assigns
    each sample the weight of its class.

    Args:
        dataset: An MRIDataset instance.

    Returns:
        A WeightedRandomSampler for use in a DataLoader.
    """
    labels = dataset.labels
    n_total = len(labels)
    class_counts = np.bincount(labels)
    class_weights = n_total / (2.0 * class_counts)

    sample_weights = torch.tensor(
        [class_weights[label] for label in labels], dtype=torch.float64
    )

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=n_total,
        replacement=True,
    )
