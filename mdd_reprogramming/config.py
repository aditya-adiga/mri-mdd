"""Configuration module for MDD Reprogramming pipeline.

All hyperparameters and paths are managed via argparse.
No hardcoded values should appear elsewhere in the codebase.
"""

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the MDD Reprogramming pipeline.

    Args:
        args: Optional list of argument strings. If None, reads from sys.argv.

    Returns:
        Parsed argument namespace with all hyperparameters.
    """
    parser = argparse.ArgumentParser(
        description="MDD classification via Modality Reprogramming with frozen LLM backbone"
    )

    # Paths
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/wc1_MDD_VBM_dataset"),
        help="Path to directory containing NIfTI files",
    )
    parser.add_argument(
        "--labels_csv",
        type=Path,
        default=Path("data/labels.csv"),
        help="Path to CSV file with subject IDs and labels",
    )

    # Model architecture
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        help="Size of non-overlapping 3D patches",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=128,
        help="Dimensionality of patch encoder output",
    )
    parser.add_argument(
        "--llm_name",
        type=str,
        default="Qwen/Qwen1.5-0.5B",
        help="Hugging Face model name for frozen LLM backbone",
    )
    parser.add_argument(
        "--d_llm",
        type=int,
        default=1024,
        help="Hidden size of the LLM backbone",
    )

    # Training
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for AdamW optimizer",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs per fold",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate in reprogramming layer",
    )

    # Cross-validation & evaluation
    parser.add_argument(
        "--n_folds",
        type=int,
        default=10,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--blind_test_frac",
        type=float,
        default=0.1,
        help="Fraction of data held out as blind test set",
    )

    # Loss
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="ce",
        choices=["ce", "focal"],
        help="Loss function: 'ce' for CrossEntropyLoss, 'focal' for FocalLoss",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor (0 = no smoothing)",
    )

    # Mode flags
    parser.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help="Use simple pooling baseline instead of LLM backbone",
    )

    # Logging
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        default=False,
        help="Disable wandb logging",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Checkpoints
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("checkpoints"),
        help="Base directory for saving checkpoints",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to saved model checkpoint (.pt file) for evaluation",
    )

    cfg = parser.parse_args(args)
    logger.info("Configuration: %s", vars(cfg))
    return cfg
