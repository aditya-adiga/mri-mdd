"""Standalone evaluation script for MDD Reprogramming.

Loads a saved checkpoint, runs inference on the blind test set,
and reports classification metrics matching the MDD-Net paper format.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from mdd_reprogramming.config import parse_args
from mdd_reprogramming.dataset import MRIDataset
from mdd_reprogramming.model import MDDReprogrammingModel, build_loss
from mdd_reprogramming.train import compute_metrics, evaluate

logger = logging.getLogger(__name__)


def run_evaluation(args: list[str] | None = None) -> dict[str, float]:
    """Run evaluation on the blind test set using a saved checkpoint.

    Args:
        args: Optional list of argument strings.

    Returns:
        Dictionary of metric name to value.
    """
    cfg = parse_args(args)

    if cfg.checkpoint is None:
        raise ValueError("--checkpoint is required for evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load full dataset and extract blind test set
    full_dataset = MRIDataset(data_dir=cfg.data_dir, labels_csv=cfg.labels_csv)
    n_total = len(full_dataset)

    # Reproduce the same blind test split as training
    indices = np.arange(n_total)
    rng = np.random.RandomState(cfg.seed)
    rng.shuffle(indices)
    n_test = int(n_total * cfg.blind_test_frac)
    test_indices = indices[:n_test]

    test_dataset = Subset(full_dataset, test_indices.tolist())
    test_dataset.labels = [full_dataset.labels[i] for i in test_indices]

    logger.info("Blind test set: %d samples", len(test_dataset))

    # Build model and load checkpoint
    model = MDDReprogrammingModel(
        patch_size=cfg.patch_size,
        d_model=cfg.d_model,
        d_llm=cfg.d_llm,
        llm_name=cfg.llm_name,
        dropout=cfg.dropout,
        baseline=cfg.baseline,
    ).to(device)

    state_dict = torch.load(cfg.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    logger.info("Loaded checkpoint: %s", cfg.checkpoint)

    # Build loss and evaluate
    loss_fn = build_loss(test_dataset.labels, use_custom_loss=cfg.use_custom_loss)
    loss_fn = loss_fn.to(device)

    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    test_loss, metrics = evaluate(model, test_loader, loss_fn, device)

    # Report metrics in MDD-Net paper format
    logger.info("=== Blind Test Set Results ===")
    logger.info("Loss: %.4f", test_loss)
    for name, value in metrics.items():
        logger.info("%s: %.4f", name, value)

    return metrics


def main(args: list[str] | None = None) -> dict[str, float]:
    """Run standalone evaluation.

    Args:
        args: Optional list of argument strings.

    Returns:
        Dictionary of metric name to value.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    return run_evaluation(args)


if __name__ == "__main__":
    main()
