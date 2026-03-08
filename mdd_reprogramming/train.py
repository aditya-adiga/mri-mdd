"""Training pipeline for MDD Reprogramming.

Implements 10-fold cross-validation with a held-out blind test set,
per-epoch metric logging, checkpoint saving, and optional wandb integration.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

from mdd_reprogramming.config import parse_args
from mdd_reprogramming.dataset import MRIDataset, get_weighted_sampler
from mdd_reprogramming.model import MDDReprogrammingModel, build_loss

logger = logging.getLogger(__name__)


def compute_metrics(
    labels: np.ndarray, preds: np.ndarray, probs: np.ndarray
) -> dict[str, float]:
    """Compute classification metrics.

    Args:
        labels: Ground truth labels of shape (N,).
        preds: Predicted class labels of shape (N,).
        probs: Predicted probabilities for positive class of shape (N,).

    Returns:
        Dictionary with Accuracy, Sensitivity, Specificity, F1, AUROC, AUPRC.
    """
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "Accuracy": accuracy_score(labels, preds),
        "Sensitivity": recall_score(labels, preds, zero_division=0),
        "Specificity": specificity,
        "F1": f1_score(labels, preds, zero_division=0),
        "AUROC": roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
        "AUPRC": average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch.

    Args:
        model: The model to train.
        loader: Training DataLoader.
        loss_fn: Loss function.
        optimizer: Optimizer.
        device: Device to use.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Evaluate the model on a dataset.

    Args:
        model: The model to evaluate.
        loader: DataLoader for evaluation.
        loss_fn: Loss function.
        device: Device to use.

    Returns:
        Tuple of (average loss, metrics dict).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_labels = []
    all_preds = []
    all_probs = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item()
        n_batches += 1

        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)
        all_labels.append(y.cpu().numpy())
        all_preds.append(pred.cpu().numpy())
        all_probs.append(prob.cpu().numpy())

    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)
    avg_loss = total_loss / max(n_batches, 1)
    metrics = compute_metrics(labels, preds, probs)
    return avg_loss, metrics


def train_fold(
    fold: int,
    train_dataset: MRIDataset,
    val_dataset: MRIDataset,
    cfg: argparse.Namespace,
    device: torch.device,
    checkpoint_dir: Path,
    use_wandb: bool,
) -> tuple[dict[str, float], Path]:
    """Train and evaluate a single fold.

    Args:
        fold: Fold number (0-indexed).
        train_dataset: Training split dataset.
        val_dataset: Validation split dataset.
        cfg: Configuration namespace.
        device: Device to use.
        checkpoint_dir: Directory to save checkpoints.
        use_wandb: Whether to log to wandb.

    Returns:
        Tuple of (best validation metrics, path to best checkpoint).
    """
    # Build DataLoader with WeightedRandomSampler for training
    sampler = get_weighted_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    # Instantiate model
    model = MDDReprogrammingModel(
        patch_size=cfg.patch_size,
        d_model=cfg.d_model,
        d_llm=cfg.d_llm,
        llm_name=cfg.llm_name,
        dropout=cfg.dropout,
        baseline=cfg.baseline,
    ).to(device)

    # Loss
    loss_fn = build_loss(train_dataset.labels)
    loss_fn = loss_fn.to(device)

    # Optimizer — only trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr)

    best_auroc = -1.0
    best_metrics: dict[str, float] = {}
    best_epoch = -1
    fold_dir = checkpoint_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = fold_dir / "best.pt"

    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, loss_fn, device)

        logger.info(
            "Fold %d Epoch %d/%d — train_loss=%.4f val_loss=%.4f Acc=%.4f Sen=%.4f Spe=%.4f F1=%.4f AUROC=%.4f AUPRC=%.4f",
            fold, epoch + 1, cfg.epochs, train_loss, val_loss,
            val_metrics["Accuracy"], val_metrics["Sensitivity"],
            val_metrics["Specificity"], val_metrics["F1"],
            val_metrics["AUROC"], val_metrics["AUPRC"],
        )

        if use_wandb:
            import wandb

            wandb.log({
                f"fold_{fold}/train_loss": train_loss,
                f"fold_{fold}/val_loss": val_loss,
                **{f"fold_{fold}/{k}": v for k, v in val_metrics.items()},
                "epoch": epoch + 1,
            })

        if val_metrics["AUROC"] > best_auroc:
            best_auroc = val_metrics["AUROC"]
            best_metrics = val_metrics.copy()
            best_epoch = epoch + 1
            torch.save(model.state_dict(), ckpt_path)

    logger.info(
        "Fold %d best AUROC=%.4f at epoch %d", fold, best_auroc, best_epoch
    )

    if use_wandb:
        import wandb

        wandb.log({
            f"fold_{fold}/best_AUROC": best_auroc,
            f"fold_{fold}/best_epoch": best_epoch,
        })
        artifact = wandb.Artifact(f"best_model_fold_{fold}", type="model")
        artifact.add_file(str(ckpt_path))
        wandb.log_artifact(artifact)

    return best_metrics, ckpt_path


def evaluate_on_test(
    checkpoint_paths: list[Path],
    test_dataset: MRIDataset,
    cfg: argparse.Namespace,
    device: torch.device,
) -> dict[str, tuple[float, float]]:
    """Evaluate best checkpoints from all folds on the blind test set.

    Args:
        checkpoint_paths: List of paths to best checkpoints per fold.
        test_dataset: Blind test set dataset.
        cfg: Configuration namespace.
        device: Device to use.

    Returns:
        Dictionary mapping metric name to (mean, std) across folds.
    """
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    all_fold_metrics: list[dict[str, float]] = []

    for ckpt_path in checkpoint_paths:
        model = MDDReprogrammingModel(
            patch_size=cfg.patch_size,
            d_model=cfg.d_model,
            d_llm=cfg.d_llm,
            llm_name=cfg.llm_name,
            dropout=cfg.dropout,
            baseline=cfg.baseline,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

        loss_fn = build_loss(test_dataset.labels)
        loss_fn = loss_fn.to(device)
        _, metrics = evaluate(model, test_loader, loss_fn, device)
        all_fold_metrics.append(metrics)

    # Aggregate mean ± std
    metric_names = list(all_fold_metrics[0].keys())
    results: dict[str, tuple[float, float]] = {}
    for name in metric_names:
        values = [m[name] for m in all_fold_metrics]
        results[name] = (float(np.mean(values)), float(np.std(values)))

    return results


def main(args: list[str] | None = None) -> None:
    """Run the full training pipeline.

    Args:
        args: Optional list of argument strings. If None, reads from sys.argv.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = parse_args(args)

    # Reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # wandb setup
    use_wandb = not cfg.no_wandb
    if use_wandb:
        import wandb

        wandb.init(project="mdd-reprogramming")
        wandb.config.update(vars(cfg))

    # Load full dataset
    full_dataset = MRIDataset(data_dir=cfg.data_dir, labels_csv=cfg.labels_csv)
    n_total = len(full_dataset)

    # Hold out blind test set
    indices = np.arange(n_total)
    rng = np.random.RandomState(cfg.seed)
    rng.shuffle(indices)
    n_test = int(n_total * cfg.blind_test_frac)
    test_indices = indices[:n_test]
    train_val_indices = indices[n_test:]

    test_dataset = Subset(full_dataset, test_indices.tolist())
    # Give test_dataset a .labels property for build_loss
    test_dataset.labels = [full_dataset.labels[i] for i in test_indices]

    logger.info(
        "Split: %d train+val, %d blind test", len(train_val_indices), n_test
    )

    # Checkpoint directory: checkpoints/<timestamp>_<mode>/
    from datetime import datetime
    import json

    mode_tag = "baseline" if cfg.baseline else "llm"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = cfg.checkpoint_dir / f"{timestamp}_{mode_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    with open(run_dir / "config.json", "w") as f:
        config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(cfg).items()}
        json.dump(config_dict, f, indent=2)
    logger.info("Run directory: %s", run_dir)

    # K-fold cross-validation (no stratification)
    kf = KFold(n_splits=cfg.n_folds, shuffle=False)
    fold_metrics: list[dict[str, float]] = []
    checkpoint_paths: list[Path] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_indices)):
        logger.info("=== Fold %d/%d ===", fold + 1, cfg.n_folds)
        train_subset_indices = train_val_indices[train_idx].tolist()
        val_subset_indices = train_val_indices[val_idx].tolist()

        train_ds = Subset(full_dataset, train_subset_indices)
        val_ds = Subset(full_dataset, val_subset_indices)

        # Give subsets a .labels property for build_loss and get_weighted_sampler
        train_ds.labels = [full_dataset.labels[i] for i in train_subset_indices]
        val_ds.labels = [full_dataset.labels[i] for i in val_subset_indices]

        best_metrics, ckpt_path = train_fold(
            fold=fold,
            train_dataset=train_ds,
            val_dataset=val_ds,
            cfg=cfg,
            device=device,
            checkpoint_dir=run_dir,
            use_wandb=use_wandb,
        )
        fold_metrics.append(best_metrics)
        checkpoint_paths.append(ckpt_path)

        if use_wandb:
            import wandb

            wandb.log({
                f"fold_{fold}/summary_{k}": v for k, v in best_metrics.items()
            })

    # Report cross-validation summary
    metric_names = list(fold_metrics[0].keys())
    logger.info("=== Cross-Validation Summary ===")
    for name in metric_names:
        values = [m[name] for m in fold_metrics]
        logger.info(
            "%s: %.4f ± %.4f", name, np.mean(values), np.std(values)
        )

    # Evaluate on blind test set
    logger.info("=== Blind Test Set Evaluation ===")
    test_results = evaluate_on_test(checkpoint_paths, test_dataset, cfg, device)
    for name, (mean, std) in test_results.items():
        logger.info("Test %s: %.4f ± %.4f", name, mean, std)

    if use_wandb:
        import wandb

        for name, (mean, std) in test_results.items():
            wandb.log({f"test/{name}_mean": mean, f"test/{name}_std": std})
        wandb.finish()

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
