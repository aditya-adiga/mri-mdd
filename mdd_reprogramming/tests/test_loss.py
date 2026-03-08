"""Tests for the loss function module."""

import torch

from mdd_reprogramming.model import (
    CustomCrossEntropyLoss,
    build_loss,
    compute_class_weights,
)


class TestComputeClassWeights:
    """Tests for compute_class_weights function."""

    def test_balanced_classes(self) -> None:
        """Equal class counts produce equal weights of 1.0."""
        labels = [0, 0, 0, 1, 1, 1]
        weights = compute_class_weights(labels)
        assert weights.shape == (2,)
        assert torch.allclose(weights, torch.tensor([1.0, 1.0]))

    def test_imbalanced_classes(self) -> None:
        """Weights are N_total / (2 * N_class) for imbalanced data."""
        # 8 total: 6 MDD (label=1), 2 HC (label=0)
        labels = [1, 1, 1, 1, 1, 1, 0, 0]
        weights = compute_class_weights(labels)
        expected_hc = 8.0 / (2 * 2)   # 2.0
        expected_mdd = 8.0 / (2 * 6)  # 0.6667
        assert abs(weights[0].item() - expected_hc) < 1e-5
        assert abs(weights[1].item() - expected_mdd) < 1e-4

    def test_output_dtype(self) -> None:
        """Weights tensor is float32."""
        weights = compute_class_weights([0, 1, 1])
        assert weights.dtype == torch.float32


class TestCustomCrossEntropyLoss:
    """Tests for CustomCrossEntropyLoss."""

    def test_returns_scalar(self) -> None:
        """Custom loss returns a scalar tensor."""
        loss_fn = CustomCrossEntropyLoss()
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 1, 0])
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_no_nan(self) -> None:
        """Custom loss does not produce NaN on normal inputs."""
        loss_fn = CustomCrossEntropyLoss()
        logits = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8,))
        loss = loss_fn(logits, targets)
        assert not torch.isnan(loss)

    def test_no_nan_with_weights(self) -> None:
        """Custom loss with class weights does not produce NaN."""
        weights = torch.tensor([1.5, 0.75])
        loss_fn = CustomCrossEntropyLoss(weight=weights)
        logits = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8,))
        loss = loss_fn(logits, targets)
        assert not torch.isnan(loss)

    def test_requires_grad(self) -> None:
        """Loss is differentiable (has grad_fn)."""
        loss_fn = CustomCrossEntropyLoss()
        logits = torch.randn(4, 2, requires_grad=True)
        targets = torch.tensor([0, 1, 0, 1])
        loss = loss_fn(logits, targets)
        assert loss.requires_grad


class TestBuildLoss:
    """Tests for the build_loss factory function."""

    def test_custom_loss_type(self) -> None:
        """build_loss with use_custom_loss=True returns CustomCrossEntropyLoss."""
        loss_fn = build_loss([0, 0, 1, 1], use_custom_loss=True)
        assert isinstance(loss_fn, CustomCrossEntropyLoss)

    def test_standard_loss_type(self) -> None:
        """build_loss with use_custom_loss=False returns nn.CrossEntropyLoss."""
        loss_fn = build_loss([0, 0, 1, 1], use_custom_loss=False)
        assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)

    def test_standard_loss_returns_scalar(self) -> None:
        """Standard CE loss returns a scalar tensor."""
        loss_fn = build_loss([0, 0, 1, 1], use_custom_loss=False)
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 1, 0])
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_both_modes_finite(self) -> None:
        """Both custom and standard loss produce finite values."""
        logits = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8,))
        labels = [0, 0, 0, 1, 1, 1, 1, 1]

        for use_custom in [True, False]:
            loss_fn = build_loss(labels, use_custom_loss=use_custom)
            loss = loss_fn(logits, targets)
            assert torch.isfinite(loss), f"use_custom_loss={use_custom} produced non-finite loss"
