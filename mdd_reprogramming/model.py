"""Model architecture for MDD Reprogramming with frozen LLM backbone.

Contains MDDReprogrammingModel with four components: 3D patch encoder,
reprogramming layer, frozen LLM backbone, and classification head.
A baseline mode replaces the reprogramming layer and LLM with simple pooling.
"""

import logging

import torch
import torch.nn as nn
from transformers import AutoModel

logger = logging.getLogger(__name__)


class PatchEncoder(nn.Module):
    """3D convolutional patch encoder for MRI volumes.

    Divides the input volume into non-overlapping cubic patches and
    produces per-patch feature vectors.

    Args:
        in_channels: Number of input channels (1 for single-channel MRI).
        d_model: Output feature dimensionality per patch.
        patch_size: Size of non-overlapping 3D patches.
    """

    def __init__(self, in_channels: int, d_model: int, patch_size: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, d_model, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm3d(d_model),
            nn.ReLU(),
            nn.Conv3d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm3d(d_model),
            nn.ReLU(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Kaiming initialization to all Conv3d layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode volume into patch features.

        Args:
            x: Input tensor of shape (N, 1, D, H, W).

        Returns:
            Patch features of shape (N, P, d_model).
        """
        x = self.encoder(x)  # (N, d_model, D', H', W')
        n, c = x.shape[:2]
        return x.view(n, c, -1).permute(0, 2, 1)  # (N, P, d_model)


class ReprogrammingLayer(nn.Module):
    """MLP that maps patch features to LLM embedding space.

    Args:
        d_model: Input feature dimensionality.
        d_llm: Output dimensionality matching LLM hidden size.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, d_llm: int, dropout: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_llm),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Kaiming initialization to all Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map patch features to LLM embedding space.

        Args:
            x: Patch features of shape (N, P, d_model).

        Returns:
            Reprogrammed tokens of shape (N, P, d_llm).
        """
        return self.mlp(x)


class BaselineHead(nn.Module):
    """Simple pooling baseline replacing reprogramming layer and LLM.

    Args:
        d_model: Feature dimensionality from patch encoder.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(output_size=(4, 4, 4))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(d_model * 4 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor, encoder_spatial_shape: tuple[int, int, int]) -> torch.Tensor:
        """Forward pass for baseline mode.

        Args:
            x: Patch features of shape (N, P, d_model).
            encoder_spatial_shape: (D', H', W') from the patch encoder output.

        Returns:
            Logits of shape (N, 2).
        """
        n, _p, c = x.shape
        d, h, w = encoder_spatial_shape
        x = x.permute(0, 2, 1).view(n, c, d, h, w)  # (N, d_model, D', H', W')
        x = self.pool(x)  # (N, d_model, 4, 4, 4)
        x = self.flatten(x)  # (N, d_model * 64)
        return self.fc(x)  # (N, 2)


class MDDReprogrammingModel(nn.Module):
    """MDD classification via modality reprogramming with frozen LLM backbone.

    Composes a 3D patch encoder, reprogramming layer, frozen LLM backbone,
    and classification head. In baseline mode, the reprogramming layer and
    LLM are replaced with simple pooling.

    Args:
        patch_size: Size of non-overlapping 3D patches.
        d_model: Patch encoder output dimensionality.
        d_llm: LLM hidden size.
        llm_name: Hugging Face model name for frozen LLM backbone.
        dropout: Dropout rate for reprogramming layer.
        baseline: If True, use simple pooling instead of LLM.
    """

    # Spatial dims after patch encoder: floor((input_dim - patch_size) / patch_size) + 1
    INPUT_SHAPE = (121, 145, 121)

    def __init__(
        self,
        patch_size: int = 16,
        d_model: int = 128,
        d_llm: int = 1024,
        llm_name: str = "Qwen/Qwen1.5-0.5B",
        dropout: float = 0.2,
        baseline: bool = False,
    ) -> None:
        super().__init__()
        self.baseline = baseline
        self.patch_encoder = PatchEncoder(
            in_channels=1, d_model=d_model, patch_size=patch_size
        )

        # Compute spatial dimensions after patch encoding
        self.encoder_spatial_shape = tuple(
            (s - patch_size) // patch_size + 1 for s in self.INPUT_SHAPE
        )
        n_patches = 1
        for s in self.encoder_spatial_shape:
            n_patches *= s

        if baseline:
            logger.info("Using baseline mode (no LLM)")
            self.baseline_head = BaselineHead(d_model)
        else:
            logger.info("Loading frozen LLM: %s", llm_name)
            self.reprogramming = ReprogrammingLayer(d_model, d_llm, dropout)
            self.llm = AutoModel.from_pretrained(llm_name)
            # Freeze all LLM parameters
            for param in self.llm.parameters():
                param.requires_grad = False
            self.llm.eval()
            # Gradient checkpointing: recompute activations during backward
            # to save memory while still allowing gradient flow
            self.llm.gradient_checkpointing_enable()

            self.pool = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Sequential(
                nn.Linear(d_llm, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 2),
            )

        logger.info(
            "Model initialised: patches=%d, d_model=%d, baseline=%s",
            n_patches, d_model, baseline,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the full model.

        Args:
            x: Input MRI volume of shape (N, 1, 121, 145, 121).

        Returns:
            Logits of shape (N, 2).
        """
        patches = self.patch_encoder(x)  # (N, P, d_model)

        if self.baseline:
            return self.baseline_head(patches, self.encoder_spatial_shape)

        reprogrammed = self.reprogramming(patches)  # (N, P, d_llm)

        # Create attention mask of all ones
        attention_mask = torch.ones(
            reprogrammed.shape[:2],
            dtype=torch.long,
            device=reprogrammed.device,
        )

        # Cast to LLM dtype (e.g. bfloat16) to match frozen weights
        llm_dtype = next(self.llm.parameters()).dtype
        reprogrammed = reprogrammed.to(llm_dtype)

        # Pass through frozen LLM (requires_grad=False prevents weight updates,
        # gradient checkpointing recomputes activations to save memory)
        llm_output = self.llm(
            inputs_embeds=reprogrammed,
            attention_mask=attention_mask,
        )
        hidden_states = llm_output.last_hidden_state.float()  # (N, P, d_llm)

        # Pool across sequence dimension and classify
        pooled = self.pool(hidden_states.permute(0, 2, 1))  # (N, d_llm, 1)
        pooled = pooled.squeeze(-1)  # (N, d_llm)
        return self.classifier(pooled)  # (N, 2)


def compute_class_weights(labels: list[int], num_classes: int = 2) -> torch.Tensor:
    """Compute class weights as N_total / (2 * N_class).

    Args:
        labels: List of integer class labels.
        num_classes: Number of classes (ensures output has correct size).

    Returns:
        Tensor of per-class weights with shape (num_classes,).
    """
    import numpy as np

    counts = np.bincount(labels, minlength=num_classes)
    n_total = len(labels)
    # Avoid division by zero for missing classes
    counts = np.maximum(counts, 1)
    weights = n_total / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Focal Loss with class weights and label smoothing.

    Reduces the relative loss for well-classified examples, focusing
    training on hard negatives. Applies label smoothing before computing loss.

    Args:
        weight: Per-class weights tensor of shape (num_classes,).
        gamma: Focusing parameter. Higher values down-weight easy examples more.
        label_smoothing: Label smoothing factor in [0, 1).
    """

    def __init__(
        self,
        weight: torch.Tensor,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Predicted logits of shape (N, C).
            targets: Ground truth labels of shape (N,).

        Returns:
            Scalar loss value.
        """
        num_classes = logits.size(1)
        probs = torch.softmax(logits, dim=1)

        # Label smoothing: convert hard targets to soft
        smooth_targets = torch.full_like(
            probs, self.label_smoothing / (num_classes - 1)
        )
        smooth_targets.scatter_(
            1, targets.unsqueeze(1), 1.0 - self.label_smoothing
        )

        # Focal modulation: (1 - p_t)^gamma
        focal_weight = (1.0 - probs) ** self.gamma

        # Per-sample class weight
        class_w = self.weight[targets].unsqueeze(1)  # (N, 1)

        # Focal cross-entropy with smoothed targets
        log_probs = torch.log_softmax(logits, dim=1)
        loss = -class_w * focal_weight * smooth_targets * log_probs
        return loss.sum(dim=1).mean()


def build_loss(labels: list[int], loss_fn: str = "ce", label_smoothing: float = 0.1) -> nn.Module:
    """Build loss function with class weights.

    Args:
        labels: Training fold labels for computing class weights.
        loss_fn: Loss type — "ce" for CrossEntropyLoss, "focal" for FocalLoss.
        label_smoothing: Label smoothing factor (used by both ce and focal).

    Returns:
        Loss module.
    """
    class_weights = compute_class_weights(labels)
    if loss_fn == "focal":
        logger.info("Using FocalLoss with class weights, label_smoothing=%.2f", label_smoothing)
        return FocalLoss(weight=class_weights, label_smoothing=label_smoothing)
    logger.info("Using CrossEntropyLoss with class weights, label_smoothing=%.2f", label_smoothing)
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
