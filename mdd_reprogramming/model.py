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

            self.pool = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Linear(d_llm, 2)

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

        # Pass through frozen LLM
        with torch.no_grad():
            llm_output = self.llm(
                inputs_embeds=reprogrammed,
                attention_mask=attention_mask,
            )
        hidden_states = llm_output.last_hidden_state  # (N, P, d_llm)

        # Pool across sequence dimension and classify
        pooled = self.pool(hidden_states.permute(0, 2, 1))  # (N, d_llm, 1)
        pooled = pooled.squeeze(-1)  # (N, d_llm)
        return self.classifier(pooled)  # (N, 2)
