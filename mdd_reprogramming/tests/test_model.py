"""Tests for the model module.

All tests mock the Hugging Face model download so they run fully offline.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from mdd_reprogramming.model import (
    BaselineHead,
    MDDReprogrammingModel,
    PatchEncoder,
    ReprogrammingLayer,
)

BATCH = 2
PATCH_SIZE = 16
D_MODEL = 128
D_LLM = 1024
INPUT_SHAPE = (121, 145, 121)

# Expected spatial dims after patch encoder
SPATIAL = tuple((s - PATCH_SIZE) // PATCH_SIZE + 1 for s in INPUT_SHAPE)
N_PATCHES = SPATIAL[0] * SPATIAL[1] * SPATIAL[2]  # 7 * 9 * 7 = 441


def _make_fake_llm(d_llm: int = D_LLM) -> MagicMock:
    """Create a fake LLM that returns random hidden states."""
    fake_llm = MagicMock(spec=nn.Module)
    fake_llm.parameters.return_value = [
        nn.Parameter(torch.randn(4, 4), requires_grad=True)
    ]

    def fake_forward(inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        n, p, _ = inputs_embeds.shape
        return SimpleNamespace(last_hidden_state=torch.randn(n, p, d_llm))

    fake_llm.side_effect = fake_forward
    fake_llm.eval = MagicMock(return_value=fake_llm)
    return fake_llm


@pytest.fixture()
def dummy_input() -> torch.Tensor:
    """Batch of synthetic MRI volumes."""
    return torch.randn(BATCH, 1, *INPUT_SHAPE)


class TestPatchEncoder:
    """Tests for the 3D Patch Encoder."""

    def test_output_shape(self, dummy_input: torch.Tensor) -> None:
        """Patch encoder output has shape (N, P, d_model)."""
        encoder = PatchEncoder(in_channels=1, d_model=D_MODEL, patch_size=PATCH_SIZE)
        out = encoder(dummy_input)
        assert out.shape == (BATCH, N_PATCHES, D_MODEL)

    def test_kaiming_init(self) -> None:
        """Conv3d layers use Kaiming initialization (non-zero weights)."""
        encoder = PatchEncoder(in_channels=1, d_model=D_MODEL, patch_size=PATCH_SIZE)
        for m in encoder.modules():
            if isinstance(m, nn.Conv3d):
                assert m.weight.abs().sum() > 0


class TestReprogrammingLayer:
    """Tests for the Reprogramming Layer."""

    def test_output_shape(self) -> None:
        """Reprogramming layer maps (N, P, d_model) to (N, P, d_llm)."""
        layer = ReprogrammingLayer(D_MODEL, D_LLM, dropout=0.2)
        x = torch.randn(BATCH, N_PATCHES, D_MODEL)
        out = layer(x)
        assert out.shape == (BATCH, N_PATCHES, D_LLM)

    def test_trainable(self) -> None:
        """All reprogramming layer parameters are trainable."""
        layer = ReprogrammingLayer(D_MODEL, D_LLM, dropout=0.2)
        for param in layer.parameters():
            assert param.requires_grad is True


class TestMDDReprogrammingModel:
    """Tests for the full MDDReprogrammingModel."""

    @patch("mdd_reprogramming.model.AutoModel.from_pretrained")
    def test_forward_shape(self, mock_from_pretrained: MagicMock, dummy_input: torch.Tensor) -> None:
        """Full forward pass produces logits of shape (N, 2)."""
        mock_from_pretrained.return_value = _make_fake_llm()
        model = MDDReprogrammingModel(
            patch_size=PATCH_SIZE, d_model=D_MODEL, d_llm=D_LLM, baseline=False,
        )
        logits = model(dummy_input)
        assert logits.shape == (BATCH, 2)

    @patch("mdd_reprogramming.model.AutoModel.from_pretrained")
    def test_llm_frozen(self, mock_from_pretrained: MagicMock) -> None:
        """Every LLM parameter has requires_grad=False."""
        mock_from_pretrained.return_value = _make_fake_llm()
        model = MDDReprogrammingModel(
            patch_size=PATCH_SIZE, d_model=D_MODEL, d_llm=D_LLM, baseline=False,
        )
        for param in model.llm.parameters():
            assert param.requires_grad is False

    def test_baseline_forward_shape(self, dummy_input: torch.Tensor) -> None:
        """Baseline mode forward pass produces logits of shape (N, 2)."""
        model = MDDReprogrammingModel(
            patch_size=PATCH_SIZE, d_model=D_MODEL, d_llm=D_LLM, baseline=True,
        )
        logits = model(dummy_input)
        assert logits.shape == (BATCH, 2)

    def test_baseline_no_llm(self) -> None:
        """Baseline mode does not load an LLM."""
        model = MDDReprogrammingModel(
            patch_size=PATCH_SIZE, d_model=D_MODEL, d_llm=D_LLM, baseline=True,
        )
        assert not hasattr(model, "llm")
        assert hasattr(model, "baseline_head")

    @patch("mdd_reprogramming.model.AutoModel.from_pretrained")
    def test_trainable_params_exclude_llm(self, mock_from_pretrained: MagicMock) -> None:
        """Trainable parameters do not include any LLM parameters."""
        mock_from_pretrained.return_value = _make_fake_llm()
        model = MDDReprogrammingModel(
            patch_size=PATCH_SIZE, d_model=D_MODEL, d_llm=D_LLM, baseline=False,
        )
        trainable = [p for p in model.parameters() if p.requires_grad]
        llm_params = set(id(p) for p in model.llm.parameters())
        for p in trainable:
            assert id(p) not in llm_params

    @patch("mdd_reprogramming.model.AutoModel.from_pretrained")
    def test_attention_mask_shape(self, mock_from_pretrained: MagicMock, dummy_input: torch.Tensor) -> None:
        """Attention mask passed to LLM has shape (N, P)."""
        fake_llm = _make_fake_llm()
        mock_from_pretrained.return_value = fake_llm
        model = MDDReprogrammingModel(
            patch_size=PATCH_SIZE, d_model=D_MODEL, d_llm=D_LLM, baseline=False,
        )
        model(dummy_input)
        # Check the call args to the fake LLM
        call_kwargs = fake_llm.call_args[1]
        assert call_kwargs["attention_mask"].shape == (BATCH, N_PATCHES)
        assert (call_kwargs["attention_mask"] == 1).all()
