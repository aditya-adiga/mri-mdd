"""Tests for the config module."""

from pathlib import Path

from mdd_reprogramming.config import parse_args


class TestParseArgs:
    """Tests for parse_args function."""

    def test_defaults(self) -> None:
        """All default values match the specification."""
        cfg = parse_args([])
        assert cfg.data_dir == Path("data/wc1_MDD_VBM_dataset")
        assert cfg.labels_csv == Path("data/labels.csv")
        assert cfg.patch_size == 16
        assert cfg.d_model == 128
        assert cfg.llm_name == "Qwen/Qwen1.5-0.5B"
        assert cfg.d_llm == 1024
        assert cfg.lr == 0.001
        assert cfg.batch_size == 16
        assert cfg.epochs == 50
        assert cfg.dropout == 0.2
        assert cfg.n_folds == 10
        assert cfg.blind_test_frac == 0.1
        assert cfg.baseline is False
        assert cfg.no_wandb is False
        assert cfg.seed == 42

    def test_override_values(self) -> None:
        """CLI arguments override defaults."""
        cfg = parse_args([
            "--patch_size", "32",
            "--lr", "0.0001",
            "--epochs", "10",
            "--baseline",
            "--no_wandb",
            "--seed", "123",
        ])
        assert cfg.patch_size == 32
        assert cfg.lr == 0.0001
        assert cfg.epochs == 10
        assert cfg.baseline is True
        assert cfg.no_wandb is True
        assert cfg.seed == 123

    def test_paths_are_pathlib(self) -> None:
        """Path arguments are pathlib.Path instances."""
        cfg = parse_args(["--data_dir", "/tmp/data", "--labels_csv", "/tmp/l.csv"])
        assert isinstance(cfg.data_dir, Path)
        assert isinstance(cfg.labels_csv, Path)
