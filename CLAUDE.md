# CLAUDE.md — Project Guidelines

## Project Overview

MDD classification from 3D structural MRI using modality reprogramming with a frozen LLM backbone (Qwen 0.5B). PyTorch-based, 10-fold cross-validation pipeline.

## Repository Structure

```
mdd_reprogramming/
├── config.py          # All hyperparameters — argparse, no hardcoded values elsewhere
├── dataset.py         # MRIDataset, WeightedRandomSampler
├── model.py           # MDDReprogrammingModel, PatchEncoder, ReprogrammingLayer, build_loss
├── train.py           # Training pipeline with K-fold CV
├── evaluate.py        # Standalone checkpoint evaluation
└── tests/             # pytest unit tests
```

## Data

- NIfTI files in `data/wc1_MDD_VBM_dataset/`
- Labels in `data/labels.csv` (columns: `subject_id`, `label`)
- ID convention: `S{site}-1-{num}` = MDD (label=1), `S{site}-2-{num}` = HC (label=0)
- Volume shape: 121 × 145 × 121, single channel
- 2,380 subjects: 1,276 MDD, 1,104 HC
- **Never commit data files** — `data/` is in `.gitignore`

## Code Standards

- **PEP 8** throughout
- **Type hints** on all functions and methods
- **Google-style docstrings** on all classes and public functions
- **`logging` module only** — never use `print` for output
- **`pathlib.Path`** for all file paths
- **No hardcoded values** — all hyperparameters flow through `config.py`

## Making Changes

### Before modifying code:
1. Read the relevant source files first — understand before changing
2. Check existing tests to understand expected behavior
3. Create a feature branch: `git checkout -b <descriptive-branch-name>`

### While writing code:
- Keep changes focused — one logical change per PR
- Maintain existing code style and patterns
- Don't add unnecessary abstractions or over-engineer
- Don't add comments to code you didn't change
- Ensure the LLM backbone stays frozen (`requires_grad=False`)
- Only pass `requires_grad=True` parameters to the optimizer
- Handle LLM dtype casting (bfloat16) in the forward pass
- Normalize MRI data over brain voxels only (non-zero mask), not the full volume

### After making changes:
1. Run `pytest -v` — all tests must pass before committing
2. If you changed behavior, update or add corresponding tests
3. Update `PROGRESS.md` if completing a phase or milestone
4. **Update this file (`CLAUDE.md`)** if you encountered bugs, new conventions, architectural decisions, or gotchas that future sessions should know about
5. Commit with a clear, concise message describing the "why"
6. Create a PR — never push directly to `master`

## Testing

- Run: `pytest -v`
- All HuggingFace model downloads must be mocked in tests (`unittest.mock`)
- Tests must run fully offline
- Mock LLM must use `side_effect=lambda: iter([param])` for `parameters()` (not a static list)
- Synthetic NIfTI data uses shape `(121, 145, 121)` to match real data

## Training Commands

```bash
# Full LLM model
python -m mdd_reprogramming.train --no_wandb --epochs 10

# Baseline (no LLM)
python -m mdd_reprogramming.train --no_wandb --baseline --epochs 10

# Evaluate checkpoint
python -m mdd_reprogramming.evaluate --checkpoint checkpoints/best_fold_0.pt
```

## Known Issues / Decisions

- **No custom loss** — double-softmax variant was removed; it crushed gradients. Standard `CrossEntropyLoss` with class weights is used.
- **Brain-masked normalization** — only non-zero voxels are z-normalized; background stays at 0.
- **LLM dtype** — Qwen loads in bfloat16; reprogrammed embeddings are cast to match before passing to LLM, then cast back to float32 after.
- **Class weights** — `N_total / (2 * N_class)` with `minlength=2` to handle single-class subsets.
