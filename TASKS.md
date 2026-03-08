# MDD Modality Reprogramming — Implementation Task

> Implement this incrementally. After completing each phase, update `PROGRESS.md` with exactly what files were created, what was implemented, and what still needs to be done. Then stop and wait for confirmation before proceeding to the next phase.
> Also run `git commit -m "Phase X complete: <summary>"` after each phase.

---

## Project Overview

Build a PyTorch pipeline to classify Major Depressive Disorder (MDD) from 3D structural MRI using a **Modality Reprogramming** framework with a frozen LLM backbone. The approach adapts Babu et al. (2026), which applied reprogramming to EEG signals, to 3D grey matter MRI volumes.

---

## Dataset Context

- **Source:** REST-meta-MDD project (DIRECT Consortium) — rfmri.org/REST-meta-MDD
- **Input:** Already preprocessed 3D grey matter images (segmented, normalized to MNI space via DARTEL, smoothed with 8mm FWHM Gaussian kernel)
- **Volume dimensions:** 121 × 145 × 121 (depth × height × width), single channel
- **Subjects:** 2380 total — 1276 MDD, 1104 Healthy Controls (mildly imbalanced, MDD is majority)
- **Labels:** 0 = Healthy Control, 1 = MDD
- **Split:** 10% blind test set held out before training, 10-fold cross-validation on remaining 90%

---

## Project Structure

Generate the following files:

```
mdd_reprogramming/
├── __init__.py
├── config.py
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_loss.py
│   └── test_train.py
├── requirements.txt
├── pytest.ini
├── Makefile
└── PROGRESS.md
```

---

## Phase 1 — Project Scaffold & Config ✅

Create the base project structure:

- `__init__.py` — marks the directory as a package
- `config.py` — all hyperparameters and paths managed via `argparse`, no hardcoded values anywhere else in the codebase. Include:
  - `data_dir`: path to NIfTI files
  - `labels_csv`: path to CSV with subject IDs and labels
  - `patch_size`: default 16
  - `d_model`: default 128
  - `llm_name`: default `Qwen/Qwen1.5-0.5B`
  - `d_llm`: default 1024
  - `lr`: default 0.001
  - `batch_size`: default 16
  - `epochs`: default 50
  - `dropout`: default 0.2
  - `n_folds`: default 10
  - `blind_test_frac`: default 0.1
  - `use_custom_loss`: boolean flag, default True
  - `baseline`: boolean flag to swap LLM for simple pooling baseline, default False
  - `no_wandb`: boolean flag to disable wandb logging, default False
  - `seed`: default 42
- `requirements.txt` — pinned versions for: torch, nibabel, transformers, scikit-learn, wandb, pytest, numpy, pandas, pathlib

**Standards:**
- PEP 8 throughout
- Type hints on all functions and methods
- Google-style docstrings on all classes and public functions
- Use `logging` module, never `print`
- Use `pathlib.Path` for all file handling

---

## Phase 2 — Dataset (`dataset.py`) ✅

Implement a PyTorch Dataset class `MRIDataset` that:

- Loads `.nii` or `.nii.gz` files using `nibabel`
- Reads subject IDs and labels from a CSV file
- Normalizes voxel intensities to zero mean and unit variance per volume
- Returns `(tensor, label)` where tensor shape is `(1, 121, 145, 121)` and label is a `torch.long` scalar
- Handles missing files gracefully with a logged warning and skip

Also implement a helper function `get_weighted_sampler(dataset)` that:
- Computes per-class weights as `N_total / (2 * N_class)`
- Returns a `torch.utils.data.WeightedRandomSampler` for use in the DataLoader

---

## Phase 3 — Model Architecture (`model.py`) ✅

Build a single `nn.Module` called `MDDReprogrammingModel` composed of four sequential components:

### 3D Patch Encoder
- `nn.Conv3d` with `kernel_size=patch_size, stride=patch_size` to divide the 121×145×121 volume into non-overlapping cubic patches
- Followed by `BatchNorm3d` + `ReLU`
- Second `Conv3d(d_model, d_model, kernel_size=3, padding=1)` + `BatchNorm3d` + `ReLU` to refine features
- Flatten spatial dimensions to produce output shape `(N, P, d_model)` where P is the number of patches
- Apply Kaiming initialization to all Conv3d layers

### Reprogramming Layer
- MLP: `Linear(d_model → d_model)` + `ReLU` + `Dropout` + `Linear(d_model → d_llm)`
- Fully trainable
- Apply Kaiming initialization

### Frozen LLM Backbone
- Load `Qwen/Qwen1.5-0.5B` from Hugging Face using `AutoModel`
- Immediately set `requires_grad=False` for every parameter in the LLM
- Pass reprogrammed tokens using the `inputs_embeds` argument
- Pass an attention mask of all ones with shape `(N, P)` alongside `inputs_embeds`
- Extract the final hidden states from the LLM output

### Classification Head
- `nn.AdaptiveAvgPool1d(1)` across the sequence dimension to get shape `(N, d_llm)`
- `nn.Linear(d_llm → 2)` to produce logits

### Baseline Mode
- When `--baseline` flag is set, replace the Reprogramming Layer and Frozen LLM with:
  - `nn.AdaptiveAvgPool3d(output_size=(4, 4, 4))` applied to patch encoder output
  - `nn.Flatten()`
  - `nn.Linear → 256` + ReLU + `nn.Linear → 2`
- Patch encoder remains identical in both modes

---

## Phase 4 — Loss Function ✅

Implement two loss modes in a `build_loss` function:

**Standard:** `nn.CrossEntropyLoss(weight=class_weights)` applied directly to logits.

**Custom (default):** Pass `torch.softmax(logits, dim=1)` explicitly into `nn.CrossEntropyLoss(weight=class_weights)`. This is toggled by `--use_custom_loss`. Note: in prior work on this dataset, the standard formulation caused training to stall at loss ≈ 0.69; the custom formulation resolved this.

Compute class weights from the training fold as `N_total / (2 * N_class)` and pass to the loss.

---

## Phase 5 — Training Loop (`train.py`) ✅

Implement the full training pipeline:

- Load config from `config.py`
- Hold out 10% blind test set before any cross-validation (use `seed` for reproducibility)
- Run 10-fold cross-validation (without stratification) on remaining 90%
- For each fold:
  - Build DataLoaders with `WeightedRandomSampler` on training split
  - Instantiate model, loss, and AdamW optimizer
  - Pass ONLY parameters with `requires_grad=True` to the optimizer
  - Train for `epochs` epochs
  - At each epoch log: train loss, val loss, Accuracy, Sensitivity, Specificity, F1, AUROC, AUPRC
  - Save best checkpoint (by validation AUROC) to disk
- After all folds, evaluate best checkpoints on blind test set
- Report mean ± std across folds for all metrics

**wandb integration:**
- `wandb.init(project="mdd-reprogramming")` at start of training
- `wandb.config` stores all hyperparameters from `config.py`
- Log all per-epoch metrics
- Log per-fold summary at end of each fold
- Log best AUROC and the epoch it occurred
- Save best checkpoint as a wandb artifact
- Skip all wandb calls if `--no_wandb` is set

---

## Phase 6 — Evaluation (`evaluate.py`) ✅

Implement a standalone `evaluate.py` script that:

- Loads a saved checkpoint
- Runs inference on the blind test set
- Reports: Accuracy, Sensitivity, Specificity, F1, AUROC, AUPRC
- Matches the exact metrics reported in the prior MDD-Net paper for direct comparison

---

## Phase 7 — Unit Tests (`tests/`)

Use `pytest`. Mock the Hugging Face model download in all tests using `unittest.mock` so tests run fully offline.

### `test_dataset.py`
- Dataset returns tensor of shape `(1, 121, 145, 121)`
- Label is `torch.long`
- Voxel intensities have mean ≈ 0 and std ≈ 1 after normalization
- Missing file is handled gracefully without crashing

### `test_model.py`
- Patch encoder output shape is `(N, P, d_model)`
- Reprogramming layer output shape is `(N, P, d_llm)`
- Every LLM parameter has `requires_grad=False`
- Full forward pass produces logits of shape `(N, 2)`
- Baseline mode forward pass also produces logits of shape `(N, 2)`

### `test_loss.py`
- Custom loss does not return NaN on dummy inputs
- Custom loss and standard loss both return scalar tensors
- Class weights are correctly computed from a dummy label distribution

### `test_train.py`
- Smoke test: 2 epochs on 10 synthetic `(1, 121, 145, 121)` samples completes without error
- Optimizer only contains parameters with `requires_grad=True`
- Checkpoint is saved after smoke test run

Add `pytest.ini` with `testpaths = tests` and `Makefile` with a `make test` target that runs `pytest -v`.

---

## Definition of Done

All phases are complete when:
- [x] Phase 1 — Project Scaffold & Config (4 tests passing)
- [x] Phase 2 — Dataset (12 tests passing)
- [x] Phase 3 — Model Architecture (10 tests passing)
- [x] Phase 4 — Loss Function (11 tests passing)
- [x] Phase 5 — Training Loop (6 tests passing)
- [x] Phase 6 — Evaluation (6 tests passing)
- [ ] Phase 7 — Remaining unit tests (test_model, test_loss, test_train)
- [ ] All files listed in the project structure exist
- [ ] `make test` passes with no failures
- [ ] A dry run with `--no_wandb --baseline` completes on synthetic data without errors
- [ ] `PROGRESS.md` is fully up to date