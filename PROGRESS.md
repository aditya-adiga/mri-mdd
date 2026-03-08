# MDD Reprogramming — Progress

## Phase 1 — Project Scaffold & Config (COMPLETE)

### Files Created
- `mdd_reprogramming/__init__.py` — package marker
- `mdd_reprogramming/tests/__init__.py` — test package marker
- `mdd_reprogramming/config.py` — argparse-based configuration with all hyperparameters
- `requirements.txt` — pinned dependencies
- `pytest.ini` — pytest configuration (`testpaths = mdd_reprogramming/tests`)
- `Makefile` — `make test` target

### What Was Implemented
- All hyperparameters configurable via CLI args with specified defaults
- `parse_args()` function with type hints, docstring, and logging
- `BooleanOptionalAction` for `--use_custom_loss` / `--no-use_custom_loss`
- `pathlib.Path` for all path arguments
- PEP 8, type hints, Google-style docstrings, `logging` module — all enforced

---

## Phase 2 — Dataset (COMPLETE)

### Files Created
- `mdd_reprogramming/dataset.py` — MRIDataset class and WeightedRandomSampler helper

### What Was Implemented
- `MRIDataset(Dataset)` class:
  - Loads `.nii` / `.nii.gz` files via nibabel
  - Reads subject IDs and labels from CSV (`subject_id`, `label` columns)
  - Optional `subject_ids` filter for train/val/test splits
  - Per-volume normalization to zero mean and unit variance
  - Returns `(tensor, label)` with shape `(1, 121, 145, 121)` and `torch.long`
  - Missing files logged as warnings and skipped
  - `.labels` property for convenient access to all labels
- `get_weighted_sampler(dataset)`:
  - Computes class weights as `N_total / (2 * N_class)`
  - Returns `WeightedRandomSampler` with replacement

### Remaining
- Phase 3: Model Architecture (`model.py`)
- Phase 4: Loss Function
- Phase 5: Training Loop (`train.py`)
- Phase 6: Evaluation (`evaluate.py`)
- Phase 7: Unit Tests (`tests/`)
