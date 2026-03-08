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

### Remaining
- Phase 2: Dataset (`dataset.py`)
- Phase 3: Model Architecture (`model.py`)
- Phase 4: Loss Function
- Phase 5: Training Loop (`train.py`)
- Phase 6: Evaluation (`evaluate.py`)
- Phase 7: Unit Tests (`tests/`)
