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

---

## Unit Tests — Phases 1 & 2 (COMPLETE)

### Files Created
- `mdd_reprogramming/tests/test_config.py` — 4 tests for config defaults, overrides, custom loss toggle, pathlib types
- `mdd_reprogramming/tests/test_dataset.py` — 12 tests for MRIDataset and WeightedRandomSampler

### Test Summary (16/16 passed)
**test_config.py (4 tests):**
- `test_defaults` — all 17 default values match spec
- `test_override_values` — CLI args override defaults
- `test_disable_custom_loss` — `--no-use_custom_loss` works
- `test_paths_are_pathlib` — path args are `pathlib.Path`

**test_dataset.py (12 tests):**
- `test_length` — dataset length matches valid samples
- `test_tensor_shape` — output shape is `(1, 121, 145, 121)`
- `test_tensor_dtype` — tensor is `float32`
- `test_label_dtype` — label is `torch.long` scalar
- `test_normalization_mean` — mean ≈ 0 after normalization
- `test_normalization_std` — std ≈ 1 after normalization
- `test_labels_property` — `.labels` returns correct list
- `test_subject_ids_filter` — filters to requested IDs only
- `test_missing_file_handled_gracefully` — missing file skipped, no crash
- `test_nii_extension_support` — loads both `.nii` and `.nii.gz`
- `test_sampler_length` — sampler num_samples matches dataset
- `test_balanced_weights` — weights = `N_total / (2 * N_class)`

---

## Phase 3 — Model Architecture (COMPLETE)

### Files Created
- `mdd_reprogramming/model.py` — MDDReprogrammingModel with all four components
- `mdd_reprogramming/tests/test_model.py` — 10 tests for model components

### What Was Implemented
- `PatchEncoder`: Conv3d(1→d_model, patch_size stride) + BN + ReLU + Conv3d(d_model→d_model, 3, pad=1) + BN + ReLU, Kaiming init, output (N, P, d_model) where P=441
- `ReprogrammingLayer`: Linear(d_model→d_model) + ReLU + Dropout + Linear(d_model→d_llm), Kaiming init
- Frozen LLM: AutoModel.from_pretrained, all params frozen, inputs_embeds + attention mask of ones
- Classification Head: AdaptiveAvgPool1d(1) + Linear(d_llm→2)
- `BaselineHead`: AdaptiveAvgPool3d(4,4,4) + Flatten + Linear(d_model*64→256) + ReLU + Linear(256→2)

### Tests (10/10 passed)
- `test_output_shape` — patch encoder → (N, 441, 128)
- `test_kaiming_init` — Conv3d weights non-zero
- `test_output_shape` — reprogramming → (N, 441, 1024)
- `test_trainable` — reprogramming params require grad
- `test_forward_shape` — full model → (N, 2)
- `test_llm_frozen` — all LLM params frozen
- `test_baseline_forward_shape` — baseline → (N, 2)
- `test_baseline_no_llm` — no LLM attribute in baseline
- `test_trainable_params_exclude_llm` — optimizer won't get LLM params
- `test_attention_mask_shape` — mask is (N, P) of all ones

### Cumulative: 26/26 tests passing

### Remaining
- Phase 4: Loss Function
- Phase 5: Training Loop (`train.py`)
- Phase 6: Evaluation (`evaluate.py`)
- Phase 7: Remaining unit tests (`test_loss.py`, `test_train.py`)
