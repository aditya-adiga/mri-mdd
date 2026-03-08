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

---

## Phase 4 — Loss Function (COMPLETE)

### Files Modified
- `mdd_reprogramming/model.py` — added `compute_class_weights`, `CustomCrossEntropyLoss`, `build_loss`

### Files Created
- `mdd_reprogramming/tests/test_loss.py` — 11 tests for loss functions

### What Was Implemented
- `compute_class_weights(labels)`: returns `N_total / (2 * N_class)` as float32 tensor
- `CustomCrossEntropyLoss`: applies `softmax(logits, dim=1)` before `nn.CrossEntropyLoss`
- `build_loss(labels, use_custom_loss)`: factory that returns either custom or standard CE with class weights

### Tests (11/11 passed)
- 3 tests for `compute_class_weights` (balanced, imbalanced, dtype)
- 4 tests for `CustomCrossEntropyLoss` (scalar, no NaN, no NaN with weights, differentiable)
- 4 tests for `build_loss` (custom type, standard type, standard scalar, both finite)

### Cumulative: 37/37 tests passing

---

## Phase 5 — Training Loop (COMPLETE)

### Files Created
- `mdd_reprogramming/train.py` — full training pipeline with CV and blind test evaluation
- `mdd_reprogramming/tests/test_train.py` — 6 tests for training pipeline

### Files Modified
- `mdd_reprogramming/model.py` — fixed `compute_class_weights` to handle missing classes via `minlength`

### What Was Implemented
- `compute_metrics()`: Accuracy, Sensitivity, Specificity, F1, AUROC, AUPRC
- `train_one_epoch()`: single epoch training loop
- `evaluate()`: model evaluation with all metrics
- `train_fold()`: full fold training with checkpoint saving and wandb logging
- `evaluate_on_test()`: evaluate all fold checkpoints on blind test set, report mean ± std
- `main()`: full pipeline — seed, device, wandb init, dataset loading, 10% blind holdout, K-fold CV, test evaluation
- wandb integration: config, per-epoch logs, per-fold summaries, best AUROC/epoch, model artifacts
- Only `requires_grad=True` params passed to AdamW optimizer

### Tests (6/6 passed)
- `test_perfect_predictions` — all metrics = 1.0 on perfect input
- `test_returns_all_keys` — metrics dict has all 6 expected keys
- `test_returns_float` — train_one_epoch returns float loss
- `test_smoke_baseline_2_epochs` — 2 epochs on 10 synthetic samples completes
- `test_optimizer_only_trainable_params` — optimizer excludes frozen LLM params
- `test_checkpoint_saved` — checkpoint files saved and loadable

### Cumulative: 43/43 tests passing

---

## Phase 6 — Evaluation (COMPLETE)

### Files Created
- `mdd_reprogramming/evaluate.py` — standalone evaluation script
- `mdd_reprogramming/tests/test_evaluate.py` — 6 tests for evaluation

### Files Modified
- `mdd_reprogramming/config.py` — added `--checkpoint` argument (default None)

### What Was Implemented
- `run_evaluation(args)`: loads checkpoint, reproduces blind test split, runs inference, reports all 6 metrics
- `main(args)`: entry point with logging setup
- Reuses `evaluate()` and `compute_metrics()` from train.py
- Same seed reproduces identical test split as training

### Tests (6/6 passed)
- `test_checkpoint_arg_parsed` — --checkpoint parsed as Path
- `test_checkpoint_default_none` — defaults to None
- `test_missing_checkpoint_raises` — ValueError without --checkpoint
- `test_returns_all_metrics` — all 6 metric keys present
- `test_metrics_in_valid_range` — all values in [0, 1]
- `test_reproducible_with_seed` — same seed = same results

### Cumulative: 49/49 tests passing

### Remaining
- Phase 7: Definition of Done checks
