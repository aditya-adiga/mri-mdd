# MDD Classification via Modality Reprogramming

Classifying Major Depressive Disorder (MDD) from 3D structural MRI using a **Modality Reprogramming** framework with a frozen LLM backbone.

## Approach

This project adapts the modality reprogramming technique — originally applied to EEG signals (Babu et al., 2026) — to 3D grey matter MRI volumes. Instead of training a model from scratch, we repurpose a frozen large language model (Qwen 0.5B) as a feature extractor by learning to translate MRI patch embeddings into the LLM's token space.

**Pipeline:**

1. **3D Patch Encoder** — Divides 121×145×121 grey matter volumes into non-overlapping cubic patches using 3D convolutions
2. **Reprogramming Layer** — An MLP that maps patch features into the LLM's embedding space
3. **Frozen LLM Backbone** — Qwen 0.5B processes reprogrammed MRI tokens as if they were language, extracting high-level representations
4. **Classification Head** — Pools LLM output and produces MDD vs. Healthy Control predictions

Only the patch encoder, reprogramming layer, and classification head are trained. The LLM remains completely frozen.

## Dataset

- **Source:** [REST-meta-MDD](http://rfmri.org/REST-meta-MDD) (DIRECT Consortium)
- **Subjects:** 2,380 total — 1,276 MDD, 1,104 Healthy Controls
- **Input:** Preprocessed 3D grey matter images (VBM, MNI space, 8mm FWHM smoothing)
- **Volume dimensions:** 121 × 145 × 121 voxels, single channel

## Project Structure

```
mdd_reprogramming/
├── config.py          # All hyperparameters via argparse
├── dataset.py         # MRIDataset + WeightedRandomSampler
├── model.py           # MDDReprogrammingModel + loss functions
├── train.py           # 10-fold CV training pipeline
├── evaluate.py        # Standalone evaluation script
└── tests/             # 49 unit tests
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Train with LLM backbone:**
```bash
python -m mdd_reprogramming.train \
    --data_dir data/wc1_MDD_VBM_dataset \
    --labels_csv data/labels.csv \
    --epochs 50
```

**Train baseline (no LLM):**
```bash
python -m mdd_reprogramming.train --baseline --no_wandb --epochs 50
```

**Evaluate a checkpoint:**
```bash
python -m mdd_reprogramming.evaluate \
    --checkpoint checkpoints/best_fold_0.pt \
    --data_dir data/wc1_MDD_VBM_dataset \
    --labels_csv data/labels.csv
```

**Run tests:**
```bash
pytest -v
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--patch_size` | 16 | Non-overlapping 3D patch size |
| `--d_model` | 128 | Patch encoder output dim |
| `--d_llm` | 1024 | LLM hidden size |
| `--llm_name` | Qwen/Qwen1.5-0.5B | Frozen LLM backbone |
| `--lr` | 0.001 | Learning rate |
| `--epochs` | 50 | Epochs per fold |
| `--n_folds` | 10 | Cross-validation folds |
| `--use_custom_loss` | True | Softmax-before-CE loss variant |

## Metrics

Evaluated on a 10% blind test set: Accuracy, Sensitivity, Specificity, F1, AUROC, AUPRC.

## Google Colab

To run on Colab with GPU:
```python
!git clone https://github.com/aditya-adiga/mri-mdd.git
%cd mri-mdd
!pip install -r requirements.txt
# Upload/mount your data, then:
!python -m mdd_reprogramming.train --no_wandb --epochs 10
```
