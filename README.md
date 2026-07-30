# AraNet: A Two-Stage Convolutional Neural Network for Spider Species Classification

Companion software for **AraNet: A Two-Stage Convolutional Neural Network for Spider Species Classification** by Zi Deng and Jeffrey J. Rodriguez. This repository is intended to make the analysis code, training configuration, and generated result summaries findable and reusable for peer review and publication under the research-only license described below.

AraNet is a two-stage image-classification pipeline for spider species recognition. Stage 1 performs intermediate fine-tuning of a ConvNeXtV2-Base image classifier on a broad 1,000-species spider dataset. Stage 2 freezes the embedding layer and early ConvNeXtV2 encoder stages, and fine-tunes the later layers on a target species set.

## Contents

```text
configs/      JSON configurations
images/       Generated paper figures
results/      Experiment summaries and per-run metrics
scripts/      Training, evaluation, and plotting scripts
pixi.toml     Reproducible Pixi environment and task definitions
environment.yml
              Conda environment file for FAIR-style installation
LICENSE       University of Arizona research-only software license
```

## Software License

The software in this repository is shared under the license provided by Tech Launch Arizona / the University of Arizona. The copyright notice is assigned to:

```text
© 2026 Arizona Board of Regents on behalf of the University of Arizona.
```
 See [LICENSE](LICENSE) for the full license text.

## Data and Model Sources

The manuscript datasets were curated from iNaturalist research-grade observations, using images available through the iNaturalist / AWS Open Data ecosystem. The following datasets are utilized:

| Dataset | Species | Images |
|---|---:|---:|
| iNatSpiders1k | 1,000 | 548,689 |
| iNatSpiders20 | 20 | 10,000 | 
| iNatSpiders100 | 100 | 99,979 | 
| AusSpiders | 9 | 61,362 | 
| TaiSpiders | 96 | 60,755 | 

The repository configuration files reference Hugging Face datasets and checkpoints under the `zkdeng/` namespace, including:

- `zkdeng/spiderTraining100-100`
- `zkdeng/ausSpiders`
- `zkdeng/taiwanSpiders`
- `zkdeng/10-convnextv2-base-22k-384-finetuned-spiderTraining1000-1000`

If a dataset or checkpoint is private at the time you run the code, authenticate with Hugging Face before training:

```bash
huggingface-cli login
```

or set:

```bash
export HF_TOKEN=...
```

The repository does not redistribute iNaturalist images. Users should also review iNaturalist and dataset-specific terms before redistributing data or derived artifacts.

## Installation

### Recommended: Conda

The editor specifically requested Conda-compatible installation to support FAIR reuse, though pixi support is available. Use the included `environment.yml`:

```bash
conda env create -f environment.yml
conda activate spiderml
```

Training is practical only with a CUDA-capable GPU. CPU execution is useful for import checks and small debugging runs, but full training will be slow.

## Authentication and Experiment Tracking

The scripts defaults to run without Weights & Biases. The setting `report_to` is set to `"none"` in the config file. If you want W&B logging, set:

```bash
export WANDB_API_KEY=...
```

For private Hugging Face datasets, models, or Hub uploads, set:

```bash
export HF_TOKEN=...
```

## Running the Pipeline

All scripts take a JSON configuration file. The task commands below use the configs in `configs/`.

### Full fine-tuning baseline

```bash
python scripts/huggingFaceTemplate.py --config configs/train_config.json
```

This loads the configured Hugging Face dataset, performs a stratified 80/10/10 train/validation/test split, fine-tunes all model parameters, selects the best checkpoint by validation accuracy, and evaluates on the held-out test split.

### AraNet partial-freeze training

```bash
python scripts/huggingFacePartialFreeze.py --config configs/partial_freeze_config.json
```

This is the best-performing manuscript configuration for iNatSpiders100. It loads the spider-adapted ConvNeXtV2-Base Stage 1 checkpoint, freezes the embeddings and encoder stages `0..2`, and trains only the later encoder stage, final normalization layer, and classifier head.

### Knowledge distillation experiment

```bash
python scripts/huggingFaceDistillation.py --config configs/distillation_config.json
```

This trains a student model with a combined hard-label cross-entropy loss and teacher soft-label KL-divergence loss. In the reported experiments, this strategy underperformed the partial-freeze configuration.

### Evaluation and paper figures

Evaluate a trained checkpoint and regenerate the confusion matrix / per-class metrics:

```bash
python scripts/evaluate_model.py \
  --model_dir models/10-convnextv2-base-22k-384-finetuned-spiderTraining1000-1000-freeze2-finetuned-spiderTraining100-100 \
  --dataset zkdeng/spiderTraining100-100
```

Regenerate the training-curve figure from saved JSON results:

```bash
python scripts/plot_training_curves.py
```

Outputs are written to `images/` and `results/`.

## Configuration Files

Each file in `configs/` is a JSON object parsed by the training scripts.

Common fields:

| Field | Meaning |
|---|---|
| `dataset` | Hugging Face dataset identifier |
| `model` | Hugging Face model checkpoint or local model path |
| `learning_rate` | AdamW initial learning rate |
| `num_train_epochs` | number of training epochs |
| `batch_size` | per-device batch size |
| `gradient_accumulation_steps` | gradient accumulation for effective batch size |
| `warmup_ratio` | fraction of total steps used for linear warmup |
| `seed` | random seed used for splitting and initialization |
| `push_to_hub` | whether to upload the trained model to Hugging Face Hub |
| `report_to` | `"wandb"` for Weights & Biases or `"none"` for no external logging |
| `freeze_through_stage` | partial-freeze setting; `2` freezes embeddings and stages 0 through 2 |
| `distill_alpha` | hard-label loss weight in distillation |
| `distill_temperature` | softmax temperature for distillation |

The paper's best iNatSpiders100 configuration is `configs/partial_freeze_config.json`.

## Result Files

The `results/` directory contains the analysis-generated files used to summarize experiments.

### `results/experiments.csv`

One row per formal experiment. Important columns include:

| Column | Description |
|---|---|
| `experiment_id` | short experiment identifier used in the paper and result files |
| `script` | training script used |
| `method` | full fine-tune, partial freeze, or knowledge distillation |
| `dataset` | Hugging Face dataset identifier |
| `model` | starting checkpoint |
| `num_classes` | number of target species |
| `num_images_per_class` | target image count per class when applicable |
| `learning_rate`, `num_train_epochs`, `batch_size`, `gradient_accumulation_steps` | major hyperparameters |
| `total_params`, `trainable_params`, `trainable_pct` | model-size and freeze-depth information |
| `best_val_epoch`, `best_val_accuracy`, `best_val_f1` | validation-selection metrics |
| `test_accuracy`, `test_f1`, `test_precision`, `test_recall`, `test_loss` | final held-out test metrics |
| `hardware` | hardware used for the run |
| `wandb_url` | W&B run page, when logged |

### Per-experiment JSON files

Files such as `exp006_100sp_base_freeze2_25ep.json` contain run-specific metadata, including:

- hyperparameters,
- model architecture details,
- trainable/frozen parameter counts,
- software and hardware version notes,
- validation history by epoch,
- final test metrics,
- W&B identifiers, where available,
- experiment notes.

### `results/exp006_per_class_metrics.json`

Per-class precision, recall, F1, and support for the best partial-freeze AraNet model on iNatSpiders100. This file supports the per-class error analysis in the manuscript.

### `images/confusion_matrix.png`

Confusion matrix heatmap generated by `scripts/evaluate_model.py`.

### `images/training_curves.png`

Validation accuracy/loss curves generated by `scripts/plot_training_curves.py` from `exp005` and `exp006`.

## Reported Experimental Results

The principal results currently documented in `results/experiments.csv` and the manuscript are:

| ID | Method | Dataset | Trainable parameters | Test accuracy | Test F1 |
|---|---|---|---:|---:|---:|
| exp005 | full fine-tune | iNatSpiders100 | 87.8M (100%) | 0.960 | 0.9598 |
| exp006 | partial freeze, stages 0-2 | iNatSpiders100 | 27.6M (31.4%) | 0.974 | 0.9740 |
| exp007 | knowledge distillation | iNatSpiders100 | 27.6M (31.4%) | 0.920 | 0.9111 |

The paper also reports AraNet accuracy of 99.54% on AusSpiders and 97.70% on TaiSpiders. Those cross-study comparisons should be interpreted with the manuscript's stated caveat: the AraNet experiments use larger iNaturalist-derived datasets than the compared prior work, so the reported gains reflect both model/pipeline differences and data differences. The iNatSpiders100 comparison is the cleaner controlled comparison because AraNet and the ConvNeXtV2 baseline use the same dataset.

## Reproducibility Notes

- Splits are stratified by species label using an 80/10/10 train/validation/test procedure.
- Training applies `RandomResizedCrop` and `RandomHorizontalFlip`.
- Validation and test preprocessing use resize and center crop.
- Metrics are macro-averaged for precision, recall, and F1.
- Training scripts enable TF32 on CUDA-capable NVIDIA GPUs and use BF16 mixed precision in Hugging Face `TrainingArguments`.
- Exact runtime depends strongly on GPU, driver, PyTorch build, storage, and dataset access speed.
- Some external resources may require Hugging Face or W&B authentication.

## Citation

If you use this code, please cite the manuscript when it becomes available. Until publication, cite the repository and the submitted manuscript title:

```text
Deng, Z.; Rodriguez, J. J. AraNet: A Two-Stage Convolutional Neural Network
for Spider Species Classification. Submitted to Arthropoda, 2026.
```

## Contact

Zi Deng, zkdeng@arizona.edu, Department of Electrical and Computer Engineering, The University of Arizona.
