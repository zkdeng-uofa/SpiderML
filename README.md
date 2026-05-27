# SpiderML

Code for *AraNet: A Two-Stage Convolutional Neural Network for Spider Species Classification* (Deng & Rodriguez, 2025).

AraNet is a two-stage fine-tuning pipeline built on ConvNeXtV2. The first stage trains a base model on 1,000 spider species (~549k images). The second stage freezes the early backbone layers and fine-tunes only the later stages onto a smaller target set. This approach achieves **97.4% accuracy on 100 species** and **99.5% accuracy on Australian poisonous spiders**, outperforming single-stage baselines by a noticeable margin.

---

## Setup

Dependencies are managed with [pixi](https://pixi.sh):

```bash
pixi install
```

If you prefer plain pip, see `requirements.txt` (unmaintained — pixi is the source of truth).

---

## Training

Each script takes a JSON config. The configs in `configs/` correspond to the experiments in `results/`.

```bash
# Full fine-tune (baseline)
pixi run train

# Partial freeze — best result (exp006, 97.4% test accuracy)
pixi run train-freeze

# Knowledge distillation
pixi run train-distill
```

Or call scripts directly:

```bash
python scripts/huggingFacePartialFreeze.py --config configs/partial_freeze_config.json
python scripts/huggingFaceTemplate.py --config configs/train_config.json
python scripts/huggingFaceDistillation.py --config configs/distillation_config.json
```

Datasets and pretrained checkpoints are pulled automatically from Hugging Face Hub (`zkdeng/`).

---

## Results

| ID | Method | Species | Epochs | Trainable Params | Test Acc | Test F1 |
|----|--------|---------|--------|-----------------|----------|---------|
| exp001 | Full fine-tune | 5 | 5 | 27.9M (100%) | 86.0% | 85.8% |
| exp002 | Full fine-tune (spider backbone) | 5 | 5 | 87.7M (100%) | 96.0% | 95.9% |
| exp003 | Full fine-tune | 100 | 5 | 87.8M (100%) | 94.6% | 94.6% |
| exp004 | Full fine-tune | 100 | 10 | 87.8M (100%) | 95.7% | 95.7% |
| exp005 | Full fine-tune | 100 | 25 | 87.8M (100%) | 96.0% | 96.0% |
| **exp006** | **Partial freeze (stage 2)** | **100** | **25** | **27.6M (31%)** | **97.4%** | **97.4%** |
| exp007 | Knowledge distillation | 100 | 25 | 27.6M (31%) | 92.0% | 91.1% |

The partial freeze configuration (exp006) is the best-performing setup. Freezing the first three ConvNeXtV2 stages acts as a regularizer on the small per-class dataset (80 training images/class), and also cuts training time by ~15% compared to full fine-tune at 25 epochs.

Knowledge distillation (exp007) underperformed — when teacher and student share the same pretrained backbone, the soft labels are largely redundant and the KL divergence term hurts more than it helps.

Full per-epoch metrics and hyperparameters for each experiment are in `results/`.

---

## Project Layout

```
configs/         JSON training configs
results/         Per-experiment JSON logs + summary CSV
scripts/
  huggingFaceTemplate.py       Full fine-tune
  huggingFacePartialFreeze.py  Partial backbone freeze
  huggingFaceDistillation.py   Knowledge distillation
  evaluate_model.py            Standalone evaluation
  plot_training_curves.py      Training curve plots
paper/           LaTeX source for the paper
SLURM/           HPC job scripts (4x Volta GPU, torchrun)
```


