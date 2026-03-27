# Experiment Results

Spider species image classification experiments using ConvNeXtV2 models fine-tuned on Hugging Face datasets.

## Hardware

All experiments conducted on a single NVIDIA GeForce RTX 5090 (32GB GDDR7, Blackwell architecture) with PyTorch 2.10.0+cu128, BF16 mixed precision, and TF32 enabled.

## Results Summary

| ID | Method | Dataset | Model | LR | Epochs | Eff. Batch | Trainable | Test Acc | Test F1 | Time |
|----|--------|---------|-------|-----|--------|------------|-----------|----------|---------|------|
| exp001 | Full fine-tune | 5sp×100 | ConvNeXtV2-tiny (ImageNet) | 5e-5 | 5 | 32 | 27.9M (100%) | 0.860 | 0.858 | 59s |
| exp002 | Full fine-tune | 5sp×100 | ConvNeXtV2-base (spider-1000) | 5e-5 | 5 | 32 | 87.7M (100%) | 0.960 | 0.959 | 50s |
| exp003 | Full fine-tune | 100sp×100 | ConvNeXtV2-base (spider-1000) | 5e-5 | 5 | 32 | 87.8M (100%) | 0.946 | 0.946 | 3m28s |
| exp004 | Full fine-tune | 100sp×100 | ConvNeXtV2-base (spider-1000) | 5e-4 | 10 | 128 | 87.8M (100%) | 0.957 | 0.957 | 4m53s |
| exp005 | Full fine-tune | 100sp×100 | ConvNeXtV2-base (spider-1000) | 5e-4 | 25 | 128 | 87.8M (100%) | 0.960 | 0.960 | 12m01s |
| **exp006** | **Partial freeze (stage 2)** | **100sp×100** | **ConvNeXtV2-base (spider-1000)** | **5e-4** | **25** | **128** | **27.6M (31%)** | **0.974** | **0.974** | **10m19s** |
| exp007 | Knowledge distillation | 100sp×100 | ConvNeXtV2-base (spider-1000) | 5e-4 | 25 | 128 | 27.6M (31%) | 0.920 | 0.911 | 21m39s |

## Key Findings

1. **Domain-pretrained backbone is critical.** The spider-pretrained ConvNeXtV2-base (exp002) achieved 96.0% test accuracy on 5 species vs 86.0% for the ImageNet-only tiny model (exp001) — an 11.6% absolute improvement from domain pretraining.

2. **Partial freezing outperforms full fine-tuning.** Freezing stages 0–2 (exp006) achieved 97.4% test accuracy, 1.4% higher than the best full fine-tune (exp005, 96.0%), while training 69% fewer parameters. The frozen backbone acts as regularization, preventing early-layer drift on the small dataset (80 training images per class).

3. **Higher learning rate + gradient accumulation helps.** Increasing LR from 5e-5 to 5e-4 with gradient accumulation 4 (exp004 vs exp003) improved test accuracy by 1.1% (94.6% → 95.7%).

4. **Knowledge distillation underperformed.** When teacher and student share the same backbone (exp007), the soft labels provide redundant information. The KL divergence loss dominated early training and degraded final accuracy to 92.0%.

5. **Diminishing returns from longer training.** Going from 10 to 25 epochs on full fine-tune (exp004 → exp005) gained only 0.3% (95.7% → 96.0%) at 2.5× the compute cost. With partial freeze, the best validation accuracy appeared at epoch 6.

## File Structure

```
results/
├── README.md                              # This file
├── experiments.csv                         # All experiments in tabular format (machine-readable)
├── exp001_5sp_tiny_full.json              # Per-experiment detail with epoch history
├── exp002_5sp_base_full.json
├── exp003_100sp_base_full_5ep.json
├── exp004_100sp_base_full_10ep.json
├── exp005_100sp_base_full_25ep.json
├── exp006_100sp_base_freeze2_25ep.json    # Best result
└── exp007_100sp_base_distill_25ep.json
```

Each JSON file contains:
- Full hyperparameter configuration
- Model architecture details (total/trainable params, frozen stages)
- Hardware and software versions
- Per-epoch validation metrics (accuracy, F1, precision, recall, loss)
- Final test set metrics
- WandB run ID and URL for detailed logs
- Experiment notes and observations

## Reproducing Results

```bash
# Install environment
pixi install

# Run best configuration (partial freeze)
pixi run train-freeze

# Run full fine-tune
pixi run train

# Run distillation
pixi run train-distill
```

## WandB Project

All runs are tracked at: https://wandb.ai/zkdeng-university-of-arizona/spidersML
