"""
Evaluate a trained model on the test set, producing:
  - Confusion matrix heatmap (saved as PNG)
  - Per-class precision/recall/F1 report (saved as JSON + printed as LaTeX)
  - Overall test accuracy verification

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --model_dir path/to/model --dataset zkdeng/spiderTraining100-100
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from torchvision.transforms import v2
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
)
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def make_preprocess_fn(transforms_pipeline):
    def preprocess(example_batch):
        example_batch["pixel_values"] = [
            transforms_pipeline(image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch
    return preprocess


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set")
    parser.add_argument("--model_dir", type=str,
                        default="models/10-convnextv2-base-22k-384-finetuned-spiderTraining1000-1000-freeze2-finetuned-spiderTraining100-100",
                        help="Path to saved model directory")
    parser.add_argument("--dataset", type=str, default="zkdeng/spiderTraining100-100",
                        help="HuggingFace dataset name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="images",
                        help="Directory for output figures")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory for output metrics JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    set_seed(args.seed)

    # Load dataset and split identically to training
    logger.info("Loading dataset: %s", args.dataset)
    dataset = load_dataset(args.dataset)

    splits1 = dataset["train"].train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
    splits2 = splits1["test"].train_test_split(test_size=0.5, seed=42, stratify_by_column="label")
    test_ds = splits2["test"]

    label_names = dataset["train"].features["label"].names
    num_classes = len(label_names)
    logger.info("Test set: %d samples, %d classes", len(test_ds), num_classes)

    # Load image processor and build val transforms
    image_processor = AutoImageProcessor.from_pretrained(args.model_dir)
    if "height" in image_processor.size:
        size = (image_processor.size["height"], image_processor.size["width"])
        crop_size = size
    elif "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
        crop_size = (size, size)
    else:
        raise ValueError(f"Unexpected image_processor.size format: {image_processor.size}")

    normalize = v2.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    val_transforms = v2.Compose([
        v2.Resize(size),
        v2.CenterCrop(crop_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ])
    test_ds.set_transform(make_preprocess_fn(val_transforms))

    # Load model
    logger.info("Loading model from: %s", args.model_dir)
    model = AutoModelForImageClassification.from_pretrained(args.model_dir)

    # Run inference using Trainer.predict
    eval_args = TrainingArguments(
        output_dir="/tmp/eval_output",
        per_device_eval_batch_size=32,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=collate_fn,
        processing_class=image_processor,
    )

    logger.info("Running inference on test set...")
    output = trainer.predict(test_ds)
    predictions = np.argmax(output.predictions, axis=-1)
    true_labels = output.label_ids

    # Overall metrics
    overall_acc = accuracy_score(true_labels, predictions)
    overall_f1 = f1_score(true_labels, predictions, average="macro", zero_division=0)
    logger.info("Overall test accuracy: %.4f", overall_acc)
    logger.info("Overall test F1 (macro): %.4f", overall_f1)

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(cm, cmap="Blues", norm=LogNorm(vmin=max(cm.min(), 0.1), vmax=cm.max()))
    ax.set_xlabel("Predicted Species", fontsize=14)
    ax.set_ylabel("True Species", fontsize=14)
    ax.set_title(f"Confusion Matrix — AraNet on iNatSpiders{num_classes}\n"
                 f"(Test Accuracy: {overall_acc:.1%})", fontsize=16, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Count (log scale)", fontsize=12)
    tick_positions = list(range(0, num_classes, 10))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_positions)
    ax.set_yticklabels(tick_positions)

    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=300, bbox_inches="tight")
    logger.info("Saved confusion matrix: %s", cm_path)
    plt.close(fig)

    # Per-class metrics
    report = classification_report(true_labels, predictions, target_names=label_names,
                                   output_dict=True, zero_division=0)

    # Build per-class list sorted by F1
    per_class = []
    for name in label_names:
        if name in report:
            entry = report[name]
            per_class.append({
                "species": name.replace("_", " "),
                "precision": round(entry["precision"], 4),
                "recall": round(entry["recall"], 4),
                "f1": round(entry["f1-score"], 4),
                "support": entry["support"],
            })

    per_class_sorted = sorted(per_class, key=lambda x: x["f1"], reverse=True)
    top5 = per_class_sorted[:5]
    bottom5 = per_class_sorted[-5:]

    # Save full per-class report
    metrics_output = {
        "overall_accuracy": overall_acc,
        "overall_f1_macro": overall_f1,
        "per_class": per_class_sorted,
    }
    metrics_path = os.path.join(args.results_dir, "exp006_per_class_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_output, f, indent=2)
    logger.info("Saved per-class metrics: %s", metrics_path)

    # Print LaTeX table rows
    print("\n" + "=" * 70)
    print("LaTeX rows for Table per_class (Top 5 and Bottom 5 by F1)")
    print("=" * 70)
    print("\\multicolumn{4}{l}{\\textit{Top 5}} \\\\")
    for entry in top5:
        species_latex = "\\textit{" + entry["species"] + "}"
        print(f"{species_latex} & {entry['precision']:.2f} & {entry['recall']:.2f} & {entry['f1']:.2f} \\\\")
    print("\\midrule")
    print("\\multicolumn{4}{l}{\\textit{Bottom 5}} \\\\")
    for entry in bottom5:
        species_latex = "\\textit{" + entry["species"] + "}"
        print(f"{species_latex} & {entry['precision']:.2f} & {entry['recall']:.2f} & {entry['f1']:.2f} \\\\")
    print("=" * 70)


if __name__ == "__main__":
    main()
