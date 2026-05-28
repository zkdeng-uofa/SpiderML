"""
Fine-tune an image classifier using knowledge distillation from a teacher model.

Uses the 1000-class pretrained teacher to provide soft label targets for the
100-class student. The student learns from both hard labels (cross-entropy) and
soft labels (KL divergence from teacher), with a fully trainable student backbone.

The teacher's soft outputs encode inter-species visual similarity that hard
labels cannot capture, improving generalization on confusable species.

Usage:
    python scripts/huggingFaceDistillation.py --config configs/distillation_config.json
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from dataclasses import dataclass, field
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

import wandb

logger = logging.getLogger(__name__)


@dataclass
class ScriptTrainingArguments:
    """All tunable hyperparameters for a distillation training run."""

    dataset: str = field(metadata={"help": "HuggingFace Hub dataset path"})
    model: str = field(metadata={"help": "HuggingFace Hub model checkpoint (used for both teacher and student)"})
    learning_rate: float = field(default=5e-4)
    num_train_epochs: int = field(default=25)
    batch_size: int = field(default=32)
    gradient_accumulation_steps: int = field(default=4)
    warmup_ratio: float = field(default=0.1, metadata={"help": "Fraction of total steps used for warmup"})
    seed: int = field(default=42)
    push_to_hub: bool = field(default=False)
    report_to: str = field(default="wandb")
    output_dir: str = field(default="")
    freeze_through_stage: int = field(default=2, metadata={
        "help": "Freeze encoder stages 0..N (inclusive) plus embeddings. -1 = no freezing."
    })
    distill_alpha: float = field(default=0.5, metadata={
        "help": "Weight for hard label loss. (1 - alpha) is the weight for soft label loss."
    })
    distill_temperature: float = field(default=3.0, metadata={
        "help": "Temperature for softening teacher/student distributions. Higher = softer."
    })


def parse_args() -> ScriptTrainingArguments:
    """Load training arguments from a JSON config file."""
    parser = argparse.ArgumentParser(description="Fine-tune with knowledge distillation")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    cli_args = parser.parse_args()

    with open(cli_args.config) as f:
        config = json.load(f)

    hf_parser = HfArgumentParser(ScriptTrainingArguments)
    (script_args,) = hf_parser.parse_dict(config)
    return script_args


def collate_fn(examples):
    """Stack pixel values and convert label keys for the model."""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, and F1 (macro) via sklearn."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": recall_score(labels, predictions, average="macro", zero_division=0),
        "f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def make_preprocess_fn(transforms_pipeline):
    """Factory that returns a set_transform-compatible preprocessing function."""
    def preprocess(example_batch):
        example_batch["pixel_values"] = [
            transforms_pipeline(image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch
    return preprocess


def freeze_backbone_stages(model, freeze_through_stage: int):
    """Freeze embeddings and encoder stages 0 through `freeze_through_stage` (inclusive)."""
    if freeze_through_stage < 0:
        total = sum(p.numel() for p in model.parameters())
        return total, total, 0

    for param in model.convnextv2.embeddings.parameters():
        param.requires_grad = False

    num_stages = len(model.convnextv2.encoder.stages)
    if freeze_through_stage >= num_stages:
        raise ValueError(
            f"freeze_through_stage={freeze_through_stage} but model only has "
            f"{num_stages} stages (0..{num_stages - 1})."
        )

    for i in range(freeze_through_stage + 1):
        for param in model.convnextv2.encoder.stages[i].parameters():
            param.requires_grad = False

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def build_teacher_class_indices(teacher_model, student_labels):
    """Map student label names to teacher class indices.

    Returns a list where index i is the teacher's class index for student class i.
    Raises ValueError if any student label is not found in the teacher.
    """
    teacher_label2id = teacher_model.config.label2id
    indices = []
    missing = []
    for label in student_labels:
        if label in teacher_label2id:
            indices.append(teacher_label2id[label])
        else:
            missing.append(label)

    if missing:
        raise ValueError(
            f"{len(missing)} student labels not found in teacher model: {missing[:10]}..."
        )

    return indices


class DistillationTrainer(Trainer):
    """Trainer subclass that adds knowledge distillation loss.

    Combines hard label cross-entropy with KL divergence from a frozen teacher
    model's soft predictions. The teacher runs inference on the same batch and
    its logits (for the student's subset of classes) are used as soft targets.
    """

    def __init__(self, teacher_model, teacher_class_indices, alpha, temperature, **kwargs):
        super().__init__(**kwargs)
        self.teacher = teacher_model
        self.teacher_class_indices = teacher_class_indices
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        pixel_values = inputs["pixel_values"]

        # Student forward pass
        student_outputs = model(pixel_values=pixel_values)
        student_logits = student_outputs.logits

        # Hard label loss (standard cross-entropy)
        hard_loss = F.cross_entropy(student_logits, labels)

        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher(pixel_values=pixel_values)
            # Extract only the logits for the 100 student classes from the 1000 teacher classes
            teacher_logits = teacher_outputs.logits[:, self.teacher_class_indices]

        # Soft label loss (KL divergence at temperature T)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean")
        # Scale by T^2 to keep gradient magnitudes consistent with hard loss
        soft_loss = soft_loss * (self.temperature ** 2)

        # Combined loss
        loss = self.alpha * hard_loss + (1.0 - self.alpha) * soft_loss

        return (loss, student_outputs) if return_outputs else loss


def main(config: ScriptTrainingArguments):
    # Enable TF32 for faster matmul/convolutions on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # WandB setup (conditional on config.report_to)
    if config.report_to == "wandb":
        wandb_key = os.environ.get("WANDB_API_KEY")
        if wandb_key:
            wandb.login(key=wandb_key)
        wandb.init(project="spidersML")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    model_checkpoint = config.model

    # Load dataset
    dataset = load_dataset(config.dataset)

    # Build label mappings
    labels = dataset["train"].features["label"].names
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    # Load teacher model (1000-class, frozen, eval mode)
    logger.info("Loading teacher model: %s", model_checkpoint)
    teacher_model = AutoModelForImageClassification.from_pretrained(model_checkpoint)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Build teacher-to-student class index mapping
    teacher_class_indices = build_teacher_class_indices(teacher_model, labels)
    logger.info(
        "Mapped %d student classes to teacher indices (teacher has %d classes)",
        len(teacher_class_indices),
        len(teacher_model.config.label2id),
    )

    # Load image processor and resolve size/crop dimensions
    image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
    if "height" in image_processor.size:
        size = (image_processor.size["height"], image_processor.size["width"])
        crop_size = size
    elif "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
        crop_size = (size, size)
    else:
        raise ValueError(
            f"Unexpected image_processor.size format: {image_processor.size}. "
            f"Expected keys 'height'/'width' or 'shortest_edge'."
        )

    # Build transform pipelines (torchvision v2 API)
    normalize = v2.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

    train_transforms = v2.Compose([
        v2.RandomResizedCrop(crop_size),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ])

    val_transforms = v2.Compose([
        v2.Resize(size),
        v2.CenterCrop(crop_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ])

    # Split dataset: 80% train, 10% val, 10% test (stratified)
    splits1 = dataset["train"].train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
    splits2 = splits1["test"].train_test_split(test_size=0.5, seed=42, stratify_by_column="label")
    train_ds = splits1["train"]
    val_ds = splits2["train"]
    test_ds = splits2["test"]

    # Apply lazy preprocessing via set_transform
    train_ds.set_transform(make_preprocess_fn(train_transforms))
    val_ds.set_transform(make_preprocess_fn(val_transforms))

    # Load student model (100-class, with mismatched head reinitialized)
    logger.info("Loading student model: %s (with %d-class head)", model_checkpoint, len(labels))
    student_model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    # Freeze student backbone stages
    total, trainable, frozen = freeze_backbone_stages(student_model, config.freeze_through_stage)
    if config.freeze_through_stage >= 0:
        logger.info(
            "Student frozen through stage %d: %.1fM trainable / %.1fM total (%.1f%% trainable)",
            config.freeze_through_stage,
            trainable / 1e6,
            total / 1e6,
            100 * trainable / total,
        )

    # Move teacher to same device as student (Trainer handles student placement)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device)
    logger.info("Teacher on device: %s", device)

    model_name = model_checkpoint.split("/")[-1]

    # Build output directory
    freeze_tag = f"freeze{config.freeze_through_stage}" if config.freeze_through_stage >= 0 else "full"
    output_dir = config.output_dir or f"{model_name}-{freeze_tag}-distill-{config.dataset.split('/')[-1]}"
    logger.info("Output directory: %s", output_dir)
    logger.info("Distillation: alpha=%.2f, temperature=%.1f", config.distill_alpha, config.distill_temperature)

    # Compute warmup_steps from warmup_ratio
    steps_per_epoch = len(train_ds) // config.batch_size
    total_steps = steps_per_epoch * config.num_train_epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    logger.info("Warmup steps: %d (%.0f%% of %d total)", warmup_steps, config.warmup_ratio * 100, total_steps)

    # Convert teacher class indices to a tensor on the right device
    teacher_idx_tensor = torch.tensor(teacher_class_indices, dtype=torch.long, device=device)

    # Training arguments (RTX 5090 optimized: bf16 + torch_compile)
    training_args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_train_epochs,
        warmup_steps=warmup_steps,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to=config.report_to,
        push_to_hub=config.push_to_hub,
        hub_token=hf_token,
        bf16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        torch_compile=True if config.freeze_through_stage < 0 else False,
    )

    # Build distillation trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        teacher_class_indices=teacher_idx_tensor,
        alpha=config.distill_alpha,
        temperature=config.distill_temperature,
        model=student_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    logger.info("Using device: %s", training_args.device)

    try:
        # Train
        train_results = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()

        # Evaluate on test set (evaluation uses standard CE, not distillation loss)
        test_ds.set_transform(make_preprocess_fn(val_transforms))
        metrics = trainer.evaluate(eval_dataset=test_ds)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)
    finally:
        if config.report_to == "wandb":
            wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = parse_args()
    set_seed(config.seed)
    main(config)
