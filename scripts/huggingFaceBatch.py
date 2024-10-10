import numpy as np
import torch
import torchvision.transforms as transforms
import os
import json
import argparse
import threading
import time
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import (
    AutoImageProcessor,
    set_seed,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    HfArgumentParser
)
import wandb  # Ensure wandb is imported


@dataclass
class ScriptTrainingArguments:
    """
    Arguments pertaining to this script.
    """
    dataset: str = field(
        default=None,
        metadata={"help": "Name of dataset from HG hub"}
    )
    model: str = field(
        default=None,
        metadata={"help": "Name of model from HG hub"}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "Learning rate for training"}
    )
    num_train_epochs: int = field(
        default=5,
        metadata={"help": "Number of training epochs"}
    )
    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size of training epochs"}
    )


def parse_HF_args():
    """
    Parse Hugging Face arguments from a JSON file.
    """
    parser = argparse.ArgumentParser(description="Run Hugging Face model with JSON config")
    parser.add_argument("--config", type=str, required=True, help="Path to the config JSON file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        json_args = json.load(f)
    
    hf_parser = HfArgumentParser(ScriptTrainingArguments)
    script_args = hf_parser.parse_dict(json_args)
    return script_args[0]


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred):
    metric1 = load_metric("accuracy")
    metric2 = load_metric("precision")
    metric3 = load_metric("recall")
    metric4 = load_metric("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric1.compute(predictions=predictions, references=labels)["accuracy"]
    precision = metric2.compute(predictions=predictions, references=labels, average="macro")["precision"]
    recall = metric3.compute(predictions=predictions, references=labels, average="macro")["recall"]
    f1 = metric4.compute(predictions=predictions, references=labels, average="macro")["f1"]
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def try_training_with_batch_size(script_args, train_ds, val_ds, test_ds, model, image_processor, batch_size, device):
    """
    Attempts to train with the given batch size and returns success or failure.
    Handles OutOfMemoryError by reducing the batch size.
    """
    try:
        print(f"Trying with batch size: {batch_size}")

        # Initialize W&B before each retry attempt
        wandb.init(project="spidersML", reinit=True)
        wandb.config.update({
            "model_checkpoint": script_args.model,
            "batch_size": batch_size,
            "learning_rate": script_args.learning_rate,
            "num_train_epochs": script_args.num_train_epochs,
        })

        args = TrainingArguments(
            output_dir=f"{script_args.num_train_epochs}-finetuned-{script_args.dataset.split('/')[-1]}",
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=script_args.learning_rate,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=script_args.num_train_epochs,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            report_to="wandb",  # Report directly to W&B during each retry
            push_to_hub=True  # Push to Hugging Face Hub
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=image_processor,
            compute_metrics=compute_metrics,
            data_collator=collate_fn,
        )

        # Start training
        train_results = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()

        trainer = Trainer(
            model,
            args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            tokenizer=image_processor,
            compute_metrics=compute_metrics,
            data_collator=collate_fn,
        )

        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        wandb.finish()
        return True  # Training successful

    except RuntimeError:
        print(f"Out of memory with batch size {batch_size}, reducing batch size.")
        torch.cuda.empty_cache()  # Clear GPU memory cache to free memory
        wandb.finish()  # Finish logging for this attempt, even if it failed
        return False  # Training failed due to OOM


def main(script_args):
    os.environ["HUGGINGFACE_TOKEN"] = "your_huggingface_token"
    
    model_checkpoint = script_args.model
    dataset = load_dataset(script_args.dataset)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    labels = dataset["train"].features["label"].names
    label2id, id2label = {label: i for i, label in enumerate(labels)}, {i: label for i, label in enumerate(labels)}

    image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (image_processor.size["height"], image_processor.size["width"]) if "height" in image_processor.size else (image_processor.size["shortest_edge"], image_processor.size["shortest_edge"])
    
    train_transforms = Compose([RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize])
    val_transforms = Compose([Resize(size), CenterCrop(size), ToTensor(), normalize])
    test_transforms = Compose([Resize(size), CenterCrop(size), ToTensor(), normalize])

    def preprocess_train(example_batch):
        example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    def preprocess_val(example_batch):
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    def preprocess_test(example_batch):
        example_batch["pixel_values"] = [test_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    splits1 = dataset["train"].train_test_split(test_size=0.2)
    splits2 = splits1["test"].train_test_split(test_size=0.5)
    train_ds, val_ds, test_ds = splits1["train"], splits2["train"], splits2["test"]

    train_ds.set_transform(preprocess_train)
    val_ds.set_transform(preprocess_val)
    test_ds.set_transform(preprocess_test)

    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # Loop through batch sizes until one fits in the GPU
    batch_size = 32 
    successful_training = False
    while batch_size > 0:
        successful_training = try_training_with_batch_size(script_args, train_ds, val_ds, test_ds, model, image_processor, batch_size, device)
        if successful_training:
            break
        batch_size //= 2  # Reduce batch size by half for the next attempt

    if not successful_training:
        print("Failed to find a suitable batch size.")
        return  # Exit if no batch size fits the GPU memory

    print(f"Training successful with batch size: {batch_size}")

if __name__ == "__main__":
    set_seed(42)
    args = parse_HF_args()
    main(args)
