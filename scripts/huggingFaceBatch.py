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

def raise_keyboard_exception():
    """
    Raises a timeout exception to stop the training.
    """
    raise KeyboardInterrupt

def try_training_with_batch_size(script_args, train_ds, val_ds, model, image_processor, batch_size, device):
    """
    Attempts to train with the given batch size and returns success or failure.
    """
    try:
        print(f"Trying with batch size: {batch_size}")
        args = TrainingArguments(
            output_dir=f"{script_args.num_train_epochs}-finetuned-{script_args.dataset.split('/')[-1]}",
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=script_args.learning_rate,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=1,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False
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

        # Start a timer to enforce the 30-second limit
        timeout = 60  # 30 seconds
        timer = threading.Timer(timeout, raise_keyboard_exception)
        timer.start()

        # Test training
        trainer.train()
        timer.cancel() 
        return True  # Training successful
    
    except KeyboardInterrupt as e:
        print(f"Size Found, Training stopped: {e}")
        timer.cancel()
        return True  # Training failed due to timeout
    
    except:
        timer.cancel()
        print(f"Out of memory with batch size {batch_size}, reducing batch size.")
        torch.cuda.empty_cache()  # Clear GPU memory cache to free memory
        return False  # Training failed due to OOM


def main(script_args):
    os.environ["HUGGINGFACE_TOKEN"] = "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"
    
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
    if "height" in image_processor.size:
        size = (image_processor.size["height"], image_processor.size["width"])
        crop_size = size
        max_size = None
    elif "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
        crop_size = (size, size)
        max_size = image_processor.size.get("longest_edge")
        
    train_transforms = Compose([RandomResizedCrop(crop_size), RandomHorizontalFlip(), ToTensor(), normalize])
    val_transforms = Compose([Resize(size), CenterCrop(crop_size), ToTensor(), normalize])
    test_transforms = Compose([Resize(size), CenterCrop(crop_size), ToTensor(), normalize,
    ])

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

    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # Loop through batch sizes until one fits in the GPU
    #batch_size = script_args.batch_size
    batch_size = 64
    successful_training = False
    while batch_size > 0:
        successful_training = try_training_with_batch_size(script_args, train_ds, val_ds, model, image_processor, batch_size, device)
        if successful_training:
            break
        batch_size //= 2  # Reduce batch size by half for the next attempt

    if not successful_training:
        print("Failed to find a suitable batch size.")
        return  # Exit if no batch size fits the GPU memory

    # Once correct batch size is found, initialize W&B and push to Hugging Face
    print(f"Training successful with batch size: {batch_size}")

    # Initialize wandb after finding correct batch size
    wandb.login(key="your_wandb_api_key")
    wandb.init(project="spidersML")

    # Updating wandb configuration with the found batch size and other details
    wandb.config.update({
        "model_checkpoint": model_checkpoint,
        "batch_size": batch_size,
        "learning_rate": script_args.learning_rate,
        "num_train_epochs": script_args.num_train_epochs,
    })

    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # Proceed with the final model training and evaluation
    final_args = TrainingArguments(
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
        report_to="wandb",  # Now report to W&B
        push_to_hub=True,  # Push to Hugging Face hub
    )

    print("Training Batch Size:", batch_size)
    final_trainer = Trainer(
        model=model,
        args=final_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    # Final training and evaluation
    final_train_results = final_trainer.train()
    final_trainer.save_model()
    final_trainer.log_metrics("train", final_train_results.metrics)
    final_trainer.save_metrics("train", final_train_results.metrics)
    final_trainer.save_state()

    test_ds.set_transform(preprocess_test)
    final_trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    metrics = final_trainer.evaluate()
    final_trainer.log_metrics("eval", metrics)
    final_trainer.save_metrics("eval", metrics)

    wandb.finish()

if __name__ == "__main__":
    set_seed(42)
    args = parse_HF_args()
    main(args)
