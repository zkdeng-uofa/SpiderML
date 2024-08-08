import numpy as np
import torch
import torchvision.transforms as transforms
import os
import wandb

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

@dataclass
class ScriptTrainingArguments:
    """
    Arguments pertaining to this script
    """
    dataset: str = field(
        default=None,
        metadata={"help": "Name of dataset from HG hub"}
    )
    model: str = field(
        default=None,
        metadata={"help": "Name of model from HG hub"}
    )

def parse_HF_args():
    """
    Parse hugging face arguments from the command line
    """
    parser = HfArgumentParser(ScriptTrainingArguments)
    [script_args] = parser.parse_args_into_dataclasses()
    return script_args

def collate_fn(examples):
    """
    Collate the pixel values
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def create_tensor_dataset(dataset):
    pixel_values = torch.stack([pv for pv in dataset["pixel_values"]])
    labels = torch.tensor(dataset["label"])
    return TensorDataset(pixel_values, labels)

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

def main(hubPath, hubModel):
    wandb.login(key="e68d14a1a7b3aed71e0455589cde53c783018f5a")
    wandb.init(project="t5spiders")
    
    os.environ["HUGGINGFACE_TOKEN"] = "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"
    
    model_checkpoint = hubModel
    batch_size = 16

    wandb.config.update({
        "model_checkpoint": model_checkpoint,
        "batch_size": batch_size,
        "learning_rate": 5e-5,
        "num_train_epochs": 5,
    })

    dataset = load_dataset(hubPath)
    #dataset = load_dataset("imagefolder", data_dir=hubPath)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)

    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    image_processor  = AutoImageProcessor.from_pretrained(model_checkpoint)
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    if "height" in image_processor.size:
        size = (image_processor.size["height"], image_processor.size["width"])
        crop_size = size
        max_size = None
    elif "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
        crop_size = (size, size)
        max_size = image_processor.size.get("longest_edge")
    
    train_transforms = Compose(
        [
            RandomResizedCrop(crop_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )
    transform = Compose([
        Resize(size),
        CenterCrop(crop_size),
        ToTensor(),
        normalize,
    ])
    def preprocess_train(example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [
            train_transforms(image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch

    def preprocess_val(example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [
            val_transforms(image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch

    def preprocess_test(example_batch):
        example_batch["pixel_values"] = [
            transform(image.convert("RGB")) for image in example_batch["image"]
    ]
        return example_batch
    
    splits1 = dataset["train"].train_test_split(test_size=0.2)
    splits2 = splits1["test"].train_test_split(test_size=0.5)
    train_ds = splits1['train']
    val_ds = splits2['train']
    test_ds = splits2["test"]

    train_ds.set_transform(preprocess_train)
    val_ds.set_transform(preprocess_val)

    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    model_name = model_checkpoint.split("/")[-1]

    args = TrainingArguments(
        f"{model_name}-finetuned-{hubPath.split('/')[-1]}",
        remove_unused_columns=False,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="wandb",
        push_to_hub=True,
        hub_token="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"
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    test_ds.set_transform(preprocess_test)
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
    return None

if __name__ == "__main__":
    set_seed(42)
    args = parse_HF_args()
    main(args.dataset, args.model)
    #main("zkdeng/t5spiders-1000", "facebook/convnextv2-tiny-22k-384")
