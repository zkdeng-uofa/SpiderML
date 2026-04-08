"""
Generate training curves figure for the AraNet paper.

Reads per-epoch validation metrics from exp005 (full fine-tune) and exp006
(partial freeze) and produces a two-panel plot: accuracy and loss over epochs.

Usage:
    python scripts/plot_training_curves.py
"""

import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def load_eval_history(path):
    with open(path) as f:
        data = json.load(f)
    history = data["eval_history"]
    epochs = [h["epoch"] for h in history]
    accuracy = [h["accuracy"] for h in history]
    loss = [h["loss"] for h in history]
    return epochs, accuracy, loss


def main():
    exp005_path = "results/exp005_100sp_base_full_25ep.json"
    exp006_path = "results/exp006_100sp_base_freeze2_25ep.json"

    ep5, acc5, loss5 = load_eval_history(exp005_path)
    ep6, acc6, loss6 = load_eval_history(exp006_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Validation Accuracy
    ax1.plot(ep5, acc5, "o-", color="#2196F3", linewidth=1.5, markersize=4, label="Full Fine-Tune")
    ax1.plot(ep6, acc6, "s-", color="#FF5722", linewidth=1.5, markersize=4, label="Partial Freeze (Stages 0-2)")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Validation Accuracy", fontsize=12)
    ax1.set_title("Validation Accuracy", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 26)
    ax1.set_ylim(0.87, 0.98)

    # Right panel: Validation Loss
    ax2.plot(ep5, loss5, "o-", color="#2196F3", linewidth=1.5, markersize=4, label="Full Fine-Tune")
    ax2.plot(ep6, loss6, "s-", color="#FF5722", linewidth=1.5, markersize=4, label="Partial Freeze (Stages 0-2)")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Validation Loss", fontsize=12)
    ax2.set_title("Validation Loss", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 26)

    fig.tight_layout(pad=2.0)
    output_path = "images/training_curves.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
