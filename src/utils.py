import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import ASSETS_DIR, MODELS_DIR, REPORTS_DIR


def ensure_dirs():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def save_text_report(path: Path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def plot_confusion_matrix(cm, output_path: Path):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - CNN Handwritten Digit Recognition")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_training_history(history, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
