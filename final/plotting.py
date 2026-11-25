from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_training_curves(
    history: dict[str, list[float]],
    best_epoch: int,
    title: str,
    save_path: str | Path,
) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Panel 1: Loss
    axes[0].plot(epochs, history["train_loss"], marker="o", linewidth=2, label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], marker="s", linewidth=2, label="Val Loss")
    axes[0].axvline(best_epoch, color="gray", linestyle="--", alpha=0.8, label=f"Best Epoch ({best_epoch})")
    axes[0].set_title(f"{title} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Panel 2: Macro F1
    axes[1].plot(epochs, history["train_f1"], marker="o", linewidth=2, label="Train Macro F1")
    axes[1].plot(epochs, history["val_f1"], marker="s", linewidth=2, label="Val Macro F1")
    axes[1].axvline(best_epoch, color="gray", linestyle="--", alpha=0.8, label=f"Best Epoch ({best_epoch})")
    axes[1].set_title(f"{title} — Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(
    val_true: np.ndarray,
    val_pred: np.ndarray,
    test_true: np.ndarray,
    test_pred: np.ndarray,
    class_names: list[str],
    title: str,
    save_path: str | Path,
) -> None:
    n_classes = len(class_names)
    figsize = (18, 7.5) if n_classes > 2 else (11, 4.8)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    splits = [
        ("Validation", val_true, val_pred),
        ("Test", test_true, test_pred),
    ]

    for ax, (split_name, y_t, y_p) in zip(axes, splits):
        cm_counts = confusion_matrix(y_t, y_p, labels=range(n_classes))
        row_sums = cm_counts.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm_counts, np.maximum(row_sums, 1), dtype=np.float64)

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(f"{title} — {split_name} (Row-Normalized % / Count)")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

        fontsize = 7.5 if n_classes > 2 else 11
        for i in range(n_classes):
            for j in range(n_classes):
                pct = cm_norm[i, j] * 100.0
                cnt = cm_counts[i, j]
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(
                    j, i, f"{pct:.1f}%\n({cnt})",
                    ha="center", va="center", color=color, fontsize=fontsize
                )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)