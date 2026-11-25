import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def main():
    mc_dir = Path("outputs/multiclass")
    bin_dir = Path("outputs/binary")
    out_dir = Path("outputs")

    mc_sum = json.loads((mc_dir / "summary.json").read_text())
    bin_sum = json.loads((bin_dir / "summary.json").read_text())
    mc_art = np.load(mc_dir / "multiclass_artifacts.npz")
    bin_art = np.load(bin_dir / "binary_artifacts.npz")

    mc_classes = [
        "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
        "Generic", "Normal", "Reconnaissance", "Shellcode", "Worms",
    ]
    bin_classes = ["Not Attack", "Attack"]
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.35], hspace=0.35, wspace=0.25)

    # Top-Left: Validation Macro F1 Learning Curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(1, len(mc_art["val_f1"]) + 1), mc_art["val_f1"], marker="o", label="Multiclass Best (10-class)")
    ax1.plot(range(1, len(bin_art["val_f1"]) + 1), bin_art["val_f1"], marker="s", label="Binary (Analysis vs Rest)")
    ax1.set_title("Validation Macro F1 Trajectory")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Macro F1")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Top-Right: Headline Metric Comparison Bar Chart
    ax2 = fig.add_subplot(gs[0, 1])
    metric_names = ["Val Macro F1", "Test Macro F1", "Test Weighted F1", "Test Accuracy"]
    mc_vals = [mc_sum["val_macro_f1"], mc_sum["test_macro_f1_tuned"], mc_sum["test_weighted_f1"], mc_sum["test_acc"]]
    bin_vals = [bin_sum["val_macro_f1"], bin_sum["test_macro_f1"], bin_sum["test_weighted_f1"], bin_sum["test_acc"]]

    x = np.arange(len(metric_names))
    w = 0.36
    b1 = ax2.bar(x - w / 2, mc_vals, width=w, label="Multiclass Best (10-class)")
    b2 = ax2.bar(x + w / 2, bin_vals, width=w, label="Binary (Analysis vs Rest)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_names)
    ax2.set_ylim(0, 1.12)
    ax2.set_ylabel("Score")
    ax2.set_title("Multiclass vs. Binary — Summary Metrics")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)

    # Bottom-Left: Multiclass Test Confusion Matrix (Row-Normalized %)
    ax3 = fig.add_subplot(gs[1, 0])
    cm_mc = confusion_matrix(mc_art["test_true"], mc_art["test_pred_tuned"], labels=range(len(mc_classes)))
    cm_mc_norm = cm_mc / np.maximum(cm_mc.sum(axis=1, keepdims=True), 1)
    im3 = ax3.imshow(cm_mc_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_title(f"Multiclass Test Confusion Matrix (Macro F1 = {mc_sum['test_macro_f1_tuned']:.4f})")
    ax3.set_xticks(range(len(mc_classes)))
    ax3.set_yticks(range(len(mc_classes)))
    ax3.set_xticklabels(mc_classes, rotation=45, ha="right")
    ax3.set_yticklabels(mc_classes)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("True")
    for i in range(len(mc_classes)):
        for j in range(len(mc_classes)):
            val = cm_mc_norm[i, j] * 100.0
            ax3.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=7,
                     color="white" if cm_mc_norm[i, j] > 0.5 else "black")

    # Bottom-Right: Binary Test Confusion Matrix (Row-Normalized % & Count)
    ax4 = fig.add_subplot(gs[1, 1])
    cm_bin = confusion_matrix(bin_art["test_true"], bin_art["test_pred"], labels=[0, 1])
    cm_bin_norm = cm_bin / np.maximum(cm_bin.sum(axis=1, keepdims=True), 1)
    im4 = ax4.imshow(cm_bin_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    ax4.set_title(f"Binary Test Confusion Matrix (Macro F1 = {bin_sum['test_macro_f1']:.4f}, AUC = {bin_sum['test_auc']:.4f})")
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(bin_classes)
    ax4.set_yticklabels(bin_classes)
    ax4.set_xlabel("Predicted")
    ax4.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            val = cm_bin_norm[i, j] * 100.0
            cnt = cm_bin[i, j]
            ax4.text(j, i, f"{val:.1f}%\n({cnt})", ha="center", va="center", fontsize=11,
                     color="white" if cm_bin_norm[i, j] > 0.5 else "black")

    plt.savefig(out_dir / "comparison_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved side-by-side comparison plot to {(out_dir / 'comparison_overview.png').resolve()}")


if __name__ == "__main__":
    main()