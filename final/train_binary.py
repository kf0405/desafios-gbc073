import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

from dataset import create_dataloaders
from model import TabularTransformer
from plotting import plot_training_curves, plot_confusion_matrices


def train_one_epoch(model, loader, loss_fn, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss, total_samples = 0.0, 0
    all_preds, all_targets = [], []

    for x_num_b, x_cat_b, y_b in loader:
        x_num_b, x_cat_b, y_b = x_num_b.to(device), x_cat_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(x_num_b, x_cat_b)
        loss = loss_fn(logits, y_b)
        loss.backward()
        optimizer.step()

        bs = y_b.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        all_preds.append((torch.sigmoid(logits) >= 0.5).long().view(-1).cpu().numpy())
        all_targets.append(y_b.long().view(-1).cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    return total_loss / total_samples, f1_score(y_true, y_pred, average="macro", zero_division=0)


@torch.no_grad()
def evaluate_binary(model, loader, loss_fn, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, total_samples = 0.0, 0
    all_probs, all_targets = [], []

    for x_num_b, x_cat_b, y_b in loader:
        x_num_b, x_cat_b, y_b = x_num_b.to(device), x_cat_b.to(device), y_b.to(device)
        logits = model(x_num_b, x_cat_b)
        total_loss += loss_fn(logits, y_b).item() * y_b.size(0)
        total_samples += y_b.size(0)
        all_probs.append(torch.sigmoid(logits).view(-1).cpu().numpy())
        all_targets.append(y_b.long().view(-1).cpu().numpy())

    return total_loss / total_samples, np.concatenate(all_probs), np.concatenate(all_targets)


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_th, best_f1 = 0.5, -1.0
    for th in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_true, (y_prob >= th).astype(np.int64), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = float(th)
    return best_th, best_f1


def main():
    SEED = 42
    EPOCHS = 10
    OUT_DIR = Path("outputs/binary")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_loader, val_loader, test_loader, meta = create_dataloaders(
        dir_dados="./data",
        alvo="bin",
        seed_split=SEED,
        weight_power=0.50,
    )
    class_names = meta["class_names"]

    model = TabularTransformer(
        num_features=meta["n_num"],
        vocab_sizes=meta["vocab_sizes"],
        num_classes=1,
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_layers=4,
        dropout=0.10,
        pooling="cls",
        num_tokenizer="linear",
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=meta["pos_weight"].to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history = {"train_loss": [], "train_f1": [], "val_loss": [], "val_f1": []}
    best_val_f1, best_epoch, best_weights = -1.0, 0, None

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_f1 = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        va_loss, va_prob, va_true = evaluate_binary(model, val_loader, loss_fn, device)
        th_ep, va_f1 = find_best_threshold(va_true, va_prob)

        history["train_loss"].append(tr_loss)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(va_loss)
        history["val_f1"].append(va_f1)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS:02d}] - "
            f"Train Loss: {tr_loss:.4f} | Train F1: {tr_f1:.4f} | "
            f"Val Loss: {va_loss:.4f} | Val Macro F1: {va_f1:.4f} (th={th_ep:.2f}) | "
            f"Val AUC: {roc_auc_score(va_true, va_prob):.4f}"
        )

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

    print(f"\nRestoring best checkpoint from Epoch {best_epoch} (Val Macro F1: {best_val_f1:.4f})")
    model.load_state_dict(best_weights)

    _, val_prob, val_true = evaluate_binary(model, val_loader, loss_fn, device)
    _, test_prob, test_true = evaluate_binary(model, test_loader, loss_fn, device)

    best_th, _ = find_best_threshold(val_true, val_prob)
    val_pred = (val_prob >= best_th).astype(np.int64)
    test_pred = (test_prob >= best_th).astype(np.int64)

    print(f"\n--- Binary Test Report (threshold={best_th:.2f}) ---")
    print(classification_report(test_true, test_pred, target_names=class_names, digits=4, zero_division=0))

    plot_training_curves(
        history=history,
        best_epoch=best_epoch,
        title="Binary (Linear + BCEWithLogitsLoss)",
        save_path=OUT_DIR / "binary_training_curves.png",
    )
    plot_confusion_matrices(
        val_true=val_true,
        val_pred=val_pred,
        test_true=test_true,
        test_pred=test_pred,
        class_names=class_names,
        title=f"Binary (th={best_th:.2f})",
        save_path=OUT_DIR / "binary_confusion_matrices.png",
    )

    np.savez(
        OUT_DIR / "binary_artifacts.npz",
        val_true=val_true,
        val_pred=val_pred,
        val_prob=val_prob,
        test_true=test_true,
        test_pred=test_pred,
        test_prob=test_prob,
        **{k: np.array(v) for k, v in history.items()},
    )
    summary = {
        "version": "Binary (Linear + BCEWithLogitsLoss)",
        "best_epoch": best_epoch,
        "threshold": best_th,
        "val_acc": float(accuracy_score(val_true, val_pred)),
        "val_macro_f1": float(f1_score(val_true, val_pred, average="macro", zero_division=0)),
        "val_auc": float(roc_auc_score(val_true, val_prob)),
        "test_acc": float(accuracy_score(test_true, test_pred)),
        "test_macro_f1": float(f1_score(test_true, test_pred, average="macro", zero_division=0)),
        "test_weighted_f1": float(f1_score(test_true, test_pred, average="weighted", zero_division=0)),
        "test_auc": float(roc_auc_score(test_true, test_prob)),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved binary plots and artifacts to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()