import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

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
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_targets.append(y_b.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    return total_loss / total_samples, f1_score(y_true, y_pred, average="macro", zero_division=0)


@torch.no_grad()
def evaluate_multiclass(model, loader, loss_fn, device) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, total_samples = 0.0, 0
    all_logits, all_targets = [], []

    for x_num_b, x_cat_b, y_b in loader:
        x_num_b, x_cat_b, y_b = x_num_b.to(device), x_cat_b.to(device), y_b.to(device)
        logits = model(x_num_b, x_cat_b)
        total_loss += loss_fn(logits, y_b).item() * y_b.size(0)
        total_samples += y_b.size(0)
        all_logits.append(logits.cpu().numpy())
        all_targets.append(y_b.cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    macro_f1 = f1_score(y_true, logits_np.argmax(axis=1), average="macro", zero_division=0)
    return total_loss / total_samples, macro_f1, logits_np, y_true


def fit_logit_biases_cv(val_logits: np.ndarray, val_true: np.ndarray, n_classes: int, seed: int = 42) -> np.ndarray:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    grid = np.linspace(-2.5, 2.5, 101)
    fold_biases = []

    for tr_idx, _ in skf.split(val_logits, val_true):
        l_tr, y_tr = val_logits[tr_idx], val_true[tr_idx]
        biases = np.zeros(n_classes, dtype=np.float64)
        best_score = f1_score(y_tr, l_tr.argmax(axis=1), average="macro", zero_division=0)
        class_order = np.argsort(np.bincount(y_tr, minlength=n_classes))

        for _ in range(4):
            improved = False
            for c in class_order:
                best_c = biases[c]
                for cand in grid:
                    trial = biases.copy()
                    trial[c] = cand
                    trial -= trial.mean()
                    score = f1_score(y_tr, (l_tr + trial[None, :]).argmax(axis=1), average="macro", zero_division=0)
                    score -= 1e-4 * float(np.sum(trial ** 2))
                    if score > best_score + 1e-6:
                        best_score = score
                        best_c = cand
                        improved = True
                biases[c] = best_c
                biases -= biases.mean()
            if not improved:
                break
        fold_biases.append(biases)

    mean_b = np.mean(fold_biases, axis=0)
    return mean_b - mean_b.mean()


def main():
    SEED = 42
    EPOCHS = 20
    OUT_DIR = Path("outputs/multiclass")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_loader, val_loader, test_loader, meta = create_dataloaders(
        dir_dados="./data",
        alvo="cat",
        seed_split=SEED,
        weight_power=0.30,
    )
    class_names = meta["class_names"]
    num_classes = meta["num_classes"]

    model = TabularTransformer(
        num_features=meta["n_num"],
        vocab_sizes=meta["vocab_sizes"],
        num_classes=num_classes,
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_layers=4,
        dropout=0.10,
        pooling="cls",
        num_tokenizer="plr",
        plr_frequencies=16,
        plr_sigma=1.0,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=meta["class_weights"].to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    history = {"train_loss": [], "train_f1": [], "val_loss": [], "val_f1": []}
    best_val_f1, best_epoch, best_weights = -1.0, 0, None

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_f1 = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        scheduler.step()
        va_loss, va_f1, _, _ = evaluate_multiclass(model, val_loader, loss_fn, device)

        history["train_loss"].append(tr_loss)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(va_loss)
        history["val_f1"].append(va_f1)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS:02d}] - "
            f"Train Loss: {tr_loss:.4f} | Train F1: {tr_f1:.4f} | "
            f"Val Loss: {va_loss:.4f} | Val F1: {va_f1:.4f}"
        )

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

    print(f"\nRestoring best checkpoint from Epoch {best_epoch} (Val Macro F1: {best_val_f1:.4f})")
    model.load_state_dict(best_weights)

    _, _, val_logits, val_true = evaluate_multiclass(model, val_loader, loss_fn, device)
    _, _, test_logits, test_true = evaluate_multiclass(model, test_loader, loss_fn, device)

    # Apply CV-tuned logit offsets from validation
    biases = fit_logit_biases_cv(val_logits, val_true, n_classes=num_classes, seed=SEED)
    val_pred_raw = val_logits.argmax(axis=1)
    test_pred_raw = test_logits.argmax(axis=1)
    val_pred_tuned = (val_logits + biases[None, :]).argmax(axis=1)
    test_pred_tuned = (test_logits + biases[None, :]).argmax(axis=1)

    print("\n--- Multiclass Test Report (Raw) ---")
    print(classification_report(test_true, test_pred_raw, target_names=class_names, digits=4, zero_division=0))

    print("\n--- Multiclass Test Report (CV-Tuned Logit Offsets) ---")
    print(classification_report(test_true, test_pred_tuned, target_names=class_names, digits=4, zero_division=0))

    plot_training_curves(
        history=history,
        best_epoch=best_epoch,
        title="Multiclass Best (PLR + Cosine + Weighted CE)",
        save_path=OUT_DIR / "multiclass_training_curves.png",
    )
    plot_confusion_matrices(
        val_true=val_true,
        val_pred=val_pred_tuned,
        test_true=test_true,
        test_pred=test_pred_tuned,
        class_names=class_names,
        title="Multiclass Best (CV-Tuned)",
        save_path=OUT_DIR / "multiclass_confusion_matrices.png",
    )

    np.savez(
        OUT_DIR / "multiclass_artifacts.npz",
        val_true=val_true,
        val_pred_raw=val_pred_raw,
        val_pred_tuned=val_pred_tuned,
        test_true=test_true,
        test_pred_raw=test_pred_raw,
        test_pred_tuned=test_pred_tuned,
        **{k: np.array(v) for k, v in history.items()},
    )
    summary = {
        "version": "Multiclass Best (PLR + Cosine + Weighted CE)",
        "best_epoch": best_epoch,
        "val_acc": float(accuracy_score(val_true, val_pred_tuned)),
        "val_macro_f1": float(f1_score(val_true, val_pred_tuned, average="macro", zero_division=0)),
        "test_acc": float(accuracy_score(test_true, test_pred_tuned)),
        "test_macro_f1_raw": float(f1_score(test_true, test_pred_raw, average="macro", zero_division=0)),
        "test_macro_f1_tuned": float(f1_score(test_true, test_pred_tuned, average="macro", zero_division=0)),
        "test_weighted_f1": float(f1_score(test_true, test_pred_tuned, average="weighted", zero_division=0)),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved multiclass plots and artifacts to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()