import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from dataset import create_dataloaders
from model import TabularTransformer


@torch.no_grad()
def evaluate_binary(model: nn.Module, loader, loss_fn: nn.Module, device: torch.device) -> dict:
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

    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)
    y_pred = (y_prob >= 0.5).astype(np.int64)

    return {
        "loss": total_loss / total_samples,
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main():
    SEED = 42
    EPOCHS = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_loader, val_loader, test_loader, meta = create_dataloaders(
        dir_dados="./data",
        alvo="bin",
        seed_split=SEED,
    )

    model = TabularTransformer(
        num_features=meta["n_num"],
        vocab_sizes=meta["vocab_sizes"],
        num_classes=1,
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_layers=4,
        dropout=0.1,
        pooling="cls",
        num_tokenizer="linear",
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_samples = 0.0, 0
        for x_num_b, x_cat_b, y_b in train_loader:
            x_num_b, x_cat_b, y_b = x_num_b.to(device), x_cat_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_num_b, x_cat_b)
            loss = loss_fn(logits, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y_b.size(0)
            total_samples += y_b.size(0)

        val = evaluate_binary(model, val_loader, loss_fn, device)
        print(
            f"Epoch [{epoch:02d}/{EPOCHS:02d}] - "
            f"Train Loss: {total_loss / total_samples:.4f} | "
            f"Val Loss: {val['loss']:.4f} | Val Acc: {val['acc']:.4f} | "
            f"Val Macro F1: {val['macro_f1']:.4f} | Val AUC: {val['roc_auc']:.4f}"
        )

    test = evaluate_binary(model, test_loader, loss_fn, device)
    print(f"\nTest Acc: {test['acc']:.4f} | Test Macro F1: {test['macro_f1']:.4f} | Test AUC: {test['roc_auc']:.4f}")
    print(classification_report(test["y_true"], test["y_pred"], target_names=meta["class_names"], digits=4))


if __name__ == "__main__":
    main()