from typing import Literal
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from dados import carregar

CAT = ["proto", "service", "state"]


def onehot_to_indices(X: np.ndarray, cols: list[str], n_num: int) -> np.ndarray:
    idx_cols = []
    offset = n_num

    for col in CAT:
        prefix = f"{col}_"
        matching = [c[len(prefix):] for c in cols if c.startswith(prefix)]
        block = X[:, offset : offset + len(matching)]
        has_cat = block.sum(axis=1) > 0
        indices = np.where(has_cat, block.argmax(axis=1) + 1, 0)
        idx_cols.append(indices)
        offset += len(matching)

    return np.column_stack(idx_cols).astype("int64")


def _make_loader(
    X: np.ndarray,
    Y: np.ndarray,
    cols: list[str],
    n_num: int,
    num_classes: int,
    is_binary: bool,
    normal_idx: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x_num = torch.from_numpy(X[:, :n_num]).float()
    x_cat = torch.from_numpy(onehot_to_indices(X, cols, n_num)).long()

    if is_binary:
        class_idx = Y[:, :num_classes].argmax(axis=1)
        y_bin = (class_idx != normal_idx).astype(np.float32)[:, None]  # 0 = Not Attack, 1 = Attack
        y = torch.from_numpy(y_bin)
    else:
        y = torch.from_numpy(Y[:, :num_classes].argmax(axis=1)).long()

    return DataLoader(TensorDataset(x_num, x_cat, y), batch_size=batch_size, shuffle=shuffle)


def create_dataloaders(
    dir_dados: str = "./data",
    alvo: Literal["cat", "bin"] = "cat",
    seed_split: int = 42,
    frac_val: float = 0.2,
    train_batch_size: int = 256,
    eval_batch_size: int = 512,
    weight_power: float = 0.30,
):
    # Always load 10-class targets from carregar so binary can map Normal -> 0, All 9 Attacks -> 1
    dados = carregar(dir_dados=dir_dados, alvo="cat", seed_split=seed_split, frac_val=frac_val)
    cols = dados["cols"]
    n_num = dados["n_num"]
    vocab_sizes = [sum(c.startswith(f"{col}_") for c in cols) + 1 for col in CAT]

    raw_classes = list(dados["classes"])
    total_classes = len(raw_classes)
    normal_idx = raw_classes.index("Normal")
    y_train_idx = dados["Ytr"][:, :total_classes].argmax(axis=1)

    is_binary = (alvo == "bin")
    if is_binary:
        class_names = ["Not Attack", "Attack"]
        num_classes = 1
        y_train_bin = (y_train_idx != normal_idx)
        n_pos = float(y_train_bin.sum())
        n_neg = float((~y_train_bin).sum())
        pos_weight = torch.tensor([(n_neg / max(n_pos, 1.0)) ** weight_power], dtype=torch.float32)
        class_weights = None
    else:
        class_names = raw_classes
        num_classes = total_classes
        class_counts = np.bincount(y_train_idx, minlength=num_classes).astype(np.float32)
        cw = np.power(np.maximum(class_counts, 1.0), -weight_power)
        cw = cw / cw.mean()
        class_weights = torch.tensor(cw, dtype=torch.float32)
        pos_weight = None

    train_loader = _make_loader(dados["Xtr"], dados["Ytr"], cols, n_num, total_classes, is_binary, normal_idx, train_batch_size, shuffle=True)
    val_loader   = _make_loader(dados["Xva"], dados["Yva"], cols, n_num, total_classes, is_binary, normal_idx, eval_batch_size, shuffle=False)
    test_loader  = _make_loader(dados["Xte"], dados["Yte"], cols, n_num, total_classes, is_binary, normal_idx, eval_batch_size, shuffle=False)

    meta = {
        "n_num": n_num,
        "vocab_sizes": vocab_sizes,
        "num_classes": num_classes,
        "class_names": class_names,
        "class_weights": class_weights,
        "pos_weight": pos_weight,
    }
    return train_loader, val_loader, test_loader, meta