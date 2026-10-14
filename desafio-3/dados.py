"""Carga e pré-processamento do UNSW-NB15 (training-set e testing-set).

Espera, em `dados/`:
    UNSW_NB15_training-set.csv   (175.341 linhas)
    UNSW_NB15_testing-set.csv    (82.332 linhas)

- validação = 20% estratificado do training-set (split FIXO, independente da semente do treino);
- o testing-set só é devolvido aqui, quem decide usá-lo (uma vez) é o treino.py;
- estatísticas (log1p, média, desvio, vocabulário das categóricas) ajustadas só no treino.

Alvo multilabel (BCE com logits, uma saída sigmoide por rótulo):
    alvo="cat"      -> multi-hot de attack_cat (10 rótulos: Normal, Generic, Exploits, ...)
    alvo="cat_bin"  -> idem + 1 rótulo extra "is_attack" (coluna `label`)
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CAT = ["proto", "service", "state"]
DROP = ["id", "label", "attack_cat"]


def _ler(caminho):
    df = pd.read_csv(caminho)
    df.columns = [c.strip().lower() for c in df.columns]
    df["attack_cat"] = (df["attack_cat"].fillna("Normal").astype(str).str.strip()
                        .replace({"Backdoors": "Backdoor"}))
    return df


def _alvo(df, classes, modo):
    Y = np.stack([(df["attack_cat"] == c).values for c in classes], 1).astype("float32")
    if modo == "cat_bin":
        Y = np.concatenate([Y, df[["label"]].values.astype("float32")], 1)
    return Y


@lru_cache(maxsize=None)
def carregar(dir_dados="../data", alvo="cat", seed_split=0, frac_val=0.2):
    d = Path(dir_dados)
    tr = _ler(d / "UNSW_NB15_training-set.csv")
    te = _ler(d / "UNSW_NB15_testing-set.csv")
    classes = sorted(tr["attack_cat"].unique())
    tr, va = train_test_split(tr, test_size=frac_val, stratify=tr["attack_cat"],
                              random_state=seed_split)

    num = [c for c in tr.columns if c not in CAT + DROP]
    cauda = [c for c in num if tr[c].min() >= 0 and tr[c].max() > 1e4]  # caudas pesadas -> log1p

    def numerico(df):
        x = df[num].astype("float64").copy()
        x[cauda] = np.log1p(x[cauda].clip(lower=0))
        return x

    xtr = numerico(tr)
    mu, sd = xtr.mean(), xtr.std().replace(0, 1.0).fillna(1.0)
    cols = pd.get_dummies(tr[CAT].astype(str)).columns  # categorias não vistas -> tudo zero

    def X(df):
        n = ((numerico(df) - mu) / sd).clip(-10, 10)
        c = pd.get_dummies(df[CAT].astype(str)).reindex(columns=cols, fill_value=0)
        return np.concatenate([n.values, c.values.astype("float64")], 1).astype("float32")

    nomes = list(classes) + (["is_attack"] if alvo == "cat_bin" else [])
    return dict(Xtr=X(tr), Ytr=_alvo(tr, classes, alvo),
                Xva=X(va), Yva=_alvo(va, classes, alvo),
                Xte=X(te), Yte=_alvo(te, classes, alvo),
                classes=nomes)