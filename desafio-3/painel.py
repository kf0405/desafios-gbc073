"""Gera painel.pdf a partir de experimentos.csv e logs/.

    python painel.py                         # usa o diretório atual
    python painel.py --saida . --instavel adam_lr0.1_warmup_cosseno_s0_p3000_b256

Páginas:
  1. Quatro sinais alinhados no eixo de passos (perda, ||g||, ||theta||, eta_t) para:
     melhor configuração, baseline, melhor com Muon e uma instável.
  2. Razão de atualização e norma do gradiente por camada, para as mesmas quatro.
  3. Métrica de validação x eta por otimizador (varredura de lr, warmup+cosseno, semente 0).

Escolhas automáticas (sobrescreva com argumentos):
  melhor    = maior F1 macro médio, preferindo configurações com >= 3 sementes
  baseline  = --base_opt/--base_lr/--base_sched (padrão: adamw, 1e-3, constante)
  muon      = melhor configuração com opt == muon
  instavel  = execução que divergiu; se nenhuma, a de maior pico de ||g|| (máx / mediana)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

CHAVE = ["opt", "lr", "lr_adamw", "sched", "passos", "lote"]
COR = {"sgd": "tab:gray", "nesterov": "tab:brown", "adam": "tab:blue", "adamw": "tab:cyan",
       "nadam": "tab:purple", "muon": "tab:red"}


def rotulo(r):
    s = f"{r['opt']} η={r['lr']:g}"
    if r["opt"] == "muon" and r["lr_adamw"]:
        s += f" (adamw {r['lr_adamw']:g})"
    return f"{s}\n{r['sched']}"


def melhor_config(exp):
    ok = exp[~exp["divergiu"]]
    g = ok.groupby(CHAVE).agg(media=("val_f1_macro", "mean"), n=("seed", "count")).reset_index()
    if (g["n"] >= 3).any():
        g = g[g["n"] >= 3]
    return g.sort_values("media", ascending=False).iloc[0]


def linha_seed0(exp, cfg):
    m = np.ones(len(exp), bool)
    for k in CHAVE:
        m &= np.isclose(exp[k], cfg[k]) if exp[k].dtype.kind == "f" else (exp[k] == cfg[k])
    sub = exp[m].sort_values("seed")
    return sub.iloc[0]


def ler_log(saida, rid):
    return pd.read_csv(Path(saida) / "logs" / f"{rid}.csv")


def escolher_instavel(exp, saida, excluir):
    div = exp[exp["divergiu"] & ~exp["run_id"].isin(excluir)]
    if len(div):
        return div.iloc[0]
    melhor, ratio = None, -1
    for _, r in exp[(exp["seed"] == 0) & ~exp["run_id"].isin(excluir)].iterrows():
        try:
            g = ler_log(saida, r["run_id"])["grad"].dropna()
        except FileNotFoundError:
            continue
        if len(g) and g.median() > 0 and g.max() / g.median() > ratio:
            melhor, ratio = r, g.max() / g.median()
    return melhor


def suave(y, j=5):
    return pd.Series(y).rolling(j, min_periods=1).mean().values


def pagina_sinais(pdf, itens):
    n = len(itens)
    fig, ax = plt.subplots(4, n, figsize=(4.4 * n, 10), sharex="col", squeeze=False)
    sinais = [("perda", "perda (lote)", True), ("grad", "‖∇θℒ‖₂ (antes do clip)", True),
              ("theta", "‖θ‖₂", False), ("lr", "η_t", False)]
    for j, (titulo, log, row) in enumerate(itens):
        p = log["passo"].values
        for i, (col, nome, logy) in enumerate(sinais):
            a = ax[i, j]
            if col not in log:
                continue
            y = log[col].values
            a.plot(p, y, color="tab:blue", alpha=0.25, lw=0.8)
            a.plot(p, suave(y), color="tab:blue", lw=1.4)
            if col == "perda" and "val_perda" in log:
                v = log.dropna(subset=["val_perda"])
                a.plot(v["passo"], v["val_perda"], "o-", color="tab:orange", ms=3, lw=1, label="validação")
                a.legend(fontsize=7)
            if logy and np.all(np.nan_to_num(y, nan=1.0) > 0):
                a.set_yscale("log")
            if j == 0:
                a.set_ylabel(nome)
            a.grid(alpha=0.3)
        ax[0, j].set_title(f"{titulo}\n{rotulo(row)}\nF1 macro val = {row['val_f1_macro']:.3f}"
                           + ("  [DIVERGIU]" if row["divergiu"] else ""), fontsize=9)
        ax[3, j].set_xlabel("passo")
    fig.suptitle("Sinais do treino alinhados no eixo de passos", y=0.995)
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def pagina_camadas(pdf, itens):
    n = len(itens)
    fig, ax = plt.subplots(2, n, figsize=(4.4 * n, 6.5), sharex="col", squeeze=False)
    for j, (titulo, log, row) in enumerate(itens):
        rhos = [c for c in log.columns if c.startswith("rho/")]
        grads = [c for c in log.columns if c.startswith("grad/")]
        for k, (c, c2) in enumerate(zip(rhos, grads)):
            nome = c.split("/", 1)[1]
            for a, col in ((ax[0, j], c), (ax[1, j], c2)):
                y = log[col].values
                a.plot(log["passo"], suave(y), lw=1.3, label=nome, color=f"C{k}")
        ax[0, j].axhline(1e-3, color="k", ls=":", lw=0.8)  # referência do enunciado
        ax[0, j].set_title(f"{titulo}\n{rotulo(row)}", fontsize=9)
        for a in ax[:, j]:
            a.set_yscale("log"); a.grid(alpha=0.3)
        ax[1, j].set_xlabel("passo")
        if j == 0:
            ax[0, j].set_ylabel("razão de atualização ‖Δθ‖/‖θ‖\n(pontilhado: 1e-3)")
            ax[1, j].set_ylabel("‖∇θℒ‖₂ por camada")
            ax[0, j].legend(fontsize=7)
    fig.suptitle("Por camada: primeira, meio e última", y=0.995)
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def pagina_metrica_lr(pdf, exp):
    sub = exp[(exp["sched"] == "warmup_cosseno") & (exp["seed"] == 0)]
    fig, a = plt.subplots(figsize=(8, 5))
    for o, g in sub.groupby("opt"):
        g = g[g["passos"] == g["passos"].mode()[0]].sort_values("lr")
        ok, bad = g[~g["divergiu"]], g[g["divergiu"]]
        lab = o + (" (η das matrizes)" if o == "muon" else "")
        a.plot(ok["lr"], ok["val_f1_macro"], "o-", color=COR.get(o), label=lab)
        if len(bad):
            a.plot(bad["lr"], [a.get_ylim()[0]] * len(bad), "x", color=COR.get(o), ms=9)
    a.set_xscale("log"); a.set_xlabel("η"); a.set_ylabel("F1 macro (validação)")
    a.set_title("Métrica × η por otimizador (warmup + cosseno, semente 0; × = divergiu)")
    a.grid(alpha=0.3, which="both"); a.legend()
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=".")
    ap.add_argument("--base_opt", default="adamw")
    ap.add_argument("--base_lr", type=float, default=1e-3)
    ap.add_argument("--base_sched", default="constante")
    ap.add_argument("--instavel", default=None, help="run_id da execução instável")
    a = ap.parse_args()

    exp = pd.read_csv(Path(a.saida) / "experimentos.csv")
    exp["lr_adamw"] = exp["lr_adamw"].fillna(0.0)
    exp["divergiu"] = exp["divergiu"].astype(str) == "True"

    r_melhor = linha_seed0(exp, melhor_config(exp))
    r_muon = linha_seed0(exp, melhor_config(exp[exp["opt"] == "muon"]))
    base = exp[(exp["opt"] == a.base_opt) & np.isclose(exp["lr"], a.base_lr) & (exp["sched"] == a.base_sched)]
    r_base = base.sort_values(["seed"]).iloc[0]
    usados = {r_melhor["run_id"], r_muon["run_id"], r_base["run_id"]}
    r_inst = (exp[exp["run_id"] == a.instavel].iloc[0] if a.instavel
              else escolher_instavel(exp, a.saida, usados))

    itens = [("Melhor", r_melhor), ("Baseline", r_base), ("Melhor com Muon", r_muon)]
    if r_inst is not None:
        itens.append(("Instável", r_inst))
    itens = [(t, ler_log(a.saida, r["run_id"]), r) for t, r in itens]
    for t, _, r in itens:
        print(f"{t:16s} {r['run_id']}")

    with PdfPages(Path(a.saida) / "painel.pdf") as pdf:
        pagina_sinais(pdf, itens)
        pagina_camadas(pdf, itens)
        pagina_metrica_lr(pdf, exp)
    print("painel.pdf gerado")


if __name__ == "__main__":
    main()