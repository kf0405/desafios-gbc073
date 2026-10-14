"""Desafio 3 — treino configurável (otimizador, lr, agenda, semente) com instrumentação.

Exemplo:
    python treino.py --opt muon --lr 0.02 --lr_adamw 1e-3 --sched warmup_cosseno --seed 0

Cada execução gera:
    logs/<run_id>.csv      sinais a cada k passos (perda, ||g||, ||theta||, razão de atualização, lr,
                           globais e por camada) + métricas de validação a cada eval_cada passos
    experimentos.csv       uma linha por execução (configuração, métrica de validação, divergiu?, tempo)
O conjunto de teste só é tocado com --teste (resultado vai para experimentos_teste.csv).
"""
import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dados import carregar

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OTIMIZADORES = ["sgd", "nesterov", "adam", "adamw", "nadam", "muon"]
AGENDAS = ["constante", "cosseno", "warmup_cosseno", "wsd", "linear", "onecycle", "degraus"]
WD_PADRAO = dict(sgd=1e-4, nesterov=1e-4, adam=0.0, nadam=0.0, adamw=1e-2, muon=1e-2)
CAMPOS = ["run_id", "opt", "lr", "lr_adamw", "sched", "seed", "passos", "lote", "clip", "wd",
          "val_f1_macro", "val_f1_micro", "val_perda", "divergiu", "tempo_s"]


# --------------------------------------------------------------------------- Muon
def ortogonaliza(G, passos=5):
    """Newton–Schulz: aproxima U V^T de G = U S V^T."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    T = X.size(0) > X.size(1)
    if T:
        X = X.T
    for _ in range(passos):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if T else X


class Muon(torch.optim.Optimizer):
    """Momento -> (Nesterov) -> ortogonalização -> passo. Só para matrizes 2-D das camadas ocultas."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.01, ns_passos=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay, ns_passos=ns_passos))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                buf = st["buf"]
                buf.mul_(g["momentum"]).add_(p.grad)
                m = p.grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
                u = ortogonaliza(m, g["ns_passos"]) * max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.mul_(1 - g["lr"] * g["weight_decay"])  # weight decay desacoplado
                p.add_(u, alpha=-g["lr"])


# --------------------------------------------------------------------------- modelo
class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=256, camadas=3, drop=0.1):
        super().__init__()
        dims = [d_in] + [hidden] * camadas
        self.fc = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(camadas)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(camadas)])
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(hidden, d_out)

    def forward(self, x):
        for fc, nm in zip(self.fc, self.norm):
            x = self.drop(F.gelu(nm(fc(x))))
        return self.head(x)


def camadas_monitoradas(model):
    """Primeira, uma do meio e a última matriz de pesos."""
    w = [n for n, p in model.named_parameters() if p.ndim == 2]
    return [w[0], w[len(w) // 2], w[-1]]


# --------------------------------------------------------------------------- otimizador e agenda
def construir_otimizadores(cfg, model):
    o, lr = cfg["opt"], cfg["lr"]
    wd = cfg["wd"] if cfg.get("wd") is not None else WD_PADRAO[o]
    if o == "muon":
        mp = [p for n, p in model.named_parameters() if p.ndim == 2 and "head" not in n]
        ids = {id(p) for p in mp}
        ap = [p for p in model.parameters() if id(p) not in ids]  # vieses, LayerNorm, head -> AdamW
        lr_a = cfg.get("lr_adamw") or 1e-3
        return [Muon(mp, lr=lr, weight_decay=wd), torch.optim.AdamW(ap, lr=lr_a, weight_decay=wd)], wd
    P = model.parameters()
    if o == "sgd":
        return [torch.optim.SGD(P, lr=lr, momentum=0.9, weight_decay=wd)], wd
    if o == "nesterov":
        return [torch.optim.SGD(P, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)], wd
    if o == "adam":
        return [torch.optim.Adam(P, lr=lr, weight_decay=wd)], wd
    if o == "adamw":
        return [torch.optim.AdamW(P, lr=lr, weight_decay=wd)], wd
    if o == "nadam":
        return [torch.optim.NAdam(P, lr=lr, weight_decay=wd)], wd
    raise ValueError(o)


def fator_agenda(nome, T):
    """Multiplicador de lr no passo t (aplicado igualmente a todos os otimizadores da execução)."""
    w, d = max(1, int(0.05 * T)), max(1, int(0.2 * T))

    def f(t):
        if nome == "constante":
            return 1.0
        if nome == "cosseno":
            return 0.5 * (1 + math.cos(math.pi * min(t, T) / T))
        if nome == "warmup_cosseno":
            if t < w:
                return (t + 1) / w
            return 0.5 * (1 + math.cos(math.pi * min(t - w, T - w) / (T - w)))
        if nome == "wsd":  # warmup – estável – decaimento linear
            if t < w:
                return (t + 1) / w
            return 1.0 if t < T - d else max(0.0, (T - t) / d)
        if nome == "linear":  # warmup + decaimento linear até 0
            if t < w:
                return (t + 1) / w
            return max(0.0, (T - t) / (T - w))
        if nome == "onecycle":  # sobe 30% do tempo (de 4% a 100%) e desce em cosseno até 0
            u = int(0.3 * T)
            if t < u:
                return 0.04 + 0.96 * t / u
            return 0.5 * (1 + math.cos(math.pi * min(t - u, T - u) / (T - u)))
        if nome == "degraus":  # warmup + x0.1 em 50% e 75% dos passos
            if t < w:
                return (t + 1) / w
            return 1.0 * (0.1 if t >= 0.5 * T else 1.0) * (0.1 if t >= 0.75 * T else 1.0)
        raise ValueError(nome)
    return f


# --------------------------------------------------------------------------- métricas
@torch.no_grad()
def avaliar(model, X, Y, perda="bce", lote=8192):
    """perda="bce": sigmoide por rótulo, limiar 0.5. perda="ce": softmax, previsão = argmax."""
    model.eval()
    soma, P = 0.0, []
    for i in range(0, len(X), lote):
        z, y = model(X[i:i + lote]), Y[i:i + lote]
        if perda == "ce":
            soma += F.cross_entropy(z, y.argmax(1), reduction="sum").item()
            P.append(F.one_hot(z.argmax(1), z.size(1)).bool())
        else:
            soma += F.binary_cross_entropy_with_logits(z, y, reduction="sum").item()
            P.append(z > 0)  # sigmoid(z) > 0.5
    P, Yb = torch.cat(P), Y > 0.5
    tp = (Yb & P).sum(0).float(); fp = (~Yb & P).sum(0).float(); fn = (Yb & ~P).sum(0).float()
    f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    micro = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()).clamp(min=1)
    return dict(perda=soma / (len(Y) if perda == "ce" else Y.numel()), f1_macro=f1.mean().item(),
                f1_micro=micro.item(), f1_classe=f1.tolist())


def calcula_perda(z, y, perda):
    return F.cross_entropy(z, y.argmax(1)) if perda == "ce" else F.binary_cross_entropy_with_logits(z, y)


# --------------------------------------------------------------------------- utilitários de CSV
def run_id_de(cfg):
    s = f"{cfg['opt']}_lr{cfg['lr']:g}"
    if cfg["opt"] == "muon":
        s += f"_la{(cfg.get('lr_adamw') or 1e-3):g}"
    s += f"_{cfg['sched']}_s{cfg['seed']}_p{cfg['passos']}_b{cfg['lote']}"
    return s + ("_ce" if cfg.get("perda") == "ce" else "")


def _parse(r):
    for k in ("lr", "lr_adamw", "val_f1_macro", "val_f1_micro", "val_perda", "tempo_s", "clip", "wd"):
        r[k] = float(r[k]) if r.get(k) not in (None, "") else float("nan")
    r["divergiu"] = str(r["divergiu"]) == "True"
    r["seed"], r["passos"], r["lote"] = int(r["seed"]), int(r["passos"]), int(r["lote"])
    return r


def ler_experimentos(saida="."):
    p = Path(saida) / "experimentos.csv"
    return [_parse(r) for r in csv.DictReader(open(p))] if p.exists() else []


def _anexar(caminho, linha, campos):
    novo = not Path(caminho).exists()
    with open(caminho, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        if novo:
            w.writeheader()
        w.writerow(linha)


# --------------------------------------------------------------------------- execução
def run(cfg):
    """Executa (ou recupera de experimentos.csv) uma configuração. Devolve a linha do experimento."""
    c = dict(passos=3000, lote=256, clip=1.0, k=25, eval_cada=250, dados="dados", alvo="cat",
             hidden=256, camadas=3, perda="bce", saida=".", teste=False, lr_adamw=None, wd=None, refazer=False)
    c.update(cfg)
    assert c["perda"] != "ce" or c["alvo"] == "cat", "perda=ce exige --alvo cat (uma classe por amostra)"
    assert c["eval_cada"] % c["k"] == 0 and c["passos"] % c["k"] == 0, "k deve dividir eval_cada e passos"
    rid = run_id_de(c)
    saida = Path(c["saida"])
    if not c["teste"] and not c["refazer"]:
        for r in ler_experimentos(saida):
            if r["run_id"] == rid:
                return r
    (saida / "logs").mkdir(parents=True, exist_ok=True)

    random.seed(c["seed"]); np.random.seed(c["seed"]); torch.manual_seed(c["seed"])
    D = carregar(c["dados"], c["alvo"])
    t2 = lambda a: torch.as_tensor(a, device=DEV)
    Xtr, Ytr, Xva, Yva = t2(D["Xtr"]), t2(D["Ytr"]), t2(D["Xva"]), t2(D["Yva"])
    model = MLP(Xtr.size(1), Ytr.size(1), c["hidden"], c["camadas"]).to(DEV)
    CAM = camadas_monitoradas(model)
    params = dict(model.named_parameters())
    opts, wd = construir_otimizadores(c, model)
    f = fator_agenda(c["sched"], c["passos"])
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, f) for o in opts]
    clip = c["clip"] if c["clip"] and c["clip"] > 0 else float("inf")

    g_ = torch.Generator(device=DEV); g_.manual_seed(c["seed"])

    def lotes():
        while True:
            perm = torch.randperm(len(Xtr), generator=g_, device=DEV)
            for i in range(0, len(perm) - c["lote"] + 1, c["lote"]):
                yield perm[i:i + c["lote"]]
    it = lotes()

    log, divergiu, ultimo_val = [], False, None
    t0 = time.time()
    for t in range(1, c["passos"] + 1):
        idx = next(it)
        model.train()
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss = calcula_perda(model(Xtr[idx]), Ytr[idx], c["perda"])
        loss.backward()
        logar = (t % c["k"] == 0) or t == 1
        if logar:  # norma por camada ANTES do clipping
            gcam = {n: params[n].grad.norm().item() for n in CAM}
            antes = {n: params[n].detach().clone() for n in CAM}
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), clip)  # devolve a norma total pré-corte
        for o in opts:
            o.step()
        for s in scheds:
            s.step()

        if logar:
            L = loss.item()
            if not math.isfinite(L) or not math.isfinite(gnorm.item()):
                divergiu = True
                log.append(dict(passo=t, perda=L, grad=gnorm.item()))
                break
            with torch.no_grad():
                theta = torch.linalg.vector_norm(torch.stack([p.norm() for p in params.values()])).item()
                row = dict(passo=t, perda=L, grad=gnorm.item(), theta=theta,
                           lr=opts[0].param_groups[0]["lr"])
                for n in CAM:
                    row[f"grad/{n}"] = gcam[n]
                    row[f"theta/{n}"] = params[n].norm().item()
                    row[f"rho/{n}"] = ((params[n] - antes[n]).norm() / antes[n].norm()).item()
            if t % c["eval_cada"] == 0 or t == c["passos"]:
                v = avaliar(model, Xva, Yva, c["perda"])
                ultimo_val = v
                row.update(val_perda=v["perda"], val_f1_macro=v["f1_macro"], val_f1_micro=v["f1_micro"])
                for nome_c, f1_c in zip(D["classes"], v["f1_classe"]):
                    row[f"val_f1/{nome_c}"] = f1_c  # F1 por classe, só nos logs/ (experimentos.csv não muda)
            log.append(row)

    tempo = time.time() - t0
    nan = float("nan")
    v = ultimo_val if (ultimo_val and not divergiu) else dict(perda=nan, f1_macro=nan, f1_micro=nan)
    linha = dict(run_id=rid, opt=c["opt"], lr=c["lr"], lr_adamw=c["lr_adamw"] if c["opt"] == "muon" else "",
                 sched=c["sched"], seed=c["seed"], passos=c["passos"], lote=c["lote"], clip=c["clip"],
                 wd=wd, val_f1_macro=v["f1_macro"], val_f1_micro=v["f1_micro"], val_perda=v["perda"],
                 divergiu=divergiu, tempo_s=round(tempo, 1))

    nome_log = rid + ("_teste" if c["teste"] else "")
    campos_log = sorted({k for r in log for k in r}, key=lambda k: (k != "passo", k))
    with open(saida / "logs" / f"{nome_log}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos_log, restval="")
        w.writeheader(); w.writerows(log)

    if c["teste"]:  # uso ÚNICO do conjunto de teste, só para as configurações finais
        if not divergiu:
            te = avaliar(model, t2(D["Xte"]), t2(D["Yte"]), c["perda"])
            linha.update(teste_f1_macro=te["f1_macro"], teste_f1_micro=te["f1_micro"], teste_perda=te["perda"])
            for nome, f1 in zip(D["classes"], te["f1_classe"]):
                linha[f"teste_f1/{nome}"] = f1
        _anexar(saida / "experimentos_teste.csv", linha, list(linha.keys()))
    else:
        _anexar(saida / "experimentos.csv", linha, CAMPOS)
    print(f"{rid}: val_f1_macro={linha['val_f1_macro']:.4f} divergiu={divergiu} ({tempo:.0f}s)")
    return _parse({k: str(v) for k, v in linha.items() if k in CAMPOS})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", required=True, choices=OTIMIZADORES)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--lr_adamw", type=float, default=None, help="lr da parte AdamW do Muon")
    ap.add_argument("--sched", default="warmup_cosseno", choices=AGENDAS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--passos", type=int, default=3000)
    ap.add_argument("--lote", type=int, default=256)
    ap.add_argument("--clip", type=float, default=1.0, help="<=0 desliga o clipping")
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--eval_cada", type=int, default=250)
    ap.add_argument("--dados", default="dados")
    ap.add_argument("--alvo", default="cat", choices=["cat", "cat_bin"])
    ap.add_argument("--perda", default="bce", choices=["bce", "ce"],
                    help="bce: sigmoide por rótulo (multilabel); ce: softmax + entropia cruzada (multiclasse)")
    ap.add_argument("--saida", default=".")
    ap.add_argument("--teste", action="store_true", help="avalia no teste (uma vez, só configs finais)")
    ap.add_argument("--refazer", action="store_true")
    run(vars(ap.parse_args()))