"""Protocolo do Desafio 3 (etapas 1–4). Retomável: execuções já presentes em experimentos.csv são puladas.

    python varredura.py baseline    # etapa 1: receita atual, 3 sementes
    python varredura.py lr          # etapa 2: varredura de lr por otimizador (warmup_cosseno, semente 0)
    python varredura.py agendas     # etapa 3: 4 agendas p/ os 2 melhores otimizadores + Muon
    python varredura.py confirma    # etapa 4: 3 melhores combinações x 3 sementes
    python varredura.py final       # roda vencedora e baseline no TESTE (3 sementes) — use uma vez
    python varredura.py tudo        # 1 a 4 em sequência

Para o orçamento reduzido nas etapas 2–3 (permitido se igual p/ todos e declarado), use --passos_busca.
"""
import argparse
import math
import statistics as st

from treino import AGENDAS, run

# ---- ajuste aqui -------------------------------------------------------------------------------
BASELINE = dict(opt="sgd", lr=1e-3, sched="constante")        # "receita atual do grupo"
OTIMIZADORES = ["sgd", "nesterov", "adam", "adamw", "muon"]    # + "nadam" se quiser um quinto/sexto
GRADES = {
    "sgd": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    "nesterov": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    "adam": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "adamw": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "nadam": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "muon": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
}
SEEDS = [0, 1, 2, 3, 4]
AGENDAS_ETAPA3 = ["constante", "cosseno", "warmup_cosseno", "wsd"]  # ou --agendas a,b,c
# ---------------------------------------------------------------------------------------------

COMUM = {}  # preenchido no main: passos, lote, dados, alvo, saida...


def score(r):
    return -math.inf if (r["divergiu"] or math.isnan(r["val_f1_macro"])) else r["val_f1_macro"]


def executa(opt, lr, sched, seed, passos=None, **kw):
    cfg = dict(COMUM, opt=opt, lr=lr, sched=sched, seed=seed, **kw)
    if passos:
        cfg["passos"] = passos
    return run(cfg)


def baseline():
    print("\n== Etapa 1: baseline ==")
    rs = [executa(**BASELINE, seed=s) for s in SEEDS]
    return rs


def etapa_lr(passos):
    print("\n== Etapa 2: varredura de lr ==")
    melhor = {}
    ordem = sorted(OTIMIZADORES, key=lambda o: o == "muon")  # Muon por último (usa o lr_adamw ótimo)
    for o in ordem:
        extra = {}
        if o == "muon":
            extra["lr_adamw"] = melhor["adamw"]["lr"] if "adamw" in melhor else 1e-3
        grade, res = list(GRADES[o]), {}
        for ext in range(4):  # até 3 extensões se o melhor cair na borda
            for lr in grade:
                if lr not in res:
                    res[lr] = executa(o, lr, "warmup_cosseno", 0, passos, **extra)
            b = max(res, key=lambda l: score(res[l]))
            if b == min(grade):
                grade.insert(0, b / 3)
            elif b == max(grade):
                grade.append(b * 3)
            else:
                break
        melhor[o] = dict(lr=b, score=score(res[b]), **extra)
        print(f"  melhor {o}: lr={b:g}  f1_macro_val={melhor[o]['score']:.4f}")
    return melhor


def etapa_agendas(passos):
    print("\n== Etapa 3: agendas ==")
    melhor = etapa_lr(passos)
    rank = sorted(melhor, key=lambda o: -melhor[o]["score"])
    alvo = rank[:2] + (["muon"] if "muon" not in rank[:2] else [])
    combos = []
    for o in alvo:
        extra = {"lr_adamw": melhor[o]["lr_adamw"]} if o == "muon" else {}
        for a in AGENDAS_ETAPA3:
            r = executa(o, melhor[o]["lr"], a, 0, passos, **extra)
            combos.append(dict(opt=o, lr=melhor[o]["lr"], sched=a, score=score(r), **extra))
    for c in sorted(combos, key=lambda c: -c["score"]):
        print(f"  {c['opt']:9s} {c['sched']:15s} lr={c['lr']:g}  f1_macro_val={c['score']:.4f}")
    return combos


def etapa_confirma(passos):
    print("\n== Etapa 4: confirmação (3 sementes) ==")
    todos = sorted(etapa_agendas(passos), key=lambda c: -c["score"])
    chave = lambda c: (c["opt"], c["lr"], c["sched"])
    escolhidos = todos[:3]
    sem_muon = next((c for c in todos if c["opt"] != "muon"), None)
    if sem_muon and chave(sem_muon) not in [chave(c) for c in escolhidos]:
        escolhidos.append(sem_muon)  # comparador justo: a melhor combinação SEM Muon é sempre confirmada
    tab = []
    for c in escolhidos:
        kw = {"lr_adamw": c["lr_adamw"]} if "lr_adamw" in c else {}
        # confirmação com o orçamento COMPLETO (passos=None -> padrão)
        rs = [executa(c["opt"], c["lr"], c["sched"], s, None, **kw) for s in SEEDS]
        v = [r["val_f1_macro"] for r in rs if not r["divergiu"]]
        tab.append(dict(c, media=st.mean(v), dp=st.stdev(v) if len(v) > 1 else float("nan"), kw=kw))
    base = [r["val_f1_macro"] for r in baseline()]
    mb, db = st.mean(base), st.stdev(base)
    tab.sort(key=lambda t: -t["media"])
    print(f"\n  baseline {BASELINE}: {mb:.4f} ± {db:.4f}")
    for t in tab:
        ganho = t["media"] - mb
        sig = "SUPERA a variação entre sementes" if ganho > max(t["dp"], db) else "dentro do ruído"
        print(f"  {t['opt']:9s} {t['sched']:15s} lr={t['lr']:g}: {t['media']:.4f} ± {t['dp']:.4f} "
              f"(ganho {ganho:+.4f}; {sig})")
    melhor = tab[0]
    outro = next((t for t in tab if t["opt"] != melhor["opt"]), None)
    if outro:  # melhor vs. melhor de OUTRO otimizador, ambos ajustados: é a comparação que sustenta a conclusão
        d = melhor["media"] - outro["media"]
        sig = "SUPERA a variação entre sementes" if d > max(melhor["dp"], outro["dp"]) else "dentro do ruído"
        print(f"\n  melhor ({melhor['opt']}) vs melhor com outro otimizador ({outro['opt']}, {outro['sched']}, "
              f"lr={outro['lr']:g}): {d:+.4f}; {sig}")
    return tab


def final(passos):
    tab = etapa_confirma(passos)
    v = tab[0]
    alvo = [dict(BASELINE), dict(opt=v["opt"], lr=v["lr"], sched=v["sched"], **v["kw"])]
    outro = next((t for t in tab if t["opt"] != v["opt"]), None)
    if outro:
        alvo.append(dict(opt=outro["opt"], lr=outro["lr"], sched=outro["sched"], **outro["kw"]))
    print("\n== Teste (uso único): baseline, vencedora e melhor de outro otimizador ==")
    for cfg in alvo:
        for s in SEEDS:
            run(dict(COMUM, **cfg, seed=s, teste=True))
    print("resultados em experimentos_teste.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["baseline", "lr", "agendas", "confirma", "final", "tudo"])
    ap.add_argument("--passos", type=int, default=3000, help="orçamento das etapas 1 e 4")
    ap.add_argument("--passos_busca", type=int, default=None, help="orçamento reduzido das etapas 2–3")
    ap.add_argument("--lote", type=int, default=256)
    ap.add_argument("--dados", default="dados")
    ap.add_argument("--alvo", default="cat", choices=["cat", "cat_bin"])
    ap.add_argument("--perda", default="bce", choices=["bce", "ce"])
    ap.add_argument("--saida", default=".")
    ap.add_argument("--agendas", default=None, help="lista separada por vírgula p/ a etapa 3")
    a = ap.parse_args()
    if a.agendas:
        AGENDAS_ETAPA3[:] = [s.strip() for s in a.agendas.split(",")]
        assert all(s in AGENDAS for s in AGENDAS_ETAPA3), f"agendas válidas: {AGENDAS}"
    COMUM.update(passos=a.passos, lote=a.lote, dados=a.dados, alvo=a.alvo, saida=a.saida, perda=a.perda)
    pb = a.passos_busca
    {"baseline": baseline, "lr": lambda: etapa_lr(pb), "agendas": lambda: etapa_agendas(pb),
     "confirma": lambda: etapa_confirma(pb), "final": lambda: final(pb),
     "tudo": lambda: etapa_confirma(pb)}[a.etapa]()