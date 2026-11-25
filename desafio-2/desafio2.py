import argparse
import importlib.util
import json
import math
import sys
import time
from types import SimpleNamespace

import torch


A_VALUES = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
MODES = ("gaussian", "orthogonal")

STAGE1_LOGIT = 0.65
STAGE1_GAMMA = 1.0

LOGIT_VALUES = [0.40, 0.50, 0.60, 0.65, 0.70, 0.80, 0.90]
GAMMA_VALUES = [0.90, 0.95, 1.00, 1.05, 1.10]


def carregar_harness(caminho):
    spec = importlib.util.spec_from_file_location("_d2_harness_v3", caminho)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"não foi possível carregar {caminho}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def construir_candidato(tipo, a, modo, hidden_gamma, logit_scale):
    if tipo == "leaky_external":
        e_f2 = (1.0 + a * a) / 2.0
        activation_gain = 1.0
        weight_gain = 1.0 / math.sqrt(e_f2)

        def ativacao(x):
            return torch.nn.functional.leaky_relu(x, negative_slope=a)

    elif tipo == "leaky_embedded":
        e_f2 = (1.0 + a * a) / 2.0
        activation_gain = 1.0 / math.sqrt(e_f2)
        weight_gain = 1.0

        def ativacao(x):
            return activation_gain * torch.nn.functional.leaky_relu(
                x, negative_slope=a
            )

    elif tipo == "relu_embedded":
        a = 0.0
        e_f2 = 0.5
        activation_gain = math.sqrt(2.0)
        weight_gain = 1.0

        def ativacao(x):
            return math.sqrt(2.0) * torch.relu(x)

    else:
        raise ValueError(tipo)

    @torch.no_grad()
    def inicializar(W, b, fan_in, fan_out, camada, n_camadas):
        if camada == n_camadas:
            ganho = weight_gain * logit_scale
        else:
            ganho = weight_gain * hidden_gamma

        if modo == "gaussian":
            desvio = ganho / math.sqrt(fan_in)
            W.normal_(0.0, desvio)
        elif modo == "orthogonal":
            torch.nn.init.orthogonal_(W, gain=ganho)
        else:
            raise ValueError(modo)

        b.zero_()

    nome = (
        f"{tipo}_a{a:g}_{modo}_"
        f"h{hidden_gamma:g}_logit{logit_scale:g}"
    )

    return SimpleNamespace(
        ativacao=ativacao,
        inicializar=inicializar,
        nome=nome,
        tipo=tipo,
        a=a,
        modo=modo,
        hidden_gamma=hidden_gamma,
        logit_scale=logit_scale,
        e_f2=e_f2,
        activation_gain=activation_gain,
        weight_gain=weight_gain,
    )


def executar(harness, candidato, tarefas, calib, fracao, epocas, sementes):
    t0 = time.perf_counter()

    score, linhas = harness.avaliar(
        candidato,
        tarefas,
        calib,
        sementes=sementes,
        fracao=fracao,
        epocas=epocas,
        verboso=False,
    )

    s_values = [float(x["s_t"]) for x in linhas]

    acc = {}
    for x in linhas:
        acc[x["tarefa"]] = x["acc"]

    por_dataset = {}
    for nome, valor in acc.items():
        dataset = nome.split("_L")[0]
        por_dataset.setdefault(dataset, []).append(valor)

    por_L = {}
    for nome, valor in acc.items():
        L = nome.split("_L")[1]
        por_L.setdefault(L, []).append(valor)

    return {
        "score": float(score),
        "min_s": min(s_values) if s_values else 0.0,
        "mean_s": sum(s_values) / len(s_values) if s_values else 0.0,
        "tempo_s": time.perf_counter() - t0,
        "tipo": candidato.tipo,
        "a": float(candidato.a),
        "modo": candidato.modo,
        "hidden_gamma": float(candidato.hidden_gamma),
        "logit_scale": float(candidato.logit_scale),
        "e_f2": float(candidato.e_f2),
        "activation_gain": float(candidato.activation_gain),
        "weight_gain": float(candidato.weight_gain),
        "media_dataset": {
            k: sum(v) / len(v) for k, v in por_dataset.items()
        },
        "media_L": {
            str(k): sum(v) / len(v) for k, v in por_L.items()
        },
        "acc_tarefas": acc,
        "linhas": linhas,
    }


def imprimir(resultados, titulo, limite=20):
    resultados = sorted(resultados, key=lambda x: x["score"], reverse=True)

    print("\n" + "=" * 138)
    print(titulo)
    print("=" * 138)
    print(
        f"{'#':>3} {'S':>8} {'minS':>7} {'tipo':>15} {'a':>5} "
        f"{'modo':>10} {'gamma':>6} {'logit':>6} "
        f"{'MNIST':>8} {'Fashion':>8} {'CIFAR':>8} "
        f"{'L4':>7} {'L16':>7} {'L48':>7}"
    )
    print("-" * 138)

    for i, r in enumerate(resultados[:limite], 1):
        md = r["media_dataset"]
        ml = r["media_L"]
        print(
            f"{i:>3} {r['score']:>8.3f} {r['min_s']:>7.3f} "
            f"{r['tipo']:>15} {r['a']:>5.2f} {r['modo']:>10} "
            f"{r['hidden_gamma']:>6.2f} {r['logit_scale']:>6.2f} "
            f"{md.get('mnist', 0):>8.4f} "
            f"{md.get('fashion', 0):>8.4f} "
            f"{md.get('cifar10', 0):>8.4f} "
            f"{ml.get('4', 0):>7.4f} "
            f"{ml.get('16', 0):>7.4f} "
            f"{ml.get('48', 0):>7.4f}"
        )

    return resultados


def stage1(harness, tarefas):
    configs = []

    for modo in MODES:
        for a in A_VALUES:
            configs.append(("leaky_external", a, modo))
            configs.append(("leaky_embedded", a, modo))
        configs.append(("relu_embedded", 0.0, modo))

    print(f"STAGE 1: {len(configs)} candidatos")
    print("Quick: 10% dos dados, 1 época, 1 semente")
    print("Objetivo: escolher famílias, não o vencedor final")

    calib = harness.calibrar(
        tarefas,
        harness.SEMENTES[:1],
        0.1,
        1,
    )

    resultados = []

    for i, (tipo, a, modo) in enumerate(configs, 1):
        print(
            f"[S1 {i:03d}/{len(configs)}] "
            f"{tipo:<16} a={a:.2f} {modo}"
        )

        candidato = construir_candidato(
            tipo, a, modo, STAGE1_GAMMA, STAGE1_LOGIT
        )

        try:
            resultados.append(
                executar(
                    harness,
                    candidato,
                    tarefas,
                    calib,
                    0.1,
                    1,
                    harness.SEMENTES[:1],
                )
            )
        except Exception as e:
            print(f"  ERRO: {type(e).__name__}: {e}")

    resultados = imprimir(
        resultados,
        "TOP STAGE 1 — ATIVAÇÃO / PARAMETRIZAÇÃO / ORTOGONAL",
        20,
    )

    with open("desafio2_v3_stage1.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    return resultados


def stage2(harness, tarefas, top_stage1):
    top_stage1 = sorted(
        top_stage1,
        key=lambda x: x["score"],
        reverse=True,
    )[:8]

    configs = []

    for r in top_stage1:
        for gamma in GAMMA_VALUES:
            for logit in LOGIT_VALUES:
                configs.append(
                    (
                        r["tipo"],
                        r["a"],
                        r["modo"],
                        gamma,
                        logit,
                    )
                )

    print(f"STAGE 2: {len(configs)} candidatos")
    print("Médio: 25% dos dados, 1 época, 1 semente")
    print("Objetivo: ajustar ganho oculto e escala dos logits")

    calib = harness.calibrar(
        tarefas,
        harness.SEMENTES[:1],
        0.25,
        1,
    )

    resultados = []

    for i, cfg in enumerate(configs, 1):
        tipo, a, modo, gamma, logit = cfg

        print(
            f"[S2 {i:03d}/{len(configs)}] "
            f"{tipo:<16} a={a:.2f} {modo:<10} "
            f"gamma={gamma:.2f} logit={logit:.2f}"
        )

        candidato = construir_candidato(
            tipo, a, modo, gamma, logit
        )

        try:
            resultados.append(
                executar(
                    harness,
                    candidato,
                    tarefas,
                    calib,
                    0.25,
                    1,
                    harness.SEMENTES[:1],
                )
            )
        except Exception as e:
            print(f"  ERRO: {type(e).__name__}: {e}")

    resultados = imprimir(
        resultados,
        "TOP STAGE 2 — AJUSTE EM 25% DOS DADOS",
        20,
    )

    with open("desafio2_v3_stage2.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    return resultados


def stage3(harness, tarefas, top_stage2, top_final=6):
    top_stage2 = sorted(
        top_stage2,
        key=lambda x: x["score"],
        reverse=True,
    )[:top_final]

    calib = harness.calibrar(
        tarefas,
        harness.SEMENTES,
        1.0,
        harness.EPOCAS,
    )

    resultados = []

    print(f"STAGE 3: {len(top_stage2)} candidatos")
    print("Completo: 100% dos dados, 3 épocas, 3 sementes")

    for i, r in enumerate(top_stage2, 1):
        print(
            f"[S3 {i:02d}/{len(top_stage2)}] "
            f"{r['tipo']:<16} a={r['a']:.2f} "
            f"{r['modo']:<10} "
            f"gamma={r['hidden_gamma']:.2f} "
            f"logit={r['logit_scale']:.2f}"
        )

        candidato = construir_candidato(
            r["tipo"],
            r["a"],
            r["modo"],
            r["hidden_gamma"],
            r["logit_scale"],
        )

        try:
            resultado = executar(
                harness,
                candidato,
                tarefas,
                calib,
                1.0,
                harness.EPOCAS,
                harness.SEMENTES,
            )
            resultados.append(resultado)

            print(
                f"  S={resultado['score']:.4f} "
                f"minS={resultado['min_s']:.4f}"
            )

            print(
                "  "
                + " | ".join(
                    f"{k}={v:.4f}"
                    for k, v in sorted(
                        resultado["acc_tarefas"].items()
                    )
                )
            )

        except Exception as e:
            print(f"  ERRO: {type(e).__name__}: {e}")

    resultados = imprimir(
        resultados,
        "RESULTADO FINAL V3 — 3 SEMENTES / DADOS COMPLETOS",
        len(resultados),
    )

    with open("desafio2_v3_final.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    return resultados


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--harness",
        default="desafio2harness.py",
    )
    ap.add_argument(
        "--stage",
        type=int,
        choices=(1, 2, 3),
        default=1,
    )
    ap.add_argument(
        "--top",
        type=int,
        default=8,
    )
    ap.add_argument(
        "--full-top",
        type=int,
        default=6,
    )
    args = ap.parse_args()

    harness = carregar_harness(args.harness)
    tarefas = harness.tarefas_publicas()

    if args.stage == 1:
        stage1(harness, tarefas)

    elif args.stage == 2:
        try:
            with open("desafio2_v3_stage1.json", encoding="utf-8") as f:
                s1 = json.load(f)
        except FileNotFoundError:
            raise SystemExit(
                "Rode primeiro: python desafio2_v3.py --stage 1"
            )

        stage2(
            harness,
            tarefas,
            sorted(s1, key=lambda x: x["score"], reverse=True)[:args.top],
        )

    else:
        try:
            with open("desafio2_v3_stage2.json", encoding="utf-8") as f:
                s2 = json.load(f)
        except FileNotFoundError:
            raise SystemExit(
                "Rode primeiro: python desafio2_v3.py --stage 2"
            )

        stage3(
            harness,
            tarefas,
            s2,
            top_final=args.full_top,
        )


if __name__ == "__main__":
    main()
