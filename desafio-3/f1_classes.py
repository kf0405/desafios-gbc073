"""Tabela de F1 por classe (último ponto de validação) a partir de logs/.

    python f1_classes.py --saida ce_f1 RUN_ID [RUN_ID ...]
    python f1_classes.py --saida ce_f1          # todas as execuções que tenham val_f1/<classe>

Só existem colunas por classe em execuções feitas depois da atualização do treino.py.
"""
import argparse
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("run_ids", nargs="*")
ap.add_argument("--saida", default=".")
a = ap.parse_args()

pasta = Path(a.saida) / "logs"
arquivos = [pasta / f"{r}.csv" for r in a.run_ids] if a.run_ids else sorted(pasta.glob("*.csv"))
linhas = {}
for f in arquivos:
    log = pd.read_csv(f)
    cols = [c for c in log.columns if c.startswith("val_f1/")]
    if not cols:
        continue
    ult = log.dropna(subset=cols[:1]).iloc[-1]
    linhas[f.stem] = {c.split("/", 1)[1]: ult[c] for c in cols} | {"macro": ult["val_f1_macro"]}

if not linhas:
    raise SystemExit("nenhum log com F1 por classe (rode as execuções com o treino.py atualizado)")
tab = pd.DataFrame(linhas).T
pd.set_option("display.width", 250)
print(tab.round(3).to_string())