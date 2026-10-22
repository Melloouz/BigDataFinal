"""Script de sequência (requisito R3): roda o pipeline inteiro, em ordem.

Reconstrói o datalake a partir da landing QUE JÁ EXISTE: Bronze -> Silver ->
as cinco Golds (a Gold de mapa de calor já exporta o site/mapa_calor.js no fim,
R11). Este script NÃO gera dados — ele não roda o gerador; só transforma o que
está na landing.

Como rodar:

    docker compose run --rm app python pipeline/pipeline_completo.py

Cada etapa é um job independente (R2) e roda no SEU PRÓPRIO processo, exatamente
como "python pipeline/<job>.py". Rodar cada job isolado (em vez de no mesmo
processo) garante um Spark limpo por etapa — mais robusto do que parar e recriar
a SparkSession várias vezes no mesmo processo. Se uma etapa falhar (código de
saída != 0), o script para ali e avisa qual foi; as seguintes dependem dela.

Dica: para rodar em paralelo o que não depende um do outro (as Golds só
dependem da Silver), use a tela Glue · Workflows do explorador ou:

    docker compose run --rm app python -m explorador.workflows pipeline/workflows/<nome>.json
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

# Ordem de execução. Cada item é (rótulo, arquivo do job). A Silver depende da
# Bronze; as Golds dependem da Silver. gold_mapa_calor exporta o site no fim (R11).
ETAPAS = [
    ("Bronze", "bronze.py"),
    ("Silver", "silver.py"),
    ("Gold · Funil", "gold_funil.py"),
    ("Gold · Engajamento", "gold_engajamento.py"),
    ("Gold · Receita e aquisição", "gold_receita.py"),
    ("Gold · ML", "gold_ml.py"),
    ("Gold · Mapa de calor (+ exportação R11)", "gold_mapa_calor.py"),
]

# Pasta onde este script está (pipeline/), para achar os jobs ao lado dele.
PASTA = Path(__file__).resolve().parent

def main() -> None:
    inicio_total = time.time()
    print("=" * 70)
    print("PIPELINE COMPLETO (R3) — reconstrução do datalake a partir da landing")
    print("=" * 70)

    for i, (rotulo, arquivo) in enumerate(ETAPAS, start=1):
        caminho_job = PASTA / arquivo
        print()
        print(f">>> [{i}/{len(ETAPAS)}] {rotulo}  (pipeline/{arquivo})")
        inicio = time.time()

        # roda o job no próprio processo (mesmo python que está rodando este script)
        resultado = subprocess.run([sys.executable, str(caminho_job)])

        if resultado.returncode != 0:
            print()
            print(f"[FALHOU] {rotulo} (código de saída {resultado.returncode}).")
            raise SystemExit(
                f"Pipeline interrompido em '{rotulo}'. "
                f"Veja o erro acima, corrija e rode de novo."
            )
        print(f"<<< {rotulo} concluído em {time.time() - inicio:.1f}s")

    total = time.time() - inicio_total
    print()
    print("=" * 70)
    print(f"PIPELINE COMPLETO — todas as {len(ETAPAS)} etapas com sucesso em {total:.1f}s")
    print("=" * 70)

if __name__ == "__main__":
    main()
