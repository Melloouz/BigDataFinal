"""Job Bronze — cópia fiel da landing, em Parquet (requisitos R1, R2, R4).

O que este job faz, em uma frase: lê os eventos crus de `landing/` com um
**schema explícito** (sem deixar o Spark adivinhar os tipos), acrescenta as
colunas de **proveniência** (de qual arquivo cada linha veio e quando foi lida)
e grava em `bronze/`, **sem apagar nada** do que já estava lá (append-only).

Como rodar:

    docker compose run --rm app python pipeline/bronze.py

Requisitos atendidos:
- R1: todo acesso ao S3 é feito pelo Spark com caminhos s3a:// (nada de boto3,
  nada de baixar arquivo para o disco). Quem cuida disso é o comum.py.
- R2: é um job executável sozinho e verifica que a camada de que depende
  (a landing) existe; se não existir, para avisando o que rodar antes.
- R4: Bronze append-only, com schema explícito, particionada por
  event_name/year/month/day (o dia de coleta, tirado do caminho da landing),
  com colunas de proveniência: o arquivo de origem e o instante da ingestão.
"""

from __future__ import annotations

from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    LongType,
    MapType,
    StringType,
    StructField,
    StructType,
)

from comum import (
    criar_sessao,
    exigir_camada,
    glob_eventos,
    gravar_eventos,
    ler_eventos,
    titulo,
)

# ---------------------------------------------------------------------------
# R4: schema EXPLÍCITO da landing.
#
# "Explícito" quer dizer que nós escrevemos aqui o nome e o tipo de cada campo,
# em vez de deixar o Spark inferir lendo o JSON. Isso garante que:
#   - os tipos não mudam de uma execução para outra;
#   - um campo faltando numa linha vira null (e não quebra o job);
#   - um evento defeituoso (ex.: event_ts nulo) é lido sem erro — é a Silver
#     que vai julgar o que é inválido (R7), não a Bronze.
#
# Decisões de modelagem (fiéis ao docs/CONTRATO_EVENTO.md):
#   - A Bronze é uma CÓPIA FIEL: preservamos a estrutura aninhada
#     (page, traffic, device, items) exatamente como chega.
#   - `params` tem chaves diferentes a cada evento e valores de tipos mistos
#     (número, texto, nulo). Para não perder nada nem brigar com os tipos aqui,
#     lemos `params` como um MAP<string,string> — um dicionário de chave/valor
#     em texto. A Silver (R6) é que transforma cada chave na coluna tipada certa.
#   - `event_ts` fica como string (texto). O contrato permite que ele venha nulo
#     ou no futuro; converter para timestamp e validar é trabalho da Silver.
# ---------------------------------------------------------------------------
SCHEMA_LANDING = StructType([
    StructField("event_id", StringType()),
    StructField("event_name", StringType()),
    StructField("event_ts", StringType()),            # ISO-8601; pode vir nulo (R7 trata)
    StructField("user_pseudo_id", StringType()),
    StructField("user_id", StringType()),             # nulo antes de sign_up/login
    StructField("session_hint", StringType()),        # não confiável; a Silver ignora (R8)
    StructField("page", StructType([
        StructField("location", StringType()),
        StructField("referrer", StringType()),
    ])),
    StructField("traffic", StructType([
        StructField("source", StringType()),
        StructField("medium", StringType()),
        StructField("campaign", StringType()),
    ])),
    StructField("device", StructType([
        StructField("category", StringType()),
        StructField("os", StringType()),
        StructField("browser", StringType()),
    ])),
    # chaves variáveis por evento -> dicionário de texto; a Silver tipa cada chave (R6)
    StructField("params", MapType(StringType(), StringType())),
    StructField("items", ArrayType(StructType([
        StructField("item_id", StringType()),
        StructField("item_name", StringType()),
        StructField("item_category", StringType()),
        StructField("price", DoubleType()),
        StructField("quantity", LongType()),
    ]))),
])

def main() -> None:
    spark = criar_sessao("bronze")

    # R2: a Bronze depende da landing. Se a landing não existe, pare com uma
    # mensagem dizendo o que rodar antes (função pronta do comum.py).
    exigir_camada(
        spark,
        glob_eventos("landing"),
        "docker compose run --rm app python gerador/gerar_eventos.py",
    )

    # Lê SÓ as pastas event_name=… da landing, como JSON, com o schema acima.
    # ler_eventos devolve, além dos campos do evento, as colunas de partição
    # event_name, year, month e day — tiradas do CAMINHO do arquivo. É esse
    # year/month/day que o R4 pede: o DIA DE COLETA (quando o lote caiu na
    # landing), não a data dentro do event_ts.
    landing = ler_eventos(spark, "landing", "json", schema=SCHEMA_LANDING)

    # R4: colunas de PROVENIÊNCIA — a "etiqueta de origem" de cada linha.
    #   - arquivo_origem: o caminho completo do arquivo .jsonl de onde a linha
    #     veio. input_file_name() é uma função do Spark que devolve isso.
    #   - ingerido_em: o instante em que esta execução leu o dado (UTC).
    #     current_timestamp() é avaliado no momento da gravação.
    bronze = (
        landing
        .withColumn("arquivo_origem", F.input_file_name())
        .withColumn("ingerido_em", F.current_timestamp())
    )

    # R4: Bronze APPEND-ONLY. gravar_eventos(..., "append") grava particionado
    # por event_name/year/month/day direto na raiz de bronze/, SEM apagar nada
    # do que já estava lá. Rodar o job de novo acrescenta; nunca sobrescreve.
    gravar_eventos(bronze, "bronze", "append")

    # Um resumo rápido para conferir a olho (e para o print da execução).
    titulo("Bronze — gravada em s3a://datalake-tracking/bronze/")
    total = bronze.count()
    print(f"Total de linhas gravadas nesta execução: {total:,}")
    print()
    print("Linhas por evento (event_name):")
    (bronze.groupBy("event_name")
           .count()
           .orderBy(F.desc("count"))
           .show(50, truncate=False))


    spark.stop()

if __name__ == "__main__":
    main()
