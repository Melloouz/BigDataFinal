"""Job Gold — Engajamento (requisitos R1, R2; seção 5.3).

Responde à pergunta nº 2 do negócio: o vídeo é visto até o fim? O que o
visitante procura no FAQ e não encontra? Grava quatro tabelas na Gold:

  - gold/sessoes          : uma linha por sessão (duração, eventos, scroll
                            máximo, maior video_percent, fez lead, comprou,
                            canal, dispositivo).
  - gold/retencao_video   : das sessões com video_start, o percentual que
                            atingiu 10, 25, 50, 75 e 100%.
  - gold/buscas_faq       : por termo, total de buscas e buscas sem resultado.
  - gold/saidas_downloads : cliques de saída por domínio e downloads por
                            arquivo (coluna `tipo` distingue 'saida'/'download').

Como rodar:

    docker compose run --rm app python pipeline/gold_engajamento.py

Definições (seção 5.4):
  - Canal de uma sessão: "<source> / <medium>" do PRIMEIRO evento da sessão.
  - Dispositivo da sessão: device_category do primeiro evento.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from comum import (
    caminho,
    criar_sessao,
    exigir_camada,
    glob_eventos,
    ler_eventos,
    titulo,
)

MARCOS_VIDEO = [10, 25, 50, 75, 100]

def tabela_sessoes(eventos: DataFrame) -> DataFrame:
    """gold/sessoes: uma linha por sessão com o resumo da visita.

    Canal e dispositivo vêm do PRIMEIRO evento da sessão (def. 5.4). As demais
    colunas são agregações sobre todos os eventos da sessão.
    """
    w = Window.partitionBy("session_id").orderBy("event_ts")

    # atributos do primeiro evento (canal, dispositivo)
    primeiro = (
        eventos
        .withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .select(
            "session_id",
            F.concat_ws(" / ", F.col("traffic_source"), F.col("traffic_medium")).alias("canal"),
            F.col("device_category").alias("dispositivo"),
            F.col("user_pseudo_id"),
            F.col("user_id_resolvido"),
        )
    )

    # agregações da sessão
    agg = (
        eventos.groupBy("session_id").agg(
            F.min("event_ts").alias("inicio"),
            F.max("event_ts").alias("fim"),
            F.count("*").alias("eventos"),
            F.max("percent_scrolled").alias("scroll_maximo"),
            F.max("video_percent").alias("video_percent_maximo"),
            F.max(F.when(F.col("event_name") == "generate_lead", 1).otherwise(0)).alias("_fez_lead"),
            F.max(F.when(F.col("event_name") == "purchase", 1).otherwise(0)).alias("_comprou"),
        )
        # duração em segundos entre o primeiro e o último evento
        .withColumn("duracao_seg",
                    F.col("fim").cast("long") - F.col("inicio").cast("long"))
        .withColumn("fez_lead", F.col("_fez_lead") == 1)
        .withColumn("comprou", F.col("_comprou") == 1)
        .drop("_fez_lead", "_comprou")
    )

    return agg.join(primeiro, "session_id", "left").select(
        "session_id", "user_pseudo_id", "user_id_resolvido",
        "inicio", "fim", "duracao_seg", "eventos",
        "scroll_maximo", "video_percent_maximo",
        "fez_lead", "comprou", "canal", "dispositivo",
    )

def tabela_retencao_video(eventos: DataFrame) -> DataFrame:
    """gold/retencao_video: das sessões com video_start, % que atingiu cada marco.

    O denominador é o número de sessões que deram play (têm video_start). Para
    cada marco (10, 25, 50, 75, 100), conta as sessões cujo maior video_percent
    alcançou aquele marco, e divide pelo denominador.
    """
    por_sessao = eventos.groupBy("session_id").agg(
        F.max(F.when(F.col("event_name") == "video_start", 1).otherwise(0)).alias("deu_play"),
        F.max("video_percent").alias("video_percent_maximo"),
    ).filter(F.col("deu_play") == 1)

    # Uma única agregação: o total de sessões com play e, para cada marco,
    # quantas o alcançaram. Evita vários count() em loop.
    agg = [F.count("*").alias("sessoes_com_play")] + [
        F.sum(F.when(F.col("video_percent_maximo") >= marco, 1).otherwise(0))
         .alias(f"m{marco}")
        for marco in MARCOS_VIDEO
    ]
    resumo = por_sessao.agg(*agg)

    # desdobra em uma linha por marco
    linhas = [
        resumo.select(
            F.lit(marco).alias("marco"),
            F.col(f"m{marco}").alias("sessoes_que_atingiram"),
            F.col("sessoes_com_play"),
        )
        for marco in MARCOS_VIDEO
    ]
    df = linhas[0]
    for extra in linhas[1:]:
        df = df.unionByName(extra)

    return df.withColumn(
        "percentual",
        F.round(F.try_divide(F.col("sessoes_que_atingiram"), F.col("sessoes_com_play")), 4),
    ).orderBy("marco")

def tabela_buscas_faq(eventos: DataFrame) -> DataFrame:
    """gold/buscas_faq: por termo, total de buscas e buscas sem resultado."""
    buscas = eventos.filter(F.col("event_name") == "site_search")
    return (
        buscas.groupBy(F.col("search_term").alias("termo")).agg(
            F.count("*").alias("total_buscas"),
            F.sum(F.when(F.coalesce(F.col("search_results"), F.lit(0)) == 0, 1)
                   .otherwise(0)).alias("buscas_sem_resultado"),
        )
        .orderBy(F.desc("total_buscas"))
    )

def tabela_saidas_downloads(eventos: DataFrame) -> DataFrame:
    """gold/saidas_downloads: cliques de saída por domínio + downloads por arquivo.

    Uma tabela só (conta como 1 das 15), com a coluna `tipo` distinguindo:
      - 'saida'    : cliques com outbound=true, agrupados por link_domain
      - 'download' : file_download, agrupados por file_name
    """
    saidas = (
        eventos
        .filter((F.col("event_name") == "click") & (F.col("outbound") == F.lit(True)))
        .groupBy(F.col("link_domain").alias("chave"))
        .agg(F.count("*").alias("total"))
        .withColumn("tipo", F.lit("saida"))
    )
    downloads = (
        eventos
        .filter(F.col("event_name") == "file_download")
        .groupBy(F.col("file_name").alias("chave"))
        .agg(F.count("*").alias("total"))
        .withColumn("tipo", F.lit("download"))
    )
    colunas = ["tipo", "chave", "total"]
    return saidas.select(*colunas).unionByName(downloads.select(*colunas)) \
                 .orderBy("tipo", F.desc("total"))

def main() -> None:
    spark = criar_sessao("gold_engajamento")

    # R2: depende da Silver.
    exigir_camada(spark, glob_eventos("silver"), "python pipeline/silver.py")

    eventos = ler_eventos(spark, "silver").cache()

    sessoes = tabela_sessoes(eventos)
    retencao = tabela_retencao_video(eventos)
    buscas = tabela_buscas_faq(eventos)
    saidas = tabela_saidas_downloads(eventos)

    sessoes.write.mode("overwrite").parquet(caminho("gold/sessoes"))
    retencao.write.mode("overwrite").parquet(caminho("gold/retencao_video"))
    buscas.write.mode("overwrite").parquet(caminho("gold/buscas_faq"))
    saidas.write.mode("overwrite").parquet(caminho("gold/saidas_downloads"))

    titulo("Gold Engajamento — gravada em s3a://datalake-tracking/gold/")
    print(f"sessoes:          {sessoes.count():,} linhas")
    print(f"buscas_faq:       {buscas.count():,} termos")
    print(f"saidas_downloads: {saidas.count():,} linhas")
    print()
    print("retencao_video (% das sessões que deram play):")
    retencao.show(10, truncate=False)
    print("buscas_faq (top 10):")
    buscas.show(10, truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()
