"""Job Gold — Funil (requisitos R1, R2; seção 5.3).

Responde à pergunta nº 1 do negócio: em que etapa a jornada de compra perde
mais gente, por plano e por canal? Grava duas tabelas na camada Gold:

  - gold/funil_compra : sessões que atingiram cada etapa da jornada de compra,
                        conversão sobre a etapa anterior e sobre o topo; total
                        e por plano × canal. view_item entra como taxa à parte.
  - gold/funil_lead   : por canal, page_view da landing -> scroll >= 50% ->
                        generate_lead.

Como rodar:

    docker compose run --rm app python pipeline/gold_funil.py

Definições usadas (seção 5.4 do enunciado, para a correção bater):
  - Canal de uma sessão: "<source> / <medium>" do PRIMEIRO evento da sessão.
  - Plano de uma sessão: item_id do PRIMEIRO select_item da sessão ("(nenhum)"
    se não houver).
  - Funil (def. 4): uma sessão conta numa etapa se teve aquele evento EM
    QUALQUER MOMENTO, sem exigir as etapas anteriores.
  - Conversão sobre a etapa anterior = sessões da etapa / sessões da anterior.
    Conversão sobre o topo = sessões da etapa / sessões de view_item_list.
  - Página (def. 12): caminho de page_location sem domínio, sem # e sem ?; a
    raiz vira "/". A landing é a página "/".
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

# Etapas obrigatórias do funil de compra, na ordem (seção 5.3 / 5.4).
# view_item NÃO entra na sequência: é uma taxa à parte.
ETAPAS = [
    "view_item_list",
    "select_item",
    "add_to_cart",
    "view_cart",
    "begin_checkout",
    "add_payment_info",
    "purchase",
]

def caminho_pagina(col):
    """Def. 12: page_location -> caminho sem domínio, sem # e sem ?; raiz vira '/'.

    Ex.: https://loja.exemplo/checkout -> /checkout
         https://loja.exemplo/#planos  -> /
    """
    # tira o esquema e o domínio: fica o que vem a partir da primeira '/'
    sem_dominio = F.regexp_extract(col, r"^[a-zA-Z]+://[^/]+(/.*)?$", 1)
    # corta em # ou ?
    sem_fragmento = F.regexp_replace(sem_dominio, r"[#?].*$", "")
    # vazio -> '/'
    return F.when((sem_fragmento == "") | sem_fragmento.isNull(), F.lit("/")).otherwise(sem_fragmento)

def atributos_sessao(eventos: DataFrame) -> DataFrame:
    """Uma linha por sessão com o canal e o plano, pelas definições da seção 5.4."""
    # Canal: do PRIMEIRO evento da sessão (menor event_ts).
    w_sessao = Window.partitionBy("session_id").orderBy("event_ts")
    primeiro = (
        eventos
        .withColumn("_rn", F.row_number().over(w_sessao))
        .filter(F.col("_rn") == 1)
        .select(
            "session_id",
            F.concat_ws(" / ", F.col("traffic_source"), F.col("traffic_medium")).alias("canal"),
        )
    )

    # Plano: item_id do PRIMEIRO select_item da sessão (via items[]).
    # Pegamos o item de categoria 'plano' (um select_item traz o plano escolhido).
    selects = (
        eventos
        .filter(F.col("event_name") == "select_item")
        .withColumn("item", F.explode_outer("items"))
        .filter((F.col("item.item_category") == "plano") | F.col("item").isNull())
        .withColumn("_rn", F.row_number().over(
            Window.partitionBy("session_id").orderBy("event_ts")))
        .filter(F.col("_rn") == 1)
        .select("session_id", F.col("item.item_id").alias("plano"))
    )

    return (
        primeiro
        .join(selects, "session_id", "left")
        .withColumn("plano", F.coalesce(F.col("plano"), F.lit("(nenhum)")))
    )

def marcar_etapas(eventos: DataFrame) -> DataFrame:
    """Uma linha por sessão indicando se ela atingiu cada etapa (0/1) + view_item."""
    nomes = ETAPAS + ["view_item"]
    agg = [
        F.max(F.when(F.col("event_name") == etapa, 1).otherwise(0)).alias(etapa)
        for etapa in nomes
    ]
    return eventos.groupBy("session_id").agg(*agg)

def montar_funil(por_sessao: DataFrame, dims: list[str]) -> DataFrame:
    """Monta a tabela do funil para um nível de agregação (total ou plano×canal).

    `dims` é [] para o total, ou ["plano", "canal"] para o detalhe.
    Devolve uma linha por (dims) × etapa, com sessões, conversão sobre a
    anterior e sobre o topo, e a taxa de view_item à parte.
    """
    somas = [F.sum(etapa).alias(etapa) for etapa in ETAPAS + ["view_item"]]
    g = por_sessao.groupBy(*dims).agg(*somas) if dims else por_sessao.agg(*somas)

    # "Desdobra" as colunas de etapa em linhas (uma por etapa), mantendo a ordem.
    linhas = []
    for i, etapa in enumerate(ETAPAS):
        anterior = ETAPAS[i - 1] if i > 0 else None
        sel = [F.col(d) for d in dims] + [
            F.lit(i + 1).alias("ordem"),
            F.lit(etapa).alias("etapa"),
            F.col(etapa).alias("sessoes"),
            # conversão sobre a etapa anterior (try_divide: null se denominador 0)
            (F.lit(None).cast("double") if anterior is None
             else F.round(F.try_divide(F.col(etapa), F.col(anterior)), 4)).alias("conv_etapa_anterior"),
            # conversão sobre o topo (view_item_list)
            F.round(F.try_divide(F.col(etapa), F.col(ETAPAS[0])), 4).alias("conv_topo"),
            # taxa de view_item à parte (def. 5.3): sessões com view_item / sessões
            # com select_item. É uma taxa da jornada, não da etapa, então só
            # aparece preenchida na linha de select_item; nas demais fica null.
            (F.round(F.try_divide(F.col("view_item"), F.col("select_item")), 4)
             if etapa == "select_item" else F.lit(None).cast("double")).alias("taxa_view_item"),
        ]
        linhas.append(g.select(*sel))

    funil = linhas[0]
    for extra in linhas[1:]:
        funil = funil.unionByName(extra)
    return funil

def funil_compra(eventos: DataFrame) -> DataFrame:
    """gold/funil_compra: total + por plano × canal."""
    atrib = atributos_sessao(eventos)
    etapas = marcar_etapas(eventos)
    por_sessao = etapas.join(atrib, "session_id", "left")

    total = montar_funil(por_sessao, []).withColumn("plano", F.lit("(total)")) \
                                        .withColumn("canal", F.lit("(total)"))
    detalhe = montar_funil(por_sessao, ["plano", "canal"])

    colunas = ["plano", "canal", "ordem", "etapa", "sessoes",
               "conv_etapa_anterior", "conv_topo", "taxa_view_item"]
    return total.select(*colunas).unionByName(detalhe.select(*colunas)) \
                .orderBy("plano", "canal", "ordem")

def funil_lead(eventos: DataFrame) -> DataFrame:
    """gold/funil_lead: por canal, page_view da landing -> scroll>=50% -> generate_lead.

    As três etapas são medidas na LANDING (página '/'). O scroll conta a sessão
    que rolou pelo menos 50% na landing.
    """
    pagina = caminho_pagina(F.col("page_location"))
    na_landing = pagina == "/"

    # canal por sessão (primeiro evento)
    canal = atributos_sessao(eventos).select("session_id", "canal")

    marcada = (
        eventos
        .withColumn("_pv_landing",
                    F.when((F.col("event_name") == "page_view") & na_landing, 1).otherwise(0))
        .withColumn("_scroll50",
                    F.when((F.col("event_name") == "scroll") & na_landing
                           & (F.col("percent_scrolled") >= 50), 1).otherwise(0))
        .withColumn("_lead", F.when(F.col("event_name") == "generate_lead", 1).otherwise(0))
        .groupBy("session_id")
        .agg(F.max("_pv_landing").alias("pv_landing"),
             F.max("_scroll50").alias("scroll50"),
             F.max("_lead").alias("lead"))
        .join(canal, "session_id", "left")
    )

    por_canal = (
        marcada.groupBy("canal").agg(
            F.sum("pv_landing").alias("sessoes_page_view"),
            F.sum("scroll50").alias("sessoes_scroll_50"),
            F.sum("lead").alias("sessoes_lead"),
        )
        .withColumn("conv_scroll",
                    F.round(F.try_divide(F.col("sessoes_scroll_50"), F.col("sessoes_page_view")), 4))
        .withColumn("conv_lead",
                    F.round(F.try_divide(F.col("sessoes_lead"), F.col("sessoes_scroll_50")), 4))
    )

    total = (
        marcada.agg(
            F.sum("pv_landing").alias("sessoes_page_view"),
            F.sum("scroll50").alias("sessoes_scroll_50"),
            F.sum("lead").alias("sessoes_lead"),
        )
        .withColumn("canal", F.lit("(total)"))
        .withColumn("conv_scroll",
                    F.round(F.try_divide(F.col("sessoes_scroll_50"), F.col("sessoes_page_view")), 4))
        .withColumn("conv_lead",
                    F.round(F.try_divide(F.col("sessoes_lead"), F.col("sessoes_scroll_50")), 4))
    )

    colunas = ["canal", "sessoes_page_view", "sessoes_scroll_50", "sessoes_lead",
               "conv_scroll", "conv_lead"]
    return total.select(*colunas).unionByName(por_canal.select(*colunas)).orderBy("canal")

def main() -> None:
    spark = criar_sessao("gold_funil")

    # R2: depende da Silver.
    exigir_camada(spark, glob_eventos("silver"), "python pipeline/silver.py")

    eventos = ler_eventos(spark, "silver").cache()

    fc = funil_compra(eventos)
    fl = funil_lead(eventos)

    # Gold são tabelas pequenas, gravadas em pasta própria (sem partição).
    fc.write.mode("overwrite").parquet(caminho("gold/funil_compra"))
    fl.write.mode("overwrite").parquet(caminho("gold/funil_lead"))

    titulo("Gold Funil — gravada em s3a://datalake-tracking/gold/")
    print("funil_compra (linhas do total):")
    fc.filter(F.col("plano") == "(total)").show(20, truncate=False)
    print("funil_lead:")
    fl.show(20, truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()
