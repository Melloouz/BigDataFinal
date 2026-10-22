"""Job Gold — Receita e aquisição (requisitos R1, R2; seção 5.3 e 5.4).

Responde à pergunta nº 3 do negócio: quanto entrou de fato (descontados os
reembolsos) e qual canal traz cliente que não desiste? Grava quatro tabelas:

  - gold/receita_diaria    : por data da compra, plano e canal — receita bruta,
                             reembolsada e líquida, pedidos e ticket médio.
  - gold/taxa_reembolso    : por plano e canal — pedidos com reembolso / pedidos.
  - gold/adesao_order_bump : percentual de compras com BUMP-MENTORIA.
  - gold/lead_para_compra  : leads, quantos compraram depois e mediana de dias
                             entre o lead e a compra (total e por canal do lead).

Como rodar:

    docker compose run --rm app python pipeline/gold_receita.py

Definições (seção 5.4):
  2. Receita bruta = soma de value dos purchase válidos; reembolsada = soma de
     value dos refund válidos, atribuída à compra pelo transaction_id;
     líquida = bruta - reembolsada; ticket médio = bruta / pedidos.
  3. Plano da compra = item de categoria 'plano' no items do purchase.
     Canal da compra = canal da sessão (primeiro evento da sessão).
  5. Lead->compra: unidade é a pessoa (user_id_resolvido; senão user_pseudo_id).
     Primeiro generate_lead e a primeira compra estritamente depois dele.
     Dias = diferença em segundos / 86400 (fracionário); mediana desses dias.
  7. Taxa de reembolso = pedidos com reembolso / pedidos (por plano e canal).
  10. Datas em UTC.
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

ITEM_ORDER_BUMP = "BUMP-MENTORIA"

def canal_por_sessao(eventos: DataFrame) -> DataFrame:
    """Canal da sessão = source / medium do primeiro evento (def. 5.4)."""
    w = Window.partitionBy("session_id").orderBy("event_ts")
    return (
        eventos
        .withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .select("session_id",
                F.concat_ws(" / ", F.col("traffic_source"),
                            F.col("traffic_medium")).alias("canal"))
    )

def base_compras(eventos: DataFrame, canais: DataFrame) -> DataFrame:
    """Uma linha por purchase válido, com plano, canal, data e value.

    O plano é o item de categoria 'plano' no items do purchase. O order bump é
    detectado à parte (tem_order_bump). A data é o dia em UTC do event_ts.
    """
    compras = eventos.filter(F.col("event_name") == "purchase")

    # plano da compra: item de categoria 'plano'
    com_itens = compras.withColumn("item", F.explode_outer("items"))
    plano = (
        com_itens
        .filter(F.col("item.item_category") == "plano")
        .groupBy("event_id")
        .agg(F.first("item.item_id", ignorenulls=True).alias("plano"))
    )
    # a compra tem order bump se algum item for o BUMP-MENTORIA
    bump = (
        com_itens
        .groupBy("event_id")
        .agg(F.max(F.when(F.col("item.item_id") == ITEM_ORDER_BUMP, 1)
                    .otherwise(0)).alias("_bump"))
        .withColumn("tem_order_bump", F.col("_bump") == 1)
        .drop("_bump")
    )

    return (
        compras.select(
            "event_id", "session_id", "transaction_id", "value",
            F.to_date("event_ts").alias("data_compra"),
        )
        .join(plano, "event_id", "left")
        .join(bump, "event_id", "left")
        .join(canais, "session_id", "left")
        .withColumn("plano", F.coalesce(F.col("plano"), F.lit("(sem plano)")))
        .withColumn("canal", F.coalesce(F.col("canal"), F.lit("(desconhecido)")))
    )

def reembolso_por_transacao(eventos: DataFrame) -> DataFrame:
    """Soma de value dos refund válidos por transaction_id (atribuída à compra)."""
    return (
        eventos.filter(F.col("event_name") == "refund")
        .groupBy("transaction_id")
        .agg(F.sum("value").alias("valor_reembolsado"),
             F.count("*").alias("qtd_reembolsos"))
    )

def tabela_receita_diaria(compras: DataFrame, reembolsos: DataFrame) -> DataFrame:
    """gold/receita_diaria: por data, plano e canal."""
    # junta o reembolso de cada compra (pelo transaction_id)
    enriquecida = (
        compras.join(reembolsos, "transaction_id", "left")
        .withColumn("valor_reembolsado", F.coalesce(F.col("valor_reembolsado"), F.lit(0.0)))
    )
    return (
        enriquecida.groupBy("data_compra", "plano", "canal").agg(
            F.round(F.sum("value"), 2).alias("receita_bruta"),
            F.round(F.sum("valor_reembolsado"), 2).alias("receita_reembolsada"),
            F.count("*").alias("pedidos"),
        )
        .withColumn("receita_liquida",
                    F.round(F.col("receita_bruta") - F.col("receita_reembolsada"), 2))
        .withColumn("ticket_medio",
                    F.round(F.try_divide(F.col("receita_bruta"), F.col("pedidos")), 2))
        .orderBy("data_compra", "plano", "canal")
    )

def tabela_taxa_reembolso(compras: DataFrame, reembolsos: DataFrame) -> DataFrame:
    """gold/taxa_reembolso: por plano e canal da compra.

    Taxa = pedidos com reembolso / pedidos (def. 7). Um pedido "tem reembolso"
    se o transaction_id dele aparece entre os refunds válidos.
    """
    enriquecida = (
        compras.join(reembolsos.select("transaction_id",
                     F.lit(1).alias("_tem_reembolso")), "transaction_id", "left")
        .withColumn("tem_reembolso", F.coalesce(F.col("_tem_reembolso"), F.lit(0)))
    )
    return (
        enriquecida.groupBy("plano", "canal").agg(
            F.count("*").alias("pedidos"),
            F.sum("tem_reembolso").alias("pedidos_com_reembolso"),
        )
        .withColumn("taxa_reembolso",
                    F.round(F.try_divide(F.col("pedidos_com_reembolso"), F.col("pedidos")), 4))
        .orderBy("plano", "canal")
    )

def tabela_adesao_order_bump(compras: DataFrame) -> DataFrame:
    """gold/adesao_order_bump: percentual de compras com BUMP-MENTORIA."""
    return compras.agg(
        F.count("*").alias("total_compras"),
        F.sum(F.when(F.col("tem_order_bump"), 1).otherwise(0)).alias("compras_com_bump"),
    ).withColumn(
        "adesao",
        F.round(F.try_divide(F.col("compras_com_bump"), F.col("total_compras")), 4),
    )

def tabela_lead_para_compra(eventos: DataFrame, canais: DataFrame) -> DataFrame:
    """gold/lead_para_compra: leads, quantos compraram depois e mediana de dias.

    Unidade = pessoa: user_id_resolvido se houver, senão user_pseudo_id (def. 5).
    Vale o PRIMEIRO generate_lead da pessoa e a PRIMEIRA compra estritamente
    depois dele. Dias = (ts_compra - ts_lead) em segundos / 86400 (fracionário).
    Total e uma linha por canal do lead.
    """
    pessoa = F.coalesce(F.col("user_id_resolvido"), F.col("user_pseudo_id"))

    # primeiro lead de cada pessoa (com a sessão, para pegar o canal do lead)
    leads = (
        eventos.filter(F.col("event_name") == "generate_lead")
        .withColumn("pessoa", pessoa)
        .withColumn("_rn", F.row_number().over(
            Window.partitionBy("pessoa").orderBy("event_ts")))
        .filter(F.col("_rn") == 1)
        .select("pessoa", F.col("event_ts").alias("ts_lead"), "session_id")
        .join(canais, "session_id", "left")
        .withColumnRenamed("canal", "canal_lead")
        .withColumn("canal_lead", F.coalesce(F.col("canal_lead"), F.lit("(desconhecido)")))
    )

    # compras de cada pessoa, com o timestamp
    compras = (
        eventos.filter(F.col("event_name") == "purchase")
        .withColumn("pessoa", pessoa)
        .select("pessoa", F.col("event_ts").alias("ts_compra"))
    )

    # para cada lead, a primeira compra estritamente depois do lead
    ligado = (
        leads.join(compras, "pessoa", "left")
        .withColumn("compra_valida",
                    F.when(F.col("ts_compra") > F.col("ts_lead"), F.col("ts_compra")))
        .groupBy("pessoa", "canal_lead", "ts_lead")
        .agg(F.min("compra_valida").alias("primeira_compra"))
        .withColumn("comprou", F.col("primeira_compra").isNotNull())
        .withColumn("dias",
                    F.when(F.col("comprou"),
                           (F.col("primeira_compra").cast("long") - F.col("ts_lead").cast("long"))
                           / F.lit(86400.0)))
    )

    def resumo(df: DataFrame, dims: list[str]) -> DataFrame:
        g = df.groupBy(*dims) if dims else df
        base = g.agg(
            F.count("*").alias("leads"),
            F.sum(F.when(F.col("comprou"), 1).otherwise(0)).alias("leads_que_compraram"),
            F.round(F.percentile_approx("dias", 0.5), 2).alias("mediana_dias"),
        )
        return base.withColumn(
            "taxa_conversao",
            F.round(F.try_divide(F.col("leads_que_compraram"), F.col("leads")), 4),
        )

    total = (resumo(ligado, [])
             .withColumn("canal_lead", F.lit("(total)")))
    por_canal = resumo(ligado, ["canal_lead"])

    colunas = ["canal_lead", "leads", "leads_que_compraram", "taxa_conversao", "mediana_dias"]
    return total.select(*colunas).unionByName(por_canal.select(*colunas)).orderBy("canal_lead")

def main() -> None:
    spark = criar_sessao("gold_receita")

    # R2: depende da Silver.
    exigir_camada(spark, glob_eventos("silver"), "python pipeline/silver.py")

    eventos = ler_eventos(spark, "silver").cache()
    canais = canal_por_sessao(eventos).cache()

    compras = base_compras(eventos, canais).cache()
    reembolsos = reembolso_por_transacao(eventos)

    receita = tabela_receita_diaria(compras, reembolsos)
    taxa = tabela_taxa_reembolso(compras, reembolsos)
    bump = tabela_adesao_order_bump(compras)
    lead = tabela_lead_para_compra(eventos, canais)

    receita.write.mode("overwrite").parquet(caminho("gold/receita_diaria"))
    taxa.write.mode("overwrite").parquet(caminho("gold/taxa_reembolso"))
    bump.write.mode("overwrite").parquet(caminho("gold/adesao_order_bump"))
    lead.write.mode("overwrite").parquet(caminho("gold/lead_para_compra"))

    titulo("Gold Receita e aquisição — gravada em s3a://datalake-tracking/gold/")
    print("adesao_order_bump:")
    bump.show(truncate=False)
    print("lead_para_compra:")
    lead.show(truncate=False)
    print("receita_diaria (totais gerais):")
    (receita.agg(
        F.round(F.sum("receita_bruta"), 2).alias("bruta"),
        F.round(F.sum("receita_reembolsada"), 2).alias("reembolsada"),
        F.round(F.sum("receita_liquida"), 2).alias("liquida"),
        F.sum("pedidos").alias("pedidos"),
    ).show(truncate=False))

    spark.stop()

if __name__ == "__main__":
    main()
