"""Job Silver — dado limpo (requisitos R5 a R10).

A Silver pega a Bronze (cópia fiel) e entrega o dado pronto para análise:
sem duplicatas, com os `params` virando colunas tipadas, com os eventos
inválidos separados numa tabela à parte (cada um com o seu motivo), com as
sessões calculadas e as identidades costuradas, e com uma tabela de itens.

Como rodar:

    docker compose run --rm app python pipeline/silver.py

O que grava (três destinos, todos em s3a://datalake-tracking/silver/):
  - silver/event_name=.../year=.../month=.../day=...  -> eventos APROVADOS (tipados)
  - silver/eventos_rejeitados/                         -> eventos inválidos + motivo_rejeicao
  - silver/itens/                                      -> items[] explodido (R10)

Requisitos atendidos:
- R1: todo acesso ao S3 é pelo Spark com s3a:// (nada de boto3). Cuida disso o comum.py.
- R2: job executável sozinho; verifica que a Bronze existe antes de rodar.
- R5: deduplicação por event_id (cada event_id aparece uma vez só).
- R6: extração dos params (MAP) para colunas tipadas, incluindo element_id.
- R7: validação com motivo de rejeição (6 regras mínimas), uma linha por motivo.
- R8: sessionização própria (corte após 30 min de inatividade), SEM usar session_hint.
- R9: identity stitching (todo evento de um user_pseudo_id herda o user_id da pessoa).
- R10: tabela de itens com items[] explodido (uma linha por item por evento).
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
    gravar_eventos,
    ler_eventos,
    titulo,
)

# ---------------------------------------------------------------------------
# Regras de apoio para a validação (R7) e a tipagem (R6).
# ---------------------------------------------------------------------------

# Os 20 eventos do catálogo (docs/CONTRATO_EVENTO.md, "Os 20 eventos").
# Qualquer event_name fora desta lista é "evento fora do catálogo".
CATALOGO = [
    "page_view", "scroll", "click", "site_search",
    "video_start", "video_progress", "video_complete", "file_download",
    "view_item_list", "select_item", "view_item", "add_to_cart",
    "remove_from_cart", "view_cart", "begin_checkout", "add_shipping_info",
    "add_payment_info", "purchase", "refund",
    "generate_lead", "sign_up", "login",
]
# Observação: a lista acima tem 22 nomes porque o contrato conta video_* como
# três eventos e sign_up/login como dois; são os event_name válidos.

# Eventos que carregam um valor monetário em params.value (para a regra de
# "valor monetário inválido": value negativo, ou currency != BRL onde há value).
EVENTOS_COM_VALOR = [
    "view_item", "add_to_cart", "remove_from_cart", "view_cart", "begin_checkout",
    "add_shipping_info", "add_payment_info", "purchase", "refund", "generate_lead",
]

GAP_SESSAO_SEGUNDOS = 30 * 60  # R8: 30 minutos de inatividade cortam a sessão

def p(chave: str):
    """Atalho: lê uma chave do MAP params como texto. params['chave']."""
    return F.col("params").getItem(chave)

def tipar_eventos(bronze: DataFrame) -> DataFrame:
    """R6: transforma o MAP params nas colunas tipadas certas.

    O event_ts vira timestamp com try_to_timestamp: se a data estiver mal
    escrita (ou nula), vira null em vez de quebrar o job (modo ANSI do Spark 4).
    Os valores numéricos usam try_cast pelo mesmo motivo. Mantemos também as
    estruturas aninhadas úteis (traffic, device, page, items) para a Gold.
    """
    fmt = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"
    return bronze.select(
        "event_id",
        "event_name",
        # event_ts como timestamp (null se ausente/mal formado) — a validação R7 trata
        F.try_to_timestamp(F.col("event_ts"), F.lit(fmt)).alias("event_ts"),
        "user_pseudo_id",
        "user_id",
        # traffic / device / page: achatados em colunas simples
        F.col("traffic.source").alias("traffic_source"),
        F.col("traffic.medium").alias("traffic_medium"),
        F.col("traffic.campaign").alias("traffic_campaign"),
        F.col("device.category").alias("device_category"),
        F.col("device.os").alias("device_os"),
        F.col("device.browser").alias("device_browser"),
        F.col("page.location").alias("page_location"),
        F.col("page.referrer").alias("page_referrer"),
        # --- params -> colunas tipadas (R6) ---
        p("element_id").alias("element_id"),               # onde foi o clique
        p("value").try_cast("double").alias("value"),
        p("tax").try_cast("double").alias("tax"),
        p("shipping").try_cast("double").alias("shipping"),
        p("currency").alias("currency"),
        p("transaction_id").alias("transaction_id"),
        p("payment_type").alias("payment_type"),
        p("coupon").alias("coupon"),
        p("reason").alias("reason"),
        p("percent_scrolled").try_cast("int").alias("percent_scrolled"),
        p("video_percent").try_cast("int").alias("video_percent"),
        p("video_title").alias("video_title"),
        p("search_term").alias("search_term"),
        p("search_results").try_cast("int").alias("search_results"),
        p("link_url").alias("link_url"),
        p("link_domain").alias("link_domain"),
        p("link_text").alias("link_text"),
        p("outbound").try_cast("boolean").alias("outbound"),
        p("file_name").alias("file_name"),
        p("file_extension").alias("file_extension"),
        p("item_list_id").alias("item_list_id"),
        p("item_list_name").alias("item_list_name"),
        p("method").alias("method"),
        p("lead_source").alias("lead_source"),
        p("shipping_tier").alias("shipping_tier"),
        p("page_title").alias("page_title"),
        # guardamos o map original e os items para a Gold / tabela de itens
        F.col("params"),
        F.col("items"),
        # proveniência da Bronze: usada na regra do timestamp futuro (R7)
        F.col("ingerido_em"),
        # colunas de partição vindas do caminho (ler_eventos as traz)
        "year", "month", "day",
    )

def motivo_rejeicao(df: DataFrame) -> DataFrame:
    """R7: calcula o motivo_rejeicao de cada evento (null = aprovado).

    As seis regras mínimas do enunciado (seção 5.2, R7). A ordem do CASE WHEN
    define a prioridade quando um evento viola mais de uma regra; cada linha
    rejeitada carrega UM motivo.

    A regra "reembolso sem compra" depende de conhecer os transaction_id das
    compras válidas, por isso ela é resolvida depois (em aplicar_validacao).
    """
    tem_value = F.col("event_name").isin(EVENTOS_COM_VALOR)
    marco_scroll_ok = F.col("percent_scrolled").isin(25, 50, 75, 90)
    marco_video_ok = F.col("video_percent").isin(0, 10, 25, 50, 75, 100)

    return df.withColumn(
        "motivo_rejeicao",
        F.when(~F.col("event_name").isin(CATALOGO), F.lit("evento fora do catálogo"))
        # timestamp ausente ou no futuro (depois do instante de ingestão)
        .when(F.col("event_ts").isNull() | (F.col("event_ts") > F.col("ingerido_em")),
              F.lit("timestamp ausente ou futuro"))
        # purchase/refund sem transaction_id
        .when(F.col("event_name").isin("purchase", "refund") & F.col("transaction_id").isNull(),
              F.lit("transação sem id"))
        # value negativo, ou moeda != BRL em evento que tem value
        .when(tem_value & (F.col("value") < 0), F.lit("valor monetário inválido"))
        .when(tem_value & F.col("value").isNotNull() & (F.col("currency") != "BRL"),
              F.lit("valor monetário inválido"))
        # marcos fora do padrão
        .when((F.col("event_name") == "scroll") & ~marco_scroll_ok,
              F.lit("marco fora do padrão"))
        .when(F.col("event_name").isin("video_start", "video_progress", "video_complete")
              & ~marco_video_ok, F.lit("marco fora do padrão"))
        .otherwise(F.lit(None)),
    )

def aplicar_validacao(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Separa aprovados de rejeitados e resolve a regra "reembolso sem compra".

    Devolve (aprovados, rejeitados). Os rejeitados levam a coluna motivo_rejeicao.
    """
    marcado = motivo_rejeicao(df).cache()

    # Os transaction_id das compras VÁLIDAS (passaram nas regras acima). São
    # poucos milhares no máximo, então trazemos a lista para o Python e usamos
    # isin(): evita um join que poderia duplicar linhas e trata null com clareza.
    ids_compras = [
        linha["transaction_id"]
        for linha in (
            marcado
            .filter((F.col("event_name") == "purchase") & F.col("motivo_rejeicao").isNull())
            .select("transaction_id").distinct().collect()
        )
        if linha["transaction_id"] is not None
    ]

    # Um refund ainda sem motivo, cujo transaction_id não está entre as compras
    # válidas, é "reembolso sem compra". isin([]) é sempre falso: se não houver
    # nenhuma compra válida, todo refund cai aqui (comportamento correto).
    refund_orfao = (
        (F.col("event_name") == "refund")
        & F.col("motivo_rejeicao").isNull()
        & (~F.col("transaction_id").isin(ids_compras) if ids_compras else F.lit(True))
    )

    com_refund = marcado.withColumn(
        "motivo_rejeicao",
        F.when(refund_orfao, F.lit("reembolso sem compra")).otherwise(F.col("motivo_rejeicao")),
    )

    aprovados = com_refund.filter(F.col("motivo_rejeicao").isNull()).drop("motivo_rejeicao")
    rejeitados = com_refund.filter(F.col("motivo_rejeicao").isNotNull())
    return aprovados, rejeitados

def sessionizar(df: DataFrame) -> DataFrame:
    """R8: agrupa os eventos em sessões por user_pseudo_id, cortando após 30 min.

    NÃO usa session_hint (proibido). Para cada visitante, ordena por event_ts,
    olha o evento anterior com lag() e marca o início de uma sessão nova quando
    a pausa passa de 30 minutos. A soma acumulada desses inícios dá o número da
    sessão, e o session_id junta o visitante com esse número.
    """
    w = Window.partitionBy("user_pseudo_id").orderBy("event_ts")

    anterior = F.lag("event_ts").over(w)
    gap = F.col("event_ts").cast("long") - anterior.cast("long")
    nova_sessao = F.when(anterior.isNull() | (gap > GAP_SESSAO_SEGUNDOS), 1).otherwise(0)

    return (
        df.withColumn("_nova_sessao", nova_sessao)
          .withColumn("_num_sessao", F.sum("_nova_sessao").over(
              w.rowsBetween(Window.unboundedPreceding, Window.currentRow)))
          .withColumn("session_id",
                      F.concat_ws("-", F.col("user_pseudo_id"),
                                  F.col("_num_sessao").cast("string")))
          .drop("_nova_sessao", "_num_sessao")
    )

def costurar_identidade(df: DataFrame) -> DataFrame:
    """R9: identity stitching — todo evento de um user_pseudo_id herda o user_id.

    Para cada user_pseudo_id, pega o primeiro user_id não-nulo que aparece em
    qualquer evento dele e aplica a todos os eventos desse visitante (inclusive
    os anteriores ao login). Guarda o user_id resolvido em user_id_resolvido e
    mantém o user_id original.
    """
    w = Window.partitionBy("user_pseudo_id")
    # F.first(..., ignorenulls=True) sobre a janela inteira pega o primeiro não-nulo
    resolvido = F.first("user_id", ignorenulls=True).over(w)
    return df.withColumn("user_id_resolvido", resolvido)

def explodir_itens(aprovados: DataFrame) -> DataFrame:
    """R10: uma linha por item por evento (items[] explodido).

    Só os eventos que têm items (e-commerce) entram. explode gera uma linha por
    item; eventos sem item não aparecem aqui.
    Particionada por year/month/day (definição 5.4).
    """
    return (
        aprovados
        .withColumn("item", F.explode("items"))
        .select(
            "event_id", "event_name", "event_ts", "session_id",
            "user_pseudo_id", "user_id_resolvido",
            F.col("item.item_id").alias("item_id"),
            F.col("item.item_name").alias("item_name"),
            F.col("item.item_category").alias("item_category"),
            F.col("item.price").alias("price"),
            F.col("item.quantity").alias("quantity"),
            "year", "month", "day",
        )
    )

def main() -> None:
    spark = criar_sessao("silver")

    # R2: a Silver depende da Bronze.
    exigir_camada(spark, glob_eventos("bronze"), "python pipeline/bronze.py")

    bronze = ler_eventos(spark, "bronze")

    # R5: deduplicação por event_id (cada event_id uma vez só).
    # Mantemos ingerido_em (vem da Bronze) para a regra do timestamp futuro (R7).
    antes = bronze.count()
    bronze_dedup = bronze.dropDuplicates(["event_id"])
    depois = bronze_dedup.count()
    duplicatas = antes - depois

    # R6: tipar params -> colunas. (ingerido_em, vindo da Bronze, sobrevive para o R7.)
    tipado = tipar_eventos(bronze_dedup)

    # R7: validar e separar aprovados/rejeitados.
    aprovados_crus, rejeitados = aplicar_validacao(tipado)

    # R8 + R9 nos aprovados: sessões e identidade costurada.
    aprovados = costurar_identidade(sessionizar(aprovados_crus))

    # --- Gravações ---
    # 1) eventos aprovados, particionados como a Bronze (event_name/year/month/day)
    gravar_eventos(aprovados.drop("ingerido_em", "params"), "silver", "overwrite")

    # 2) eventos rejeitados (tabela própria, ao lado das partições) — um motivo por linha
    (rejeitados.select(
        "event_id", "event_name", "event_ts", "motivo_rejeicao",
        "user_pseudo_id", "value", "currency", "transaction_id",
        "percent_scrolled", "video_percent", "year", "month", "day")
     .write.mode("overwrite").parquet(caminho("silver/eventos_rejeitados")))

    # 3) itens explodidos (R10), particionados por year/month/day
    itens = explodir_itens(aprovados)
    (itens.write.mode("overwrite")
          .option("partitionOverwriteMode", "dynamic")
          .partitionBy("year", "month", "day")
          .parquet(caminho("silver/itens")))

    # --- Resumo para conferência e para o print da execução ---
    titulo("Silver — gravada em s3a://datalake-tracking/silver/")
    n_aprovados = aprovados.count()
    n_rejeitados = rejeitados.count()
    n_itens = itens.count()
    print(f"Eventos na Bronze (com duplicatas): {antes:,}")
    print(f"Duplicatas removidas (R5):          {duplicatas:,}")
    print(f"Eventos aprovados:                  {n_aprovados:,}")
    print(f"Eventos rejeitados (R7):            {n_rejeitados:,}")
    print(f"Linhas na tabela de itens (R10):    {n_itens:,}")
    print()
    print("Rejeições por motivo (R7):")
    (rejeitados.groupBy("motivo_rejeicao").count()
               .orderBy(F.desc("count")).show(20, truncate=False))

    spark.stop()

if __name__ == "__main__":
    main()
