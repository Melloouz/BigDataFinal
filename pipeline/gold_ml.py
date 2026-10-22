"""Job Gold — ML: propensão e segmentação (requisitos R1, R2; seção 5.3 e 5.4).

Responde à pergunta nº 4 do negócio: quais sessões estão perto de comprar, e
que perfis de visitante existem? Grava três tabelas:

  - gold/propensao_sessao     : uma linha por sessão com a probabilidade de
                                compra (modelo de classificação).
  - gold/metricas_modelo      : AUC, F1, precisão e recall no conjunto de teste,
                                e a lista de features usadas.
  - gold/segmentos_visitantes : o grupo (cluster) de cada visitante e o perfil
                                médio de cada grupo.

Como rodar:

    docker compose run --rm app python pipeline/gold_ml.py

REGRA DE OURO — anti-vazamento (seção 5.3):
  As features da propensão só podem usar o comportamento da sessão ANTES do
  PRIMEIRO add_to_cart: rolagem, vídeo, planos vistos, plano selecionado,
  detalhe aberto, buscas, downloads, cliques, canal e dispositivo. Qualquer
  evento do checkout (add_to_cart, view_cart, begin_checkout, add_payment_info,
  purchase...) é PROIBIDO como feature, porque revelaria a compra.

Definições:
  6. Propensão: uma linha por sessão (todas as sessões da Silver). O rótulo é
     verdadeiro quando a sessão contém um purchase.
  11. F1, precisão e recall referem-se à classe positiva (sessão com compra).
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import (
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.functions import vector_to_array

from comum import (
    caminho,
    criar_sessao,
    exigir_camada,
    glob_eventos,
    ler_eventos,
    titulo,
)

# Features numéricas construídas a partir do comportamento PRÉ-carrinho.
FEATURES_NUMERICAS = [
    "n_eventos", "n_page_view", "n_scroll", "scroll_maximo",
    "n_video", "video_percent_maximo", "n_select_item", "n_view_item",
    "n_site_search", "n_file_download", "n_click", "duracao_pre_seg",
]
# Features categóricas (viram one-hot).
FEATURES_CATEGORICAS = ["canal", "dispositivo"]

# Nº de grupos da clusterização de visitantes.
K_GRUPOS = 4

def construir_features(eventos: DataFrame) -> DataFrame:
    """Uma linha por sessão com as features PRÉ-carrinho e o rótulo 'label'.

    Para cada sessão: acha o event_ts do PRIMEIRO add_to_cart (o marco) e só
    agrega eventos ANTERIORES a ele. Sessões sem add_to_cart usam todos os
    eventos. O rótulo (label) é 1 se a sessão tem purchase em qualquer momento.
    """
    w = Window.partitionBy("session_id")

    # marco: menor event_ts de um add_to_cart na sessão (null se não houve)
    marco = F.min(
        F.when(F.col("event_name") == "add_to_cart", F.col("event_ts"))
    ).over(w)

    rotulado = (
        eventos
        .withColumn("_marco_carrinho", marco)
        # pré-carrinho: nunca houve add_to_cart, OU evento anterior ao marco
        .withColumn("_pre",
                    F.col("_marco_carrinho").isNull()
                    | (F.col("event_ts") < F.col("_marco_carrinho")))
        # rótulo da sessão: comprou (tem purchase em qualquer momento)
        .withColumn("_comprou_sessao",
                    F.max(F.when(F.col("event_name") == "purchase", 1).otherwise(0)).over(w))
    )

    pre = rotulado.filter(F.col("_pre"))

    # canal e dispositivo: do primeiro evento da sessão (sempre pré-carrinho)
    w_ord = Window.partitionBy("session_id").orderBy("event_ts")
    atributos = (
        rotulado
        .withColumn("_rn", F.row_number().over(w_ord))
        .filter(F.col("_rn") == 1)
        .select(
            "session_id",
            F.concat_ws(" / ", F.col("traffic_source"), F.col("traffic_medium")).alias("canal"),
            F.coalesce(F.col("device_category"), F.lit("(desconhecido)")).alias("dispositivo"),
            F.col("user_pseudo_id"),
            F.col("_comprou_sessao").cast("double").alias("label"),
        )
    )

    def conta(nome):
        return F.sum(F.when(F.col("event_name") == nome, 1).otherwise(0))

    agg = (
        pre.groupBy("session_id").agg(
            F.count("*").alias("n_eventos"),
            conta("page_view").alias("n_page_view"),
            conta("scroll").alias("n_scroll"),
            F.coalesce(F.max("percent_scrolled"), F.lit(0)).alias("scroll_maximo"),
            (conta("video_start") + conta("video_progress")
             + conta("video_complete")).alias("n_video"),
            F.coalesce(F.max("video_percent"), F.lit(0)).alias("video_percent_maximo"),
            conta("select_item").alias("n_select_item"),
            conta("view_item").alias("n_view_item"),
            conta("site_search").alias("n_site_search"),
            conta("file_download").alias("n_file_download"),
            conta("click").alias("n_click"),
            (F.max("event_ts").cast("long")
             - F.min("event_ts").cast("long")).alias("duracao_pre_seg"),
        )
    )

    # junta atributos (sessões sem eventos pré-carrinho ficam com 0 nas contagens)
    base = atributos.join(agg, "session_id", "left")
    for c in FEATURES_NUMERICAS:
        base = base.withColumn(c, F.coalesce(F.col(c), F.lit(0)).cast("double"))
    return base

def treinar_propensao(base: DataFrame):
    """Treina a Regressão Logística. Devolve (predicoes_todas, metricas_dict).

    Split 80/20, treina só no treino, avalia no teste (AUC, F1, precisão e
    recall da classe positiva) e prevê a probabilidade de compra para TODAS as
    sessões.
    """
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in FEATURES_CATEGORICAS
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_oh")
        for c in FEATURES_CATEGORICAS
    ]
    entradas = FEATURES_NUMERICAS + [f"{c}_oh" for c in FEATURES_CATEGORICAS]
    assembler = VectorAssembler(inputCols=entradas, outputCol="features_brutas")
    scaler = StandardScaler(inputCol="features_brutas", outputCol="features",
                            withMean=True, withStd=True)
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50)

    pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, lr])

    treino, teste = base.randomSplit([0.8, 0.2], seed=42)
    modelo = pipeline.fit(treino)

    pred_teste = modelo.transform(teste)
    auc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction",
        metricName="areaUnderROC").evaluate(pred_teste)

    # precisão / recall / F1 da classe positiva (label=1)
    vp = pred_teste.filter((F.col("label") == 1) & (F.col("prediction") == 1)).count()
    fp = pred_teste.filter((F.col("label") == 0) & (F.col("prediction") == 1)).count()
    fn = pred_teste.filter((F.col("label") == 1) & (F.col("prediction") == 0)).count()
    precisao = vp / (vp + fp) if (vp + fp) > 0 else 0.0
    recall = vp / (vp + fn) if (vp + fn) > 0 else 0.0
    f1 = (2 * precisao * recall / (precisao + recall)) if (precisao + recall) > 0 else 0.0

    pred_todas = modelo.transform(base)

    metricas = {
        "auc": round(float(auc), 4),
        "f1": round(f1, 4),
        "precisao": round(precisao, 4),
        "recall": round(recall, 4),
    }
    return pred_todas, metricas

def segmentar_visitantes(base: DataFrame) -> DataFrame:
    """gold/segmentos_visitantes: agrupa VISITANTES (user_pseudo_id) em K grupos.

    Agrega as sessões por visitante (nº de sessões, médias de comportamento,
    se comprou) e roda o KMeans. Devolve o grupo de cada visitante + as colunas
    de perfil (o perfil médio de cada grupo sai no resumo do main).
    """
    por_visitante = base.groupBy("user_pseudo_id").agg(
        F.count("*").alias("sessoes"),
        F.round(F.avg("n_eventos"), 3).alias("media_eventos"),
        F.round(F.avg("scroll_maximo"), 3).alias("media_scroll"),
        F.round(F.avg("video_percent_maximo"), 3).alias("media_video"),
        F.round(F.avg("n_select_item"), 3).alias("media_select_item"),
        F.max("label").alias("comprou"),
    )

    cols = ["sessoes", "media_eventos", "media_scroll", "media_video", "media_select_item"]
    assembler = VectorAssembler(inputCols=cols, outputCol="features_brutas")
    scaler = StandardScaler(inputCol="features_brutas", outputCol="features",
                            withMean=True, withStd=True)
    kmeans = KMeans(featuresCol="features", predictionCol="grupo", k=K_GRUPOS, seed=42)

    pipeline = Pipeline(stages=[assembler, scaler, kmeans])
    modelo = pipeline.fit(por_visitante)
    resultado = modelo.transform(por_visitante)

    return resultado.select(
        "user_pseudo_id", "grupo", "sessoes", "media_eventos", "media_scroll",
        "media_video", "media_select_item", "comprou",
    )

def main() -> None:
    spark = criar_sessao("gold_ml")

    # R2: depende da Silver.
    exigir_camada(spark, glob_eventos("silver"), "python pipeline/silver.py")

    eventos = ler_eventos(spark, "silver").cache()
    base = construir_features(eventos).cache()

    # --- Propensão (classificação) ---
    pred_todas, metricas = treinar_propensao(base)

    # probabilidade da classe positiva = 2º elemento do vetor probability
    prob1 = F.element_at(vector_to_array(F.col("probability")), 2)
    propensao = (
        pred_todas.select(
            "session_id", "user_pseudo_id",
            F.col("label").cast("int").alias("comprou"),
            F.round(prob1, 4).alias("probabilidade_compra"),
            F.col("prediction").cast("int").alias("previsao"),
        )
        .orderBy(F.desc("probabilidade_compra"))
    )
    propensao.write.mode("overwrite").parquet(caminho("gold/propensao_sessao"))

    # --- Métricas do modelo ---
    metr = spark.createDataFrame(
        [(metricas["auc"], metricas["f1"], metricas["precisao"], metricas["recall"],
          ", ".join(FEATURES_NUMERICAS + FEATURES_CATEGORICAS))],
        ["auc", "f1", "precisao", "recall", "features"],
    )
    metr.write.mode("overwrite").parquet(caminho("gold/metricas_modelo"))

    # --- Segmentação (clusterização por visitante) ---
    segmentos = segmentar_visitantes(base)
    segmentos.write.mode("overwrite").parquet(caminho("gold/segmentos_visitantes"))

    titulo("Gold ML — gravada em s3a://datalake-tracking/gold/")
    print("metricas_modelo:")
    metr.show(truncate=False)
    print("propensao_sessao (top 10 mais prováveis):")
    propensao.show(10, truncate=False)
    print("segmentos_visitantes (perfil médio por grupo):")
    (segmentos.groupBy("grupo").agg(
        F.count("*").alias("visitantes"),
        F.round(F.avg("sessoes"), 2).alias("media_sessoes"),
        F.round(F.avg("media_scroll"), 2).alias("media_scroll"),
        F.round(F.avg("media_video"), 2).alias("media_video"),
        F.round(F.avg("comprou"), 4).alias("taxa_compra"),
    ).orderBy("grupo").show(truncate=False))

    spark.stop()

if __name__ == "__main__":
    main()
