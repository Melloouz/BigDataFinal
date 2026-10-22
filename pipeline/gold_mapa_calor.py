"""Job Gold — Mapa de calor + exportação para o site (R1, R2, R11; seção 5.3/5.5).

Grava as duas tabelas do mapa de calor e, no fim, exporta os números para o
arquivo que o site lê (site/mapa_calor.js). Responde: em que lugares da página
as pessoas mais clicam, e até onde rolam a landing?

  - gold/mapa_calor_cliques  : uma linha por (pagina, element_id).
  - gold/mapa_calor_rolagem  : uma linha por marco (0, 25, 50, 75, 90) da landing.
  - site/mapa_calor.js       : window.MAPA_CALOR = {...} (R11), gerado no disco.

Como rodar:

    docker compose run --rm app python pipeline/gold_mapa_calor.py

NOMES DE COLUNA EXATOS (R11) — o site lê exatamente estes nomes; com outro
nome, o mapa não acende:
  cliques : pagina, element_id, cliques, sessoes, sessoes_na_pagina,
            taxa_clique, intensidade
  rolagem : pagina, marco, sessoes_que_chegaram, pct_sessoes

Definições:
  12. Página = caminho de page_location, sem domínio, sem # e sem ?; raiz -> "/".
  - taxa_clique = sessoes / sessoes_na_pagina (null se divisor 0), 4 casas.
  - intensidade = cliques / maior cliques da MESMA página (0 a 1).
  - rolagem: marco 0 = todas as sessões com page_view na landing; os demais =
    sessões com scroll na landing e percent_scrolled >= marco.
    pct_sessoes = sessoes_que_chegaram / sessões do marco 0, 4 casas.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from comum import (
    arquivo_do_projeto,
    caminho,
    criar_sessao,
    exigir_camada,
    glob_eventos,
    ler_eventos,
    titulo,
)

MARCOS_ROLAGEM = [0, 25, 50, 75, 90]

def caminho_pagina(col):
    """Def. 12: page_location -> caminho sem domínio, sem # e sem ?; raiz vira '/'."""
    sem_dominio = F.regexp_extract(col, r"^[a-zA-Z]+://[^/]+(/.*)?$", 1)
    sem_fragmento = F.regexp_replace(sem_dominio, r"[#?].*$", "")
    return F.when((sem_fragmento == "") | sem_fragmento.isNull(), F.lit("/")) \
            .otherwise(sem_fragmento)

def tabela_cliques(eventos: DataFrame) -> DataFrame:
    """gold/mapa_calor_cliques: uma linha por (pagina, element_id)."""
    com_pagina = eventos.withColumn("pagina", caminho_pagina(F.col("page_location")))

    # só eventos que têm element_id E page_location (sem página não há onde desenhar)
    cliques_validos = com_pagina.filter(
        F.col("element_id").isNotNull()
        & (F.col("element_id") != "")
        & F.col("page_location").isNotNull()
    )

    # cliques (eventos) e sessões distintas que clicaram, por (pagina, element_id)
    agregado = cliques_validos.groupBy("pagina", "element_id").agg(
        F.count("*").alias("cliques"),
        F.countDistinct("session_id").alias("sessoes"),
    )

    # sessões com page_view em cada página (denominador da taxa_clique)
    pv_por_pagina = (
        com_pagina.filter(F.col("event_name") == "page_view")
        .groupBy("pagina")
        .agg(F.countDistinct("session_id").alias("sessoes_na_pagina"))
    )

    # maior "cliques" da mesma página (denominador da intensidade)
    w_pagina = Window.partitionBy("pagina")

    return (
        agregado
        .join(pv_por_pagina, "pagina", "left")
        .withColumn("sessoes_na_pagina", F.coalesce(F.col("sessoes_na_pagina"), F.lit(0)))
        .withColumn("taxa_clique",
                    F.round(F.try_divide(F.col("sessoes"), F.col("sessoes_na_pagina")), 4))
        .withColumn("_max_cliques", F.max("cliques").over(w_pagina))
        .withColumn("intensidade",
                    F.round(F.try_divide(F.col("cliques"), F.col("_max_cliques")), 4))
        .select("pagina", "element_id", "cliques", "sessoes",
                "sessoes_na_pagina", "taxa_clique", "intensidade")
        .orderBy("pagina", F.desc("cliques"))
    )

def tabela_rolagem(eventos: DataFrame) -> DataFrame:
    """gold/mapa_calor_rolagem: uma linha por marco (0,25,50,75,90) da landing."""
    com_pagina = eventos.withColumn("pagina", caminho_pagina(F.col("page_location")))
    na_landing = F.col("pagina") == "/"

    # sessões com page_view na landing (marco 0 = todas)
    pv_landing = (
        com_pagina.filter((F.col("event_name") == "page_view") & na_landing)
        .select("session_id").distinct()
    )
    total_marco0 = pv_landing.count()

    # para cada marco > 0: sessões com scroll na landing e percent_scrolled >= marco
    scrolls = com_pagina.filter((F.col("event_name") == "scroll") & na_landing)

    linhas = []
    for marco in MARCOS_ROLAGEM:
        if marco == 0:
            chegaram = total_marco0
        else:
            chegaram = (
                scrolls.filter(F.col("percent_scrolled") >= marco)
                .select("session_id").distinct().count()
            )
        linhas.append(("/", marco, chegaram, total_marco0))

    spark = eventos.sparkSession
    df = spark.createDataFrame(
        linhas, ["pagina", "marco", "sessoes_que_chegaram", "_marco0"])
    return (
        df.withColumn("pct_sessoes",
                      F.round(F.try_divide(F.col("sessoes_que_chegaram"), F.col("_marco0")), 4))
          .select("pagina", "marco", "sessoes_que_chegaram", "pct_sessoes")
          .orderBy("marco")
    )

def exportar_para_o_site(spark) -> None:
    """R11: lê as duas tabelas Gold e grava site/mapa_calor.js (window.MAPA_CALOR).

    As tabelas são pequenas (poucas dezenas de linhas), então collect() é seguro.
    O arquivo é gravado com Python comum, fora do S3 (o site lê do disco).
    """
    cliques = [l.asDict() for l in
               spark.read.parquet(caminho("gold/mapa_calor_cliques")).collect()]
    rolagem = [l.asDict() for l in
               spark.read.parquet(caminho("gold/mapa_calor_rolagem")).collect()]

    agora = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    dados = {"gerado_em": agora, "cliques": cliques, "rolagem": rolagem}

    destino = arquivo_do_projeto("site/mapa_calor.js")
    texto = json.dumps(dados, ensure_ascii=False, default=float)
    conteudo = "window.MAPA_CALOR = " + texto + ";" + chr(10)
    destino.write_text(conteudo, encoding="utf-8")

    print(f"Exportado: {destino}  ({len(cliques)} cliques, {len(rolagem)} marcos)")

def main() -> None:
    spark = criar_sessao("gold_mapa_calor")

    # R2: depende da Silver.
    exigir_camada(spark, glob_eventos("silver"), "python pipeline/silver.py")

    eventos = ler_eventos(spark, "silver").cache()

    cliques = tabela_cliques(eventos)
    rolagem = tabela_rolagem(eventos)

    cliques.write.mode("overwrite").parquet(caminho("gold/mapa_calor_cliques"))
    rolagem.write.mode("overwrite").parquet(caminho("gold/mapa_calor_rolagem"))

    # R11: exporta para o site a partir das tabelas recém-gravadas.
    exportar_para_o_site(spark)

    titulo("Gold Mapa de calor — gravada em s3a://datalake-tracking/gold/ + site/mapa_calor.js")
    print("mapa_calor_rolagem:")
    rolagem.show(truncate=False)
    print("mapa_calor_cliques (top 15 por cliques):")
    cliques.orderBy(F.desc("cliques")).show(15, truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()
