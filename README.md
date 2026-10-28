# Previsão da demanda horária da Uber por zona de Nova York com Spark ML

Trabalho da disciplina **Processamento de Dados Massivos** (IESB, Prof. Alexandre Roriz) — tipo de problema: **regressão**.

**Pergunta de negócio:** quantas corridas a Uber terá em cada zona de Nova York na próxima hora?

O notebook [`trabalho_spark_ml_regressao.ipynb`](trabalho_spark_ml_regressao.ipynb) cobre todo o pipeline com PySpark: ingestão com schema manual (`StructType`), exploração e limpeza, agregação para a grade zona × hora, atributos temporais com funções de janela, `Pipeline` do Spark ML, ajuste de hiperparâmetros com validação temporal, comparação com referências ingênuas, interpretação e persistência com `save()`.

## Base de dados

| | |
|---|---|
| Fonte | [FiveThirtyEight — Uber TLC FOIL Response](https://github.com/fivethirtyeight/uber-tlc-foil-response) (dados da NYC Taxi & Limousine Commission obtidos via pedido de acesso à informação) |
| Arquivo | `uber-trip-data/uber-raw-data-janjune-15.csv.zip` + `uber-trip-data/taxi-zone-lookup.csv` |
| Período | 01/01/2015 a 30/06/2015 |
| Tamanho | 14.270.479 registros × 4 colunas (551,7 MB descompactado) |

Os dados **não estão no repositório**: o notebook baixa os arquivos da fonte original na primeira execução (pasta `data/`, ignorada pelo git).

## Abordagem

1. **Ingestão** com `StructType` (sem `inferSchema`) e tratamento do terminador de linha `\r` da tabela de zonas.
2. **Limpeza:** nulos em `base_afiliada` rotulados; zonas "Unknown" (264/265) removidas; duplicatas exatas analisadas (precisão do horário e volume da zona) e mantidas como coincidências legítimas.
3. **Tabela analítica:** agregação zona × hora, grade completa com `crossJoin` (216 zonas × 4.344 horas, zeros incluídos), atributos de calendário e defasagens/médias móveis com `Window.partitionBy("zona_id")` — 901.800 linhas.
4. **Divisão temporal:** treino de 08/01 a 30/04, validação em maio (ajuste de hiperparâmetros), treino final até 31/05 e teste em junho. Sem divisão aleatória, para não vazar o futuro.
5. **Modelos (`Pipeline`):** Regressão Linear (`StringIndexer` → `OneHotEncoder` → `StandardScaler`) e GBT com alvo relativo ao nível da zona (`SQLTransformer` → `StringIndexer` → `GBTRegressor` → `SQLTransformer`), comparados com quatro referências ingênuas.

## Resultados no teste (junho/2015)

| Modelo | RMSE | MAE | R² | WAPE |
|---|---:|---:|---:|---:|
| Ref.: perfil histórico (zona × dia × hora) | 11,08 | 4,77 | 0,892 | 26,4% |
| Ref.: persistência (hora anterior) | 11,50 | 5,43 | 0,884 | 30,1% |
| Ref.: ingênuo sazonal (mesma hora, semana anterior) | 12,71 | 5,19 | 0,858 | 28,7% |
| Regressão Linear | 8,48 | 4,20 | 0,937 | 23,3% |
| **GBT (alvo relativo)** | **7,73** | **3,64** | **0,948** | **20,2%** |

O GBT reduz o RMSE em 30% e o MAE em 24% em relação à melhor referência: erro típico de 3,6 corridas por zona-hora, para uma demanda média de 18,1. Os sinais mais importantes são a demanda da hora anterior, a identidade da zona e a mesma hora da semana anterior. O modelo final é salvo com `save()` em `modelos/` (pasta gerada ao executar o notebook) e recarregado com `PipelineModel.load()`.

## Como executar

Requisitos: Python 3.10+ e Java 17 ou 21 (exigido pelo Spark 4.x; com PySpark 3.5, Java 11 ou 17).

```bash
pip install -r requirements.txt
jupyter notebook trabalho_spark_ml_regressao.ipynb
```

No Google Colab, basta abrir o notebook e rodar `!pip install pyspark` antes da primeira célula.

A execução completa leva cerca de 20 minutos em uma máquina com 2 núcleos (o ajuste de hiperparâmetros do GBT é a etapa mais demorada).

## Estrutura

```
├── trabalho_spark_ml_regressao.ipynb   # notebook completo, já executado
├── requirements.txt
└── .gitignore                          # exclui data/, modelos/ e artefatos temporários
```

Ao executar o notebook são criadas as pastas `data/` (arquivos baixados da fonte) e `modelos/` (modelos salvos com `save()` do Spark ML). Nenhuma das duas é versionada.
