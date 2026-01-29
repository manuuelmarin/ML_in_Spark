# ML in Spark

Proyecto de Machine Learning con PySpark cuyo objetivo es evaluar si la selección de características mejora el rendimiento (o lo mantiene) de una Regresión Logística para predecir la suscripción a un depósito bancario (`deposit`).

Se trabaja sobre el dataset bancario `bank_23.pkl`, convertido a CSV y cargado en Spark (Colab) para construir pipelines completos de preprocesado, selección de variables y modelado.

## Enfoques evaluados

Se comparan cuatro estrategias:

- LR sin selección de características (baseline).  
- LR + `UnivariateFeatureSelector` (modo **fpr**, menos conservador).  
- LR + `UnivariateFeatureSelector` (modo **fwe**, más conservador).  
- LR + `UnivariateFeatureSelector` seleccionando el **25%** de las características.

En todos los casos se usa una partición train/validation y se mide el rendimiento con AUC-ROC, analizando el compromiso entre número de variables y desempeño del modelo.

## Contenido del repositorio

- `Assignment.ipynb`  
  Notebook único del proyecto. Incluye:
  - Montaje de Google Drive (en Colab) y acceso al CSV bancario.  
  - Creación de la sesión de Spark y carga del dataset con `spark.read.csv`.  
  - Definición de variables numéricas y categóricas.  
  - Pipeline común de preprocesado: `StringIndexer` y `OneHotEncoder` para categóricas, `VectorAssembler` para `features`, e indexación de `deposit` en `label`.  
  - División en entrenamiento y validación.  
  - Definición de dos modelos de `LogisticRegression` (con y sin selección de características).  
  - Construcción de cuatro pipelines (baseline, FPR, FWE, Top 25%).  
  - Evaluación con `BinaryClassificationEvaluator` (AUC-ROC) y análisis del número de características seleccionadas.  
  - Gráfico que muestra el trade-off entre complejidad (número de features) y rendimiento (AUC).  
  - Extracción e interpretación de las características seleccionadas por el mejor selector.

- `bank_23.pkl`
 Dataset utilizado. La extracción del mismo se hizo desde Drive, por lo que si se busca replicar el procedimiento se debe adaptar el primer bloque de código.
