# ML in Spark

Machine Learning project with PySpark whose goal is to evaluate whether feature selection improves the performance (or at least maintains it) of a Logistic Regression model used to predict subscription to a bank deposit (`deposit`).

The work is based on the bank dataset `bank_23.pkl`, converted to CSV and loaded into Spark (Colab) to build full pipelines for preprocessing, feature selection, and modeling.

## Evaluated approaches

Four strategies are compared:

- Logistic Regression without feature selection (baseline).  
- Logistic Regression + `UnivariateFeatureSelector` (**fpr** mode, less conservative).  
- Logistic Regression + `UnivariateFeatureSelector` (**fwe** mode, more conservative).  
- Logistic Regression + `UnivariateFeatureSelector` selecting the **top 25%** of features.

In all cases, a train/validation split is used and performance is measured with AUC-ROC, analyzing the trade-off between the number of features and model performance.

## Repository contents

- `Assignment.ipynb`  
  Single notebook for the project. It includes:
  - Mounting Google Drive (in Colab) and accessing the bank CSV.  
  - Creation of the Spark session and dataset loading with `spark.read.csv`.  
  - Definition of numerical and categorical variables.  
  - Common preprocessing pipeline: `StringIndexer` and `OneHotEncoder` for categorical variables, `VectorAssembler` for `features`, and indexing of `deposit` into `label`.  
  - Train/validation split.  
  - Definition of two `LogisticRegression` models (with and without feature selection).  
  - Construction of four pipelines (baseline, FPR, FWE, Top 25%).  
  - Evaluation with `BinaryClassificationEvaluator` (AUC-ROC) and analysis of the number of selected features.  
  - Plot showing the trade-off between complexity (number of features) and performance (AUC).  
  - Extraction and interpretation of the feature names selected by the best-performing selector.

- `bank_23.pkl`  
  Dataset used. It is originally loaded from Google Drive, so to fully reproduce the workflow, the first code block must be adapted to the user’s own storage configuration.
