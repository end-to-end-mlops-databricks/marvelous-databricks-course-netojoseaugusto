# Databricks notebook source
# MAGIC %pip install /Volumes/mlops_students/netojoseaugusto/package/loans-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from loans.data_processor import DataBuilder
from loans.helpers import open_yaml_file
from loans.predict_loans import Evaluator, Loans
from logging_config import setup_logging

setup_logging()

# COMMAND ----------

configs = open_yaml_file("../project_config.yml")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Training Data

# COMMAND ----------

builder = DataBuilder()

builder = (
    builder.load_data(configs.get("train_file_path"))
    .drop_columns(configs.get("dropped_columns"))
    .separate_features_and_target(configs.get("target_column"))
)

dataframe = builder.get_dataframe()
X, Y = builder.get_features_and_target()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Model

# COMMAND ----------

evaluator = Evaluator(metric_function=roc_auc_score)

# COMMAND ----------

loans = Loans(configs=configs, evaluator=evaluator, model_class=CatBoostClassifier)

# COMMAND ----------

loans.perform_cv(X, Y, nfolds=2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Test Data

# COMMAND ----------

test_builder = DataBuilder()

# COMMAND ----------

test_builder = test_builder.load_data(configs.get("test_file_path"))

# COMMAND ----------

test_df = test_builder.get_dataframe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predict

# COMMAND ----------

result = loans.predict_cv(test_df)

# COMMAND ----------


