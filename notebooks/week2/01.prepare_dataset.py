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

configs = open_yaml_file("../../project_config.yml")

# COMMAND ----------

builder = DataBuilder()

builder = (
    builder.load_data(configs.get("train_file_path"))
    .drop_columns(configs.get("dropped_columns"))
    .separate_features_and_target(configs.get("target_column"))
)

# COMMAND ----------


