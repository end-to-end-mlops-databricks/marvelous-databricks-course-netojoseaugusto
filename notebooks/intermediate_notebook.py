# Databricks notebook source
!pip install catboost

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

# COMMAND ----------

from src.loans.data_processor import DataBuilder
from src.loans.helpers import open_yaml_file
from src.loans.predict_loans import Evaluator, Loans
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

# COMMAND ----------

configs = open_yaml_file('../project_config.yml')

# COMMAND ----------

builder = DataBuilder()

builder = (
    builder
    .load_data(configs.get('train_file_path'))
    .drop_columns(configs.get('dropped_columns'))
    .separate_features_and_target(configs.get('target_column'))
)

dataframe = builder.get_dataframe()
X, Y = builder.get_features_and_target()

# COMMAND ----------

evaluator = Evaluator(metric_function=roc_auc_score)

# COMMAND ----------

loans = Loans(configs=configs, evaluator=evaluator, model_class=CatBoostClassifier)

# COMMAND ----------

models, vals_x, vals_y = loans.perform_cv(X, Y, nfolds=5)

# COMMAND ----------


