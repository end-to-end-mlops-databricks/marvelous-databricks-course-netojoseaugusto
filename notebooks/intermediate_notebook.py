# Databricks notebook source
import sys
import os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

# COMMAND ----------

from src.loans.data_processor import DataBuilder
from src.loans.helpers import open_yaml_file

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

Y

# COMMAND ----------


