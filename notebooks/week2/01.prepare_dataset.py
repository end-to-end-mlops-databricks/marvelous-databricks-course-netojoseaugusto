# Databricks notebook source
# MAGIC %pip install /Volumes/mlops_students/netojoseaugusto/package/loans-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------


from pyspark.sql import SparkSession

from loans.data_processor import DataBuilder
from loans.helpers import open_yaml_file
from logging_config import setup_logging

setup_logging()

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

configs = open_yaml_file("../../project_config.yml")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Train

# COMMAND ----------

builder = DataBuilder()

builder = builder.load_data(configs.get("train_file_path")).separate_features_and_target(configs.get("target_column"))

# COMMAND ----------

builder.save_dataset(configs.get("train_uc_location"), spark)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Test

# COMMAND ----------

test_builder = DataBuilder()

# COMMAND ----------

test_builder = test_builder.load_data(configs.get("test_file_path"))

# COMMAND ----------

test_builder.save_dataset(configs.get("test_uc_location"), spark)

# COMMAND ----------
