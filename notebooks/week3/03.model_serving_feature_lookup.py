# Databricks notebook source
# MAGIC %pip install /Volumes/mlops_students/netojoseaugusto/package/loans-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import mlflow
from mlflow.models import infer_signature

import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier

from pyspark.sql import SparkSession

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput, TrafficConfig, Route
from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup

from loans.helpers import open_yaml_file

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# Initialize Databricks clients
workspace = WorkspaceClient()

# COMMAND ----------

configs = open_yaml_file("../../project_config.yml")

# COMMAND ----------

catalog_name = configs.get("catalog_name")
schema_name = configs.get("schema_name")
continuous_variables = configs.get("continuous_variables")
categorical_variables = configs.get("categorical_variables")

# COMMAND ----------

online_table_name = f"{catalog_name}.{schema_name}.loan_features_online"
spec = OnlineTableSpec(
    primary_key_columns=["id"],
    source_table_full_name=f"{catalog_name}.{schema_name}.loan_features", # Ensure this table exists
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
    perform_full_copy=False,
)


# COMMAND ----------

online_table_pipeline = workspace.online_tables.create(name=online_table_name, spec=spec)

# COMMAND ----------

workspace.serving_endpoints.create(
    name="catboost-loans-serving-fe",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.catboost_fe",
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=1,
            )
        ]
    ),
)


# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

required_columns = continuous_variables + categorical_variables

# COMMAND ----------

excluded_columns = ["person_emp_length", "loan_amnt"]

# COMMAND ----------

required_columns = [c for c in required_columns if c not in excluded_columns]

# COMMAND ----------

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()

# COMMAND ----------

train_set["person_income_euro"] = train_set["person_income"] * 1.15

# COMMAND ----------

train_set['person_income_euro'] = train_set['person_income_euro'].astype(int) 

# COMMAND ----------

sampled_records = train_set[['id'] + required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

train_set.dtypes

# COMMAND ----------

start_time = time.time()

model_serving_endpoint = f"https://{host}/serving-endpoints/catboost-loans-serving-fe/invocations"

response = requests.post(
    f"{model_serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": dataframe_records[0]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")


# COMMAND ----------


