# Databricks notebook source
# MAGIC %pip install /Volumes/mlops_students/netojoseaugusto/package/loans-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import mlflow
import pandas as pd
import requests
from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from pyspark.sql import SparkSession

from loans.helpers import open_yaml_file

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# Initialize Databricks clients
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# Set the MLflow registry URI
mlflow.set_registry_uri("databricks-uc")


# COMMAND ----------

configs = open_yaml_file("../../project_config.yml")

# COMMAND ----------

continuous_variables = configs.get("continuous_variables")
categorical_variables = configs.get("categorical_variables")
target_column = configs.get("target_column")
parameters = configs.get("model_params_simple")
model_verbose = configs.get("model_verbose")
catalog_name = configs.get("catalog_name")
schema_name = configs.get("schema_name")

parameters["cat_features"] = categorical_variables
parameters["verbose"] = model_verbose

# COMMAND ----------

feature_table_name = f"{catalog_name}.{schema_name}.loans_preds"
online_table_name = f"{catalog_name}.{schema_name}.loans_preds_online"

# COMMAND ----------

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# COMMAND ----------

df = pd.concat([train_set, test_set])

# COMMAND ----------

pipeline = mlflow.sklearn.load_model(f"models:/{catalog_name}.{schema_name}.catboost_model_basic/2")

# COMMAND ----------

preds_df = df[["id", "person_age", "person_income"]]
preds_df["predicted_loan_status"] = pipeline.predict(df[continuous_variables + categorical_variables])

preds_df = spark.createDataFrame(preds_df)

# COMMAND ----------

fe.create_table(
    name=feature_table_name, primary_keys=["id"], df=preds_df, description="Loans predictions feature table"
)

# Enable Change Data Feed
spark.sql(f"""
    ALTER TABLE {feature_table_name}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# COMMAND ----------

spec = OnlineTableSpec(
    primary_key_columns=["id"],
    source_table_full_name=feature_table_name,
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
    perform_full_copy=False,
)

# COMMAND ----------

online_table_pipeline = workspace.online_tables.create(name=online_table_name, spec=spec)

# COMMAND ----------

features = [
    FeatureLookup(
        table_name=feature_table_name,
        lookup_key="Id",
        feature_names=["person_age", "person_income", "predicted_loan_status"],
    )
]

# COMMAND ----------

# Create the feature spec for serving
feature_spec_name = f"{catalog_name}.{schema_name}.return_predictions"

fe.create_feature_spec(name=feature_spec_name, features=features, exclude_columns=None)

# COMMAND ----------

# 4. Create endpoing using feature spec

# Create a serving endpoint for the house prices predictions
workspace.serving_endpoints.create(
    name="loans-feature-serving",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=feature_spec_name,  # feature spec name defined in the previous step
                scale_to_zero_enabled=True,
                workload_size="Small",  # Define the workload size (Small, Medium, Large)
            )
        ]
    ),
)

# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

id_list = preds_df["id"]


# COMMAND ----------

start_time = time.time()
serving_endpoint = f"https://{host}/serving-endpoints/loans-feature-serving/invocations"
response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": [{"Id": "182"}]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")

# COMMAND ----------

response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_split": {"columns": ["Id"], "data": [["182"]]}},
)

# COMMAND ----------

serving_endpoint = f"https://{host}/serving-endpoints/loans-feature-serving/invocations"
id_list = preds_df.select("Id").rdd.flatMap(lambda x: x).collect()
headers = {"Authorization": f"Bearer {token}"}
num_requests = 10

# COMMAND ----------


def send_request():
    random_id = random.choice(id_list)
    start_time = time.time()
    response = requests.post(
        serving_endpoint,
        headers=headers,
        json={"dataframe_records": [{"Id": random_id}]},
    )
    end_time = time.time()
    latency = end_time - start_time  # Calculate latency for this request
    return response.status_code, latency


# COMMAND ----------

total_start_time = time.time()
latencies = []

# COMMAND ----------

with ThreadPoolExecutor(max_workers=100) as executor:
    futures = [executor.submit(send_request) for _ in range(num_requests)]

    for future in as_completed(futures):
        status_code, latency = future.result()
        latencies.append(latency)

total_end_time = time.time()
total_execution_time = total_end_time - total_start_time

# Calculate the average latency
average_latency = sum(latencies) / len(latencies)

print("\nTotal execution time:", total_execution_time, "seconds")
print("Average latency per request:", average_latency, "seconds")

# COMMAND ----------
