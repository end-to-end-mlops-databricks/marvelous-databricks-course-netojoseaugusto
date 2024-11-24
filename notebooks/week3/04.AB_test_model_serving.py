# Databricks notebook source
# MAGIC %pip install /Volumes/mlops_students/netojoseaugusto/package/loans-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env

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

from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    TrafficConfig,
    Route,
)

from loans.helpers import open_yaml_file
from loans.utils import adjust_predictions
from logging_config import setup_logging
import copy
import hashlib

# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")


# COMMAND ----------

client = MlflowClient()

# COMMAND ----------

configs = open_yaml_file("../../project_config.yml")

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

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

parameters_2 = copy.deepcopy(parameters)

# COMMAND ----------

parameters_2['iterations'] = 1000

# COMMAND ----------

train_set_spark = spark.table(f"{configs.get('train_uc_location')}")
train_set = spark.table(f"{configs.get('train_uc_location')}").toPandas()
test_set = spark.table(f"{configs.get('test_uc_location')}").toPandas()

# COMMAND ----------

X_train = train_set[continuous_variables + categorical_variables]
y_train = train_set[target_column]

X_test = test_set[continuous_variables + categorical_variables]
y_test = test_set[target_column]

# COMMAND ----------

pipeline = Pipeline(steps=[("classifier", CatBoostClassifier(**parameters))])

# COMMAND ----------

mlflow.set_experiment(experiment_name="/Shared/loans-netojoseaugusto_ab")
model_name = f"{catalog_name}.{schema_name}.catboost_model_ab"
git_sha = "ffa63b430205ff7"

# COMMAND ----------

with mlflow.start_run(
    tags={"model_class": "A", "git_sha": f"{git_sha}", "branch": "week2"},
) as run:
    run_id = run.info.run_id

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    roc = roc_auc_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, ROC AUC: {roc}")

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "Catboost")
    mlflow.log_params(parameters)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{configs.get('train_uc_location')}", version="0")
    mlflow.log_input(dataset, context="training")

    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="catboost-model", signature=signature)

# COMMAND ----------

model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/catboost-model", name=model_name, tags={"git_sha": f"{git_sha}"}
)

# COMMAND ----------

model_version_alias = "model_A"

client.set_registered_model_alias(model_name, model_version_alias, f"{model_version.version}")
model_uri = f"models:/{model_name}@{model_version_alias}"
model_A = mlflow.sklearn.load_model(model_uri)


# COMMAND ----------

pipeline = Pipeline(steps=[("classifier", CatBoostClassifier(**parameters_2))])

# COMMAND ----------

with mlflow.start_run(
    tags={"model_class": "B", "git_sha": f"{git_sha}", "branch": "week2"},
) as run:
    run_id = run.info.run_id

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    roc = roc_auc_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, ROC AUC: {roc}")

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "Catboost")
    mlflow.log_params(parameters)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{configs.get('train_uc_location')}", version="0")
    mlflow.log_input(dataset, context="training")

    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="catboost-model", signature=signature)

# COMMAND ----------

model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/catboost-model", name=model_name, tags={"git_sha": f"{git_sha}"}
)

# COMMAND ----------

model_version_alias = "model_B"

client.set_registered_model_alias(model_name, model_version_alias, f"{model_version.version}")
model_uri = f"models:/{model_name}@{model_version_alias}"
model_B = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

class LoansModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, models):
        self.models = models
        self.model_a = models[0]
        self.model_b = models[1]

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            loan_id = str(model_input["id"].values[0])
            hashed_id = hashlib.md5(loan_id.encode(encoding="UTF-8")).hexdigest()
            # convert a hexadecimal (base-16) string into an integer
            if int(hashed_id, 16) % 2:
                predictions = self.model_a.predict(model_input.drop(["id"], axis=1))
                return {"Prediction": predictions[0], "model": "Model A"}
            else:
                predictions = self.model_b.predict(model_input.drop(["id"], axis=1))
                return {"Prediction": predictions[0], "model": "Model B"}
        else:
            raise ValueError("Input must be a pandas DataFrame.")

# COMMAND ----------

X_train = train_set[continuous_variables + categorical_variables + ["id"]]
X_test = test_set[continuous_variables + categorical_variables + ["id"]]

# COMMAND ----------

models = [model_A, model_B]
wrapped_model = LoansModelWrapper(models)  # we pass the loaded models to the wrapper
example_input = X_test.iloc[0:1]  # Select the first row for prediction as example
example_prediction = wrapped_model.predict(
    context=None,
    model_input=example_input)
print("Example Prediction:", example_prediction)

# COMMAND ----------

mlflow.set_experiment(experiment_name="/Shared/loans-ab-testing")
model_name = f"{catalog_name}.{schema_name}.loans_model_pyfunc_ab_test"

# COMMAND ----------

with mlflow.start_run() as run:
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train,
                                model_output={"Prediction": 0,
                                              "model": "Model B"})
    dataset = mlflow.data.from_spark(train_set_spark,
                                     table_name=f"{catalog_name}.{schema_name}.train_set",
                                     version="0")
    mlflow.log_input(dataset, context="training")
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="pyfunc-loans-model-ab",
        signature=signature
    )

# COMMAND ----------

model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/pyfunc-loans-model-ab",
    name=model_name,
    tags={"git_sha": f"{git_sha}"}
)


# COMMAND ----------

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version.version}")


# COMMAND ----------

predictions = model.predict(X_test.iloc[0:1])

# COMMAND ----------

workspace = WorkspaceClient()

workspace.serving_endpoints.create(
    name="loans-model-serving-ab-test",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.catboost_model_ab",
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=model_version.version,
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

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

start_time = time.time()

model_serving_endpoint = (
    f"https://{host}/serving-endpoints/loans-model-serving-ab-test/invocations"
)

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


