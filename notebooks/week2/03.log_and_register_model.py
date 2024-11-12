# Databricks notebook source
# MAGIC %pip install /Volumes/mlops_students/netojoseaugusto/package/loans-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import mlflow
from catboost import CatBoostClassifier
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from loans.helpers import open_yaml_file
from logging_config import setup_logging

setup_logging()

# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

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

parameters["cat_features"] = categorical_variables
parameters["verbose"] = model_verbose

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

mlflow.set_experiment(experiment_name="/Shared/loans-netojoseaugusto")
git_sha = "ffa63b430205ff7"

# COMMAND ----------

with mlflow.start_run(
    tags={"git_sha": f"{git_sha}", "branch": "week2"},
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
    model_uri=f"runs:/{run_id}/catboost-model",
    name=f"{configs.get('catalog_name')}.{configs.get('schema_name')}.catboost_model_basic",
    tags={"git_sha": f"{git_sha}"},
)

# COMMAND ----------

run = mlflow.get_run(run_id)
dataset_info = run.inputs.dataset_inputs[0].dataset
dataset_source = mlflow.data.get_source(dataset_info)
dataset_source.load()
