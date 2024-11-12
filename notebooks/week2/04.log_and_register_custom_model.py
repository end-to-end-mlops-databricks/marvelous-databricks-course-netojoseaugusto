# Databricks notebook source
# MAGIC %pip install /Volumes/mlops_students/netojoseaugusto/package/loans-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from pyspark.sql import SparkSession
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

from loans.data_processor import DataBuilder
from loans.helpers import open_yaml_file
from loans.predict_loans import Evaluator, Loans
from logging_config import setup_logging
from sklearn.pipeline import Pipeline
from loans.utils import adjust_predictions
from mlflow.utils.environment import _mlflow_conda_env
from mlflow import MlflowClient
import json

import mlflow
from mlflow.models import infer_signature

setup_logging()

# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri('databricks-uc') 
client = MlflowClient()

# COMMAND ----------

configs = open_yaml_file("../../project_config.yml")

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

continuous_variables = configs.get('continuous_variables')
categorical_variables = configs.get('categorical_variables')
target_column = configs.get('target_column')
parameters = configs.get('model_params_simple')
model_verbose = configs.get('model_verbose')
catalog_name = configs.get('catalog_name')
schema_name = configs.get('schema_name')

parameters['cat_features'] = categorical_variables
parameters['verbose'] = model_verbose

# COMMAND ----------

run_id = mlflow.search_runs(
    experiment_names=["/Shared/loans-netojoseaugusto"],
    filter_string="tags.branch='week2'",
).run_id[0]

# COMMAND ----------

model = mlflow.sklearn.load_model(f'runs:/{run_id}/catboost-model')

# COMMAND ----------

class LoanModelWrapper(mlflow.pyfunc.PythonModel):
    
    def __init__(self, model):
        self.model = model
        
    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            predictions = self.model.predict_proba(model_input)[:, 1]
            predictions = {"Prediction": adjust_predictions(
                predictions[0])}
            return predictions
        else:
            raise ValueError("Input must be a pandas DataFrame.")

# COMMAND ----------

spark_train_set = spark.table(f"{configs.get('train_uc_location')}")

# COMMAND ----------

train_set = spark.table(f"{configs.get('train_uc_location')}").toPandas()
test_set = spark.table(f"{configs.get('test_uc_location')}").toPandas()

# COMMAND ----------

X_train = train_set[continuous_variables + categorical_variables]
y_train = train_set[target_column]

X_test = test_set[continuous_variables + categorical_variables]
y_test = test_set[target_column]

# COMMAND ----------

wrapped_model = LoanModelWrapper(model) 
example_input = X_test.iloc[0:1] 
example_prediction = wrapped_model.predict(context=None, model_input=example_input)
print("Example Prediction:", example_prediction)

# COMMAND ----------

mlflow.set_experiment(experiment_name="/Shared/loans-netojoseaugusto-pyfunc")
git_sha = "ffa63b430205ff7"

# COMMAND ----------

with mlflow.start_run(tags={"branch": "week2",
                            "git_sha": f"{git_sha}"}) as run:
    
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train, model_output={'Prediction': example_prediction})
    dataset = mlflow.data.from_spark(
        spark_train_set, table_name=f"{configs.get('train_uc_location')}", version="0")
    mlflow.log_input(dataset, context="training")
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["package/loans-0.0.1-py3-none-any.whl",
                             ],
        additional_conda_channels=None,
    )
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="pyfunc-catboost-model",
        code_paths = ["../loans-0.0.1-py3-none-any.whl"],
        signature=signature
    )

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(f'runs:/{run_id}/pyfunc-catboost-model')
loaded_model.unwrap_python_model()

# COMMAND ----------

model_name = f"{catalog_name}.{schema_name}.pyfunc-catboost-model"

# COMMAND ----------

model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/pyfunc-catboost-model',
    name=model_name,
    tags={"git_sha": f"{git_sha}"})

# COMMAND ----------

with open("model_version.json", "w") as json_file:
    json.dump(model_version.__dict__, json_file, indent=4)

# COMMAND ----------

model_version_alias = "the_best_model"
client.set_registered_model_alias(model_name, model_version_alias, "1")  

# COMMAND ----------

model_uri = f"models:/{model_name}@{model_version_alias}"
model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

client.get_model_version_by_alias(model_name, model_version_alias)

# COMMAND ----------

model

# COMMAND ----------


