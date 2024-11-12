# Databricks notebook source
# MAGIC %pip install /Volumes/mlops_students/netojoseaugusto/package/loans-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from pyspark.sql import SparkSession
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from databricks.sdk import WorkspaceClient

from loans.data_processor import DataBuilder
from loans.helpers import open_yaml_file
from loans.predict_loans import Evaluator, Loans
from logging_config import setup_logging
from sklearn.pipeline import Pipeline
from loans.utils import adjust_predictions
from mlflow.utils.environment import _mlflow_conda_env
from mlflow import MlflowClient
import json
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks import feature_engineering
from pyspark.sql.functions import monotonically_increasing_id, col, lit

import mlflow
from mlflow.models import infer_signature

setup_logging()

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# COMMAND ----------

configs = open_yaml_file("../../project_config.yml")

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

feature_table_name = f"{catalog_name}.{schema_name}.loan_features"
function_name = f"{catalog_name}.{schema_name}.calculate_person_income_euro"

# COMMAND ----------

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.loan_features
(person_age LONG,
 person_income LONG,
 person_home_ownership STRING,
 person_emp_length LONG,
 loan_intent STRING,
 loan_grade STRING,
 loan_amnt LONG,
 loan_int_rate DOUBLE,
 loan_percent_income DOUBLE,
 cb_person_default_on_file STRING,
 cb_person_cred_hist_length LONG,
 loan_status LONG,
 update_timestamp_utc TIMESTAMP,
 id STRING NOT NULL);
""")

# COMMAND ----------

spark.sql(f"""ALTER TABLE {catalog_name}.{schema_name}.loan_features
              ADD CONSTRAINT loan_pk PRIMARY KEY(id);""")

# COMMAND ----------

spark.sql(f"""ALTER TABLE {catalog_name}.{schema_name}.loan_features
             SET TBLPROPERTIES (delta.enableChangeDataFeed = true);""")

# COMMAND ----------


# Convert id to string in test_set
test_set = test_set.withColumn("id", col("id").cast("string"))

# Convert id to string in train_set
train_set = train_set.withColumn("id", col("id").cast("string"))

# COMMAND ----------

train_set.write.format("delta").mode("append").saveAsTable(f"{catalog_name}.{schema_name}.loan_features")
test_set.write.format("delta").mode("append").saveAsTable(f"{catalog_name}.{schema_name}.loan_features")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {function_name}(person_income INT)
RETURNS INT
LANGUAGE PYTHON AS
$$
return person_income * 1.15
$$
""")

# COMMAND ----------

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").drop("person_emp_length", "loan_amnt")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# COMMAND ----------

train_set = train_set.withColumn("person_income", train_set["person_income"].cast("int"))
train_set = train_set.withColumn("id", train_set["id"].cast("string"))

# COMMAND ----------

training_set = fe.create_training_set(
    df=train_set,
    label=target_column,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["person_emp_length", "loan_amnt"],
            lookup_key="id",
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="person_income_euro",
            input_bindings={"person_income": "person_income"},
        ),
    ],
    exclude_columns=["update_timestamp_utc"]
)

# COMMAND ----------

training_df = training_set.load_df().toPandas()

# COMMAND ----------

test_set["person_income_euro"] =  test_set["person_income"]*1.15

# COMMAND ----------

X_train = training_df[continuous_variables + categorical_variables + ["person_income_euro"]]
y_train = training_df[target_column]
X_test = test_set[continuous_variables + categorical_variables + ["person_income_euro"]]
y_test = test_set[target_column]

# COMMAND ----------

pipeline = Pipeline(steps=[
    ('classifier', CatBoostClassifier(**parameters))
])

# COMMAND ----------

mlflow.set_experiment(experiment_name="/Shared/loans-netojoseaugusto-fe")
git_sha = "ffa63b430205ff7"

with mlflow.start_run(tags={"branch": "week2",
                            "git_sha": f"{git_sha}"}) as run:
    run_id = run.info.run_id
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate and print metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc = roc_auc_score(y_test, y_pred)

    mlflow.log_param("model_type", "Catboost")
    mlflow.log_params(parameters)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc)
    
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, ROC AUC: {roc}")

    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Log model with feature engineering
    fe.log_model(
        model=pipeline,
        flavor=mlflow.sklearn,
        artifact_path="catboost-model-fe",
        training_set=training_set,
        signature=signature,
    )
mlflow.register_model(
    model_uri=f'runs:/{run_id}/catboost-model-fe',
    name=f"{catalog_name}.{schema_name}.catboost_fe")

# COMMAND ----------


