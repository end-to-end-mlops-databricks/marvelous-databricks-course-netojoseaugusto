# Databricks notebook source
# MAIGC %pip install catboost

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import logging
from typing import Generator, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

# COMMAND ----------


def open_yaml_file(file_path: str) -> dict:
    """
    Open a YAML file and return its contents as a dictionary.

    Parameters:
    file_path (str): The path to the YAML file.

    Returns:
    dict: The contents of the YAML file.
    """
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


# COMMAND ----------


def load_data(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters:
    path (str): The file path to the CSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the data from the CSV file.
    """
    return pd.read_csv(path)


# COMMAND ----------


def drop_columns(dataframe: pd.DataFrame, column_names: List[str]) -> pd.DataFrame:
    """
    Drops a list of columns from the given dataframe.

    Parameters:
    - dataframe (DataFrame): The input pandas DataFrame.
    - column_names (list[str]): A list of column names to be dropped.

    Returns:
    DataFrame: A new DataFrame with the specified columns removed.
    """
    return dataframe.drop(columns=column_names)


# COMMAND ----------


def separate_features_and_target(dataframe: pd.DataFrame, target_column: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Separates the features and the target column for modeling.

    Parameters:
    - dataframe: DataFrame containing the dataset.
    - target_column: String specifying the column name to be used as the target variable.

    Returns:
    - X: DataFrame containing the features.
    - Y: DataFrame containing the target variable.
    """
    X = dataframe.drop(columns=[target_column])
    Y = dataframe[[target_column]]
    return X, Y


# COMMAND ----------


def fit_model(
    train_x: pd.DataFrame, train_y: pd.DataFrame, val_x: pd.DataFrame, val_y: pd.DataFrame
) -> CatBoostClassifier:
    """
    Fits a CatBoost model.

    Parameters:
    - train_x (DataFrame): Training features.
    - train_y (DataFrame): Training target.
    - val_x (DataFrame): Validation features.
    - val_y (DataFrame): Validation target.
    - fold_ (int): The current fold number.

    Returns:
    - CatBoostClassifier: The fitted model.
    """
    catboost_params = configs.get("model_params")
    categorical_variables = configs.get("categorical_variables")
    model = CatBoostClassifier(**catboost_params)
    eval_dataset = Pool(val_x, val_y, cat_features=categorical_variables)
    fitted_model = model.fit(
        train_x,
        train_y,
        eval_set=eval_dataset,
        cat_features=categorical_variables,
        verbose=configs.get("model_verbose"),
    )
    return fitted_model


# COMMAND ----------


def evaluate_model(fitted_model: CatBoostClassifier, val_x: pd.DataFrame, val_y: pd.DataFrame) -> float:
    """
    Evaluates a CatBoost model using ROC AUC score.

    Parameters:
    - fitted_model (CatBoostClassifier): The fitted model.
    - val_x (DataFrame): Validation features.
    - val_y (DataFrame): Validation target.

    Returns:
    - float: The ROC AUC score.
    """

    predictions = fitted_model.predict_proba(val_x)[:, 1]
    return roc_auc_score(val_y, predictions)


# COMMAND ----------


def generate_train_val_indices_for_cv(X: pd.DataFrame, Y: pd.DataFrame, nfolds: int) -> Generator:
    """
    Generates training and validation indices for cross-validation.

    Parameters:
    - X (DataFrame): The training feature dataset.
    - Y (DataFrame): The training target dataset.
    - nfolds (int): Number of folds for cross-validation.

    Returns:
    - Generator: Yields train and validation indices for each fold.
    """
    folds = StratifiedKFold(n_splits=nfolds, shuffle=True)
    for train_idx, val_idx in folds.split(X, Y):
        yield train_idx, val_idx


# COMMAND ----------


def perform_cv(
    X: pd.DataFrame, Y: pd.DataFrame, nfolds: int = 4
) -> Tuple[List[CatBoostClassifier], List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Performs cross-validation and predicts using a CatBoost model.

    Parameters:
    - X (DataFrame): The training feature dataset.
    - Y (DataFrame): The training target dataset.
    - nfolds (int): Number of folds for cross-validation.

    Returns:
    - Tuple[List[CatBoostClassifier], List[DataFrame], List[DataFrame]]: Models, validation features, validation targets.
    """
    models, vals_x, vals_y, roc_auc = [], [], [], []

    for train_idx, val_idx in generate_train_val_indices_for_cv(X, Y, nfolds):
        train_x, train_y, val_x, val_y = X.iloc[train_idx], Y.iloc[train_idx], X.iloc[val_idx], Y.iloc[val_idx]
        fitted_model = fit_model(train_x, train_y, val_x, val_y)
        roc_auc.append(evaluate_model(fitted_model, val_x, val_y))
        models.append(fitted_model)
        vals_x.append(val_x)
        vals_y.append(val_y)

    logging.info(f"Mean ROC AUC: {np.mean(roc_auc)}")
    return models, vals_x, vals_y


# COMMAND ----------


def catboost_predictions(models: List[CatBoostClassifier], data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate mean predictions from a list of CatBoost models.

    Args:
    - models: A list of trained CatBoost model objects.
    - data: A pandas DataFrame containing the features for prediction, including an 'id' column.

    Returns:
    - A pandas DataFrame with two columns: 'id' and 'loan_status', where 'loan_status' is the mean prediction from the models.
    """
    data_without_id = data.drop(columns=["id"])
    predictions = [model.predict_proba(data_without_id)[:, 1] for model in models]
    mean_predictions = np.mean(predictions, axis=0)
    return pd.DataFrame({"id": data["id"], "loan_status": mean_predictions})


# COMMAND ----------


def plot_catboost_roc(models: List[CatBoostClassifier], vals_x: List[pd.DataFrame], vals_y: List[pd.DataFrame]) -> None:
    """
    Plots the ROC curve for CatBoostClassifier models.

    Args:
    - models: A list of CatBoostClassifier models.
    - vals_x: A list of validation feature datasets corresponding to each model.
    - vals_y: A list of validation target datasets corresponding to each model.

    Returns:
    None. Displays the ROC curve plot.
    """
    plt.figure(figsize=(10, 8))
    for model, x_val, y_val in zip(models, vals_x, vals_y, strict=False):
        if isinstance(model, CatBoostClassifier):
            y_score = model.predict_proba(x_val)[:, 1]
            fpr, tpr, _ = roc_curve(y_val, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{model.__class__.__name__} (area = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")


# COMMAND ----------

configs = open_yaml_file("../project_config.yml")

# COMMAND ----------

df = load_data(path=configs.get("train_file_path"))

# COMMAND ----------

df = drop_columns(dataframe=df, column_names=configs.get("dropped_columns"))

# COMMAND ----------

X, Y = separate_features_and_target(dataframe=df, target_column=configs.get("target_column"))

# COMMAND ----------

models, vals_x, vals_y = perform_cv(X, Y)

# COMMAND ----------

plot_catboost_roc(models, vals_x, vals_y)

# COMMAND ----------

data = load_data(path=configs.get("test_file_path"))

# COMMAND ----------

predictions = catboost_predictions(models, data)
