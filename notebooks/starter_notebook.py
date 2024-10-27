# Databricks notebook source
!pip install catboost

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import pandas as pd
import seaborn as sns
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from typing import List, Tuple, Generator
import yaml
import logging

# COMMAND ----------

def open_yaml_file(file_path: str) -> dict:
    """
    Open a YAML file and return its contents as a dictionary.

    Parameters:
    file_path (str): The path to the YAML file.

    Returns:
    dict: The contents of the YAML file.
    """
    with open(file_path, 'r') as file:
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

def fit_model(train_x: pd.DataFrame, train_y: pd.DataFrame, val_x: pd.DataFrame, val_y: pd.DataFrame) -> CatBoostClassifier:
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
    catboost_params = configs.get('catboost_params')
    categorical_variables = configs.get('categorical_variables')
    model = CatBoostClassifier(**catboost_params)
    eval_dataset = Pool(val_x, val_y, cat_features=categorical_variables)
    fitted_model = model.fit(train_x, train_y, eval_set=eval_dataset, cat_features=categorical_variables, verbose=configs.get('catboost_verbose'))
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

def perform_cv(X: pd.DataFrame, Y: pd.DataFrame, nfolds: int = 4) -> Tuple[List[CatBoostClassifier], List[pd.DataFrame], List[pd.DataFrame]]:
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
    
    logging.info(f'Mean ROC AUC: {np.mean(roc_auc)}')
    return models, vals_x, vals_y

# COMMAND ----------

configs = open_yaml_file('../project_config.yml')

# COMMAND ----------

df = load_data(path=configs.get('file_path'))

# COMMAND ----------

df = drop_columns(dataframe=df, column_names=configs.get('dropped_columns'))

# COMMAND ----------

X, Y = separate_features_and_target(dataframe=df, target_column=configs.get('target_column'))

# COMMAND ----------

models, vals_x, vals_y = perform_cv(X, Y)

# COMMAND ----------


