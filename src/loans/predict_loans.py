import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import logging
from typing import List, Tuple, Generator, Callable, Type

class Evaluator:
    def __init__(self, metric_function: Callable = None):
        """
        Initialize the Evaluator with a metric function.

        Parameters:
        - metric_function (Callable): A function that takes true labels and predictions and returns a score.
        """
        self.metric_function = metric_function or roc_auc_score

    def evaluate(self, y_true, y_pred) -> float:
        """
        Evaluate predictions using the metric function.

        Parameters:
        - y_true: True labels.
        - y_pred: Predicted labels or probabilities.

        Returns:
        - float: The evaluation score.
        """
        return self.metric_function(y_true, y_pred)

class Loans:
    """
    Loans class for training, evaluating, and predicting loan status using machine learning models.
    """
    def __init__(self, configs: dict, evaluator: Evaluator = None, model_class: Type = CatBoostClassifier):
        """
        Initialize the Loans class with configuration parameters.

        Parameters:
        - configs (dict): A dictionary containing configuration parameters.
        - evaluator (Evaluator): An Evaluator object for evaluating the model.
        - model_class (Type): The class of the model to use.
        """
        self.configs = configs
        self.model_class = model_class
        self.model_params = configs.get('model_params', {})
        self.categorical_variables = configs.get('categorical_variables', [])
        self.verbose = configs.get('catboost_verbose', True)
        self.random_state = configs.get('random_state', 42)
        self.evaluator = evaluator or Evaluator()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def fit_model(self, train_x: pd.DataFrame, train_y: pd.Series, val_x: pd.DataFrame, val_y: pd.Series):
        """
        Fits a model.

        Parameters:
        - train_x (DataFrame): Training features.
        - train_y (Series): Training target.
        - val_x (DataFrame): Validation features.
        - val_y (Series): Validation target.

        Returns:
        - The fitted model.
        """
        model = self.model_class(**self.model_params)
        if isinstance(model, CatBoostClassifier):
            eval_dataset = Pool(val_x, val_y, cat_features=self.categorical_variables)
            model.fit(
                train_x,
                train_y,
                eval_set=eval_dataset,
                cat_features=self.categorical_variables,
                verbose=self.verbose
            )
        else:
            model.fit(train_x, train_y)
        return model
    
    def evaluate_model(self, fitted_model, val_x: pd.DataFrame, val_y: pd.Series) -> float:
        """
        Evaluates a model using the Evaluator.

        Parameters:
        - fitted_model: The fitted model.
        - val_x (DataFrame): Validation features.
        - val_y (Series): Validation target.

        Returns:
        - float: The evaluation score.
        """
        if hasattr(fitted_model, 'predict_proba'):
            predictions = fitted_model.predict_proba(val_x)[:, 1]
        else:
            predictions = fitted_model.predict(val_x)
        return self.evaluator.evaluate(val_y, predictions)
    
    def generate_train_val_indices_for_cv(self, X: pd.DataFrame, Y: pd.Series, nfolds: int) -> Generator:
        """
        Generates training and validation indices for cross-validation.

        Parameters:
        - X (DataFrame): The training feature dataset.
        - Y (Series): The training target dataset.
        - nfolds (int): Number of folds for cross-validation.

        Returns:
        - Generator: Yields train and validation indices for each fold.
        """
        folds = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=self.random_state)
        for train_idx, val_idx in folds.split(X, Y):
            yield train_idx, val_idx
    
    def perform_cv(self, X: pd.DataFrame, Y: pd.Series, nfolds: int = 4) -> Tuple[List, List[pd.DataFrame], List[pd.Series]]:
        """
        Performs cross-validation and predicts using the model.

        Parameters:
        - X (DataFrame): The training feature dataset.
        - Y (Series): The training target dataset.
        - nfolds (int): Number of folds for cross-validation.

        Returns:
        - Tuple[List, List[DataFrame], List[Series]]: Models, validation features, validation targets.
        """
        models, vals_x, vals_y, scores = [], [], [], []
        
        for train_idx, val_idx in self.generate_train_val_indices_for_cv(X, Y, nfolds):
            train_x, train_y = X.iloc[train_idx], Y.iloc[train_idx]
            val_x, val_y = X.iloc[val_idx], Y.iloc[val_idx]
            fitted_model = self.fit_model(train_x, train_y, val_x, val_y)
            score = self.evaluate_model(fitted_model, val_x, val_y)
            scores.append(score)
            models.append(fitted_model)
            vals_x.append(val_x)
            vals_y.append(val_y)
            self.logger.info(f'Fold score: {score}')
        
        mean_score = np.mean(scores)
        self.logger.info(f'Mean score: {mean_score}')
        return models, vals_x, vals_y
    
    def predict(self, models: List, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mean predictions from a list of models.

        Parameters:
        - models: A list of trained model objects.
        - data: A pandas DataFrame containing the features for prediction, including an 'id' column.

        Returns:
        - A pandas DataFrame with two columns: 'id' and 'loan_status', where 'loan_status' is the mean prediction from the models.
        """
        data_without_id = data.drop(columns=['id'])
        predictions = np.mean(
            [model.predict_proba(data_without_id)[:, 1] if hasattr(model, 'predict_proba') else model.predict(data_without_id) for model in models],
            axis=0
        )
        return pd.DataFrame({'id': data['id'], 'loan_status': predictions})