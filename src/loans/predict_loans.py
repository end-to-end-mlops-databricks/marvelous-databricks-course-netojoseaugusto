import logging
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        configs: Dict[str, Any],
        evaluator: Optional[Evaluator] = None,
        model_class: Type = CatBoostClassifier,
    ) -> None:
        """
        Initialize the Loans class with configuration parameters.

        Parameters
        ----------
        configs : dict
            A dictionary containing configuration parameters.
        evaluator : Evaluator, optional
            An Evaluator object for evaluating the model.
        model_class : Type, default=CatBoostClassifier
            The class of the model to use.
        """
        self.configs = configs
        self.model_class = model_class
        self.model_params = configs.get("model_params", {})
        self.categorical_variables = configs.get("categorical_variables", [])
        self.verbose = configs.get("model_verbose", True)
        self.evaluator = evaluator or Evaluator()
        self.models: List[Any] = []
        self.vals_x: List[pd.DataFrame] = []
        self.vals_y: List[pd.Series] = []

    def fit_model(
        self,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        val_x: pd.DataFrame,
        val_y: pd.Series,
    ) -> Any:
        """
        Fit a model.

        Parameters
        ----------
        train_x : pd.DataFrame
            Training features.
        train_y : pd.Series
            Training target.
        val_x : pd.DataFrame
            Validation features.
        val_y : pd.Series
            Validation target.

        Returns
        -------
        Any
            The fitted model.
        """
        model = self.model_class(**self.model_params)
        logger.info(f"Starting model with params {self.model_params}")

        if isinstance(model, CatBoostClassifier):
            eval_dataset = Pool(val_x, val_y, cat_features=self.categorical_variables)
            model.fit(
                train_x,
                train_y,
                eval_set=eval_dataset,
                cat_features=self.categorical_variables,
                verbose=self.verbose,
            )
        else:
            model.fit(train_x, train_y)
        return model

    def evaluate_model(self, fitted_model: Any, val_x: pd.DataFrame, val_y: pd.Series) -> float:
        """
        Evaluate a model using the Evaluator.

        Parameters
        ----------
        fitted_model : Any
            The fitted model.
        val_x : pd.DataFrame
            Validation features.
        val_y : pd.Series
            Validation target.

        Returns
        -------
        float
            The evaluation score.
        """
        if hasattr(fitted_model, "predict_proba"):
            predictions = fitted_model.predict_proba(val_x)[:, 1]
        else:
            predictions = fitted_model.predict(val_x)
        return self.evaluator.evaluate(val_y, predictions)

    def generate_train_val_indices_for_cv(
        self, X: pd.DataFrame, Y: pd.Series, nfolds: int
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate training and validation indices for cross-validation.

        Parameters
        ----------
        X : pd.DataFrame
            The training feature dataset.
        Y : pd.Series
            The training target dataset.
        nfolds : int
            Number of folds for cross-validation.

        Yields
        ------
        tuple of np.ndarray
            Train and validation indices for each fold.
        """
        folds = StratifiedKFold(n_splits=nfolds, shuffle=True)
        for train_idx, val_idx in folds.split(X, Y):
            yield train_idx, val_idx

    def perform_cv(self, X: pd.DataFrame, Y: pd.Series, nfolds: int = 4) -> None:
        """
        Perform cross-validation and train models.

        Parameters
        ----------
        X : pd.DataFrame
            The training feature dataset.
        Y : pd.Series
            The training target dataset.
        nfolds : int, default=4
            Number of folds for cross-validation.

        Returns
        -------
        None
        """
        models: List[Any] = []
        vals_x: List[pd.DataFrame] = []
        vals_y: List[pd.Series] = []
        scores: List[float] = []

        for train_idx, val_idx in self.generate_train_val_indices_for_cv(X, Y, nfolds):
            train_x, train_y = X.iloc[train_idx], Y.iloc[train_idx]
            val_x, val_y = X.iloc[val_idx], Y.iloc[val_idx]
            try:
                fitted_model = self.fit_model(train_x, train_y, val_x, val_y)
                score = self.evaluate_model(fitted_model, val_x, val_y)
                scores.append(score)
                models.append(fitted_model)
                vals_x.append(val_x)
                vals_y.append(val_y)
                logger.info(f"Fold score: {score}")
            except Exception as e:
                logger.error(f"Error during cross-validation fold: {e}")
                continue

        if scores:
            mean_score = np.mean(scores)
            logger.info(f"Mean score: {mean_score}")
        else:
            logger.warning("No successful folds were completed.")

        self.models = models
        self.vals_x = vals_x
        self.vals_y = vals_y

    def get_validation_features(self) -> List[pd.DataFrame]:
        """
        Get the list of validation feature DataFrames from cross-validation.

        Returns
        -------
        List[pd.DataFrame]
            List of validation feature DataFrames.
        """
        return self.vals_x

    def get_validation_targets(self) -> List[pd.Series]:
        """
        Get the list of validation target Series from cross-validation.

        Returns
        -------
        List[pd.Series]
            List of validation target Series.
        """
        return self.vals_y

    def get_cv_models(self) -> List[Any]:
        """
        Get the list of models

        Returns
        -------
        List[pd.Series]
            List of models
        """
        return self.models

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions from the first model.

        Parameters
        ----------
        data : pd.DataFrame
            A pandas DataFrame containing the features for prediction, including an 'id' column.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame with two columns: 'id' and 'loan_status',
            where 'loan_status' is the prediction from the first model.
        """
        if "id" not in data.columns:
            raise ValueError("Data must contain an 'id' column.")
        if not self.models:
            raise ValueError("No models have been trained. Please call perform_cv first.")

        data_without_id = data.drop(columns=["id"])
        model = self.models[0]  # Use only the first model
        if hasattr(model, "predict_proba"):
            predictions = model.predict_proba(data_without_id)[:, 1]
        else:
            predictions = model.predict(data_without_id)
        return pd.DataFrame({"id": data["id"], "loan_status": predictions})

    def predict_cv(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mean predictions from all cross-validated models.

        Parameters
        ----------
        data : pd.DataFrame
            A pandas DataFrame containing the features for prediction, including an 'id' column.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame with two columns: 'id' and 'loan_status',
            where 'loan_status' is the mean prediction from all models.
        """
        if "id" not in data.columns:
            raise ValueError("Data must contain an 'id' column.")
        if not self.models:
            raise ValueError("No models have been trained. Please call perform_cv first.")

        data_without_id = data.drop(columns=["id"])
        predictions_list = []
        for model in self.models:
            if hasattr(model, "predict_proba"):
                predictions = model.predict_proba(data_without_id)[:, 1]
            else:
                predictions = model.predict(data_without_id)
            predictions_list.append(predictions)
        predictions_mean = np.mean(predictions_list, axis=0)
        return pd.DataFrame({"id": data["id"], "loan_status": predictions_mean})
