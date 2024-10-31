from src.loans.predict_loans import Evaluator
import pytest
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score


@pytest.fixture
def evaluator():
    return Evaluator()

# Evaluator Tests

def test_evaluator_init_default():
    evaluator = Evaluator()
    assert evaluator.metric_function is not None
    assert evaluator.metric_function == roc_auc_score


def test_evaluator_init_custom():
    def custom_metric(y_true, y_pred):
        return np.mean(y_true == y_pred)
    evaluator = Evaluator(metric_function=custom_metric)
    assert evaluator.metric_function == custom_metric


def test_evaluator_evaluate():
    evaluator = Evaluator()
    y_true = [0, 1, 1, 0]
    y_pred = [0.1, 0.9, 0.8, 0.2]
    score = evaluator.evaluate(y_true, y_pred)
    expected_score = roc_auc_score(y_true, y_pred)
    assert score == expected_score