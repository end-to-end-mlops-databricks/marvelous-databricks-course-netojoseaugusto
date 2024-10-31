from src.loans.predict_loans import Loans, Evaluator
import pytest
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

@pytest.fixture
def sample_configs():
    return {
        'model_params': {'iterations': 10, 'learning_rate': 0.1},
        'categorical_variables': ['cat_feature'],
        'model_verbose': False
    }


@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'cat_feature': ['A', 'B', 'A', 'B', 'A']
    })
    Y = pd.Series([0, 1, 0, 1, 0])
    return X, Y

@pytest.fixture
def loans(sample_configs, evaluator):
    return Loans(configs=sample_configs, evaluator=evaluator)


@pytest.fixture
def evaluator():
    return Evaluator()


def test_loans_init(sample_configs, evaluator):
    loans = Loans(configs=sample_configs, evaluator=evaluator)
    assert loans.configs == sample_configs
    assert loans.model_class == CatBoostClassifier
    assert loans.model_params == {'iterations': 10, 'learning_rate': 0.1}
    assert loans.categorical_variables == ['cat_feature']
    assert loans.verbose is False
    assert loans.evaluator == evaluator
    assert loans.models == []
    assert loans.vals_x == []
    assert loans.vals_y == []


def test_loans_fit_model(sample_data, loans):
    X, Y = sample_data
    train_x = X.iloc[:3]
    train_y = Y.iloc[:3]
    val_x = X.iloc[3:]
    val_y = Y.iloc[3:]
    model = loans.fit_model(train_x, train_y, val_x, val_y)
    assert model is not None
    assert isinstance(model, CatBoostClassifier)


def test_loans_evaluate_model(sample_data, loans):
    X, Y = sample_data
    train_x = X.iloc[:3]
    train_y = Y.iloc[:3]
    val_x = X.iloc[3:]
    val_y = Y.iloc[3:]
    model = loans.fit_model(train_x, train_y, val_x, val_y)
    score = loans.evaluate_model(model, val_x, val_y)
    assert isinstance(score, float)


def test_loans_generate_train_val_indices_for_cv(sample_data, loans):
    X, Y = sample_data
    generator = loans.generate_train_val_indices_for_cv(X, Y, nfolds=2)
    indices = list(generator)
    assert len(indices) == 2
    for train_idx, val_idx in indices:
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(val_idx, np.ndarray)
        assert len(train_idx) > 0
        assert len(val_idx) > 0


def test_loans_perform_cv(sample_data, loans):
    X, Y = sample_data
    loans.perform_cv(X, Y, nfolds=2)
    assert len(loans.models) == 2
    assert len(loans.vals_x) == 2
    assert len(loans.vals_y) == 2


def test_loans_get_validation_features(sample_data, loans):
    X, Y = sample_data
    loans.perform_cv(X, Y, nfolds=2)
    vals_x = loans.get_validation_features()
    assert len(vals_x) == 2
    for val_x in vals_x:
        assert isinstance(val_x, pd.DataFrame)


def test_loans_get_validation_targets(sample_data, loans):
    X, Y = sample_data
    loans.perform_cv(X, Y, nfolds=2)
    vals_y = loans.get_validation_targets()
    assert len(vals_y) == 2
    for val_y in vals_y:
        assert isinstance(val_y, pd.Series)


def test_loans_get_cv_models(sample_data, loans):
    X, Y = sample_data
    loans.perform_cv(X, Y, nfolds=2)
    models = loans.get_cv_models()
    assert len(models) == 2
    for model in models:
        assert isinstance(model, CatBoostClassifier)


def test_loans_predict(sample_data, loans):
    X, Y = sample_data
    loans.perform_cv(X, Y, nfolds=2)
    data = X.copy()
    data['id'] = [1, 2, 3, 4, 5]
    predictions = loans.predict(data)
    assert isinstance(predictions, pd.DataFrame)
    assert 'id' in predictions.columns
    assert 'loan_status' in predictions.columns
    assert len(predictions) == len(data)


def test_loans_predict_no_id_column(sample_data, loans):
    X, Y = sample_data
    loans.perform_cv(X, Y, nfolds=2)
    data = X.copy()
    with pytest.raises(ValueError) as excinfo:
        loans.predict(data)
    assert "Data must contain an 'id' column." in str(excinfo.value)


def test_loans_predict_no_models_trained(sample_data, loans):
    X, _ = sample_data
    data = X.copy()
    data['id'] = [1, 2, 3, 4, 5]
    with pytest.raises(ValueError) as excinfo:
        loans.predict(data)
    assert "No models have been trained. Please call perform_cv first." in str(excinfo.value)


def test_loans_predict_cv(sample_data, loans):
    X, Y = sample_data
    loans.perform_cv(X, Y, nfolds=2)
    data = X.copy()
    data['id'] = [1, 2, 3, 4, 5]
    predictions = loans.predict_cv(data)
    assert isinstance(predictions, pd.DataFrame)
    assert 'id' in predictions.columns
    assert 'loan_status' in predictions.columns
    assert len(predictions) == len(data)


def test_loans_predict_cv_no_id_column(sample_data, loans):
    X, Y = sample_data
    loans.perform_cv(X, Y, nfolds=2)
    data = X.copy()
    with pytest.raises(ValueError) as excinfo:
        loans.predict_cv(data)
    assert "Data must contain an 'id' column." in str(excinfo.value)


def test_loans_predict_cv_no_models_trained(sample_data, loans):
    X, _ = sample_data
    data = X.copy()
    data['id'] = [1, 2, 3, 4, 5]
    with pytest.raises(ValueError) as excinfo:
        loans.predict_cv(data)
    assert "No models have been trained. Please call perform_cv first." in str(excinfo.value)