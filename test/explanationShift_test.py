from tools.xaiUtils import ExplanationShiftDetector


from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np


X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
X_ood, y_ood = make_blobs(n_samples=1000, centers=1, n_features=5, random_state=0)


def test_return_shapDF():
    """
    If X is dataframe, return shap values as dataframe.
    """
    XX = pd.DataFrame(X, columns=["a", "b", "c", "d", "e"])

    esd = ExplanationShiftDetector(
        model=LinearRegression(), gmodel=LogisticRegression()
    )
    esd.fit_model(XX, y)
    ex = esd.get_explanations(XX)
    assert all([a == b for a, b in zip(ex.columns, XX.columns)])


def test_supported_models():
    """
    Check that models are supported.
    """
    for model in [XGBRegressor(), LogisticRegression()]:
        for gmodel in [XGBClassifier(), LogisticRegression()]:
            assert (
                type(ExplanationShiftDetector(model=model, gmodel=gmodel))
                is ExplanationShiftDetector
            )


def test_not_supported_models():
    """
    Check that models are not supported.
    """

    from sklearn.neural_network import MLPClassifier
    import pytest

    with pytest.raises(ValueError):
        ExplanationShiftDetector(model=MLPClassifier(), gmodel=LogisticRegression())
    with pytest.raises(ValueError):
        ExplanationShiftDetector(model=LinearRegression(), gmodel=MLPClassifier())


def test_doc_examples():
    """
    Check that doc examples work.
    """

    detector = ExplanationShiftDetector(
        model=XGBRegressor(random_state=0), gmodel=LogisticRegression()
    )
    # On OOD
    detector.fit(X_source=X_tr, y_source=y_tr, X_ood=X_ood)
    assert np.round(detector.get_auc_val(), decimals=2) == 0.77
    # On test
    detector.fit(X_source=X_tr, y_source=y_tr, X_ood=X_te)
    assert np.round(detector.get_auc_val(), decimals=2) == 0.53


def test_no_nan():
    """
    Check that no NaNs are present in the shap values.
    """
    esd = ExplanationShiftDetector(
        model=LinearRegression(), gmodel=LogisticRegression()
    )
    esd.fit_model(X, y)
    ex = esd.get_explanations(X)
    assert not np.any(np.isnan(ex))


def test_get_coefs():
    """
    Check that the coefficients are returned correctly.
    """
    esd = ExplanationShiftDetector(
        model=LinearRegression(), gmodel=LogisticRegression()
    )
    esd.fit(X, y, X_ood)
    coefs = esd.get_coefs()
    # Assert shape
    assert coefs.shape[1] == X.shape[1]
    # Assert that there is non NaNs
    assert not np.any(np.isnan(coefs))
