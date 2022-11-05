from tools.xaiUtils import ExplanationShiftDetector


from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np


X, y = make_blobs(n_samples=100, centers=2, n_features=5, random_state=0)
X = pd.DataFrame(X, columns=["a", "b", "c", "d", "e"])


def test_return_shapDF():
    """
    If X is dataframe, return shap values as dataframe.
    """

    esd = ExplanationShiftDetector(
        model=LinearRegression(), gmodel=LogisticRegression()
    )
    esd.fit_model(X, y)
    ex = esd.get_explanations(X)
    assert all([a == b for a, b in zip(ex.columns, X.columns)])


def check_supported_models():
    # dont think this works
    """
    Check that models are supported.
    """

    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier, XGBRegressor

    for model in [XGBRegressor(), LogisticRegression()]:
        for gmodel in [XGBClassifier(), LogisticRegression()]:
            assert (
                type(ExplanationShiftDetector(model=model, gmodel=gmodel))
                is tools.xaiUtils.ExplanationShiftDetector
            )


def check_not_supported_models():
    # dont think this works
    """
    Check that models are not supported.
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    with pytest.raises(ValueError):
        ExplanationShiftDetector(model=jjj, gmodel=LogisticRegression())


def test_doc_examples():
    """
    Check that doc examples work.
    """
    XX, yy = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
    XX_tr, XX_te, yy_tr, yy_te = train_test_split(XX, yy, test_size=0.5, random_state=0)
    XX_ood, yy_ood = make_blobs(n_samples=1000, centers=1, n_features=5, random_state=0)

    detector = ExplanationShiftDetector(
        model=XGBRegressor(), gmodel=LogisticRegression()
    )
    assert np.round(detector.get_auc(XX_tr, yy_tr, XX_ood), decimals=2) == 0.77

    assert np.round(detector.get_auc(XX_tr, yy_tr, XX_te), decimals=2) == 0.53
