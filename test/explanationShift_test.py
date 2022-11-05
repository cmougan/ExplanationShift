from tools.xaiUtils import ExplanationShiftDetector

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.datasets import make_blobs
import pandas as pd


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
