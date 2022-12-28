from tools.explanationShift import ExplanationShiftDetector


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
        model=LinearRegression(), gmodel=LogisticRegression(), masker=True
    )
    esd.fit_model(XX, y)
    ex = esd.get_explanations(XX)
    assert all([a == b for a, b in zip(ex.columns, XX.columns)])


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
        model=LinearRegression(), gmodel=LogisticRegression(), masker=True
    )
    esd.fit_model(X, y)
    ex = esd.get_explanations(X)
    assert not np.any(np.isnan(ex))


def test_get_coefs_linear():
    """
    Check that the coefficients are returned correctly for the linear regression.
    """
    esd = ExplanationShiftDetector(
        model=LinearRegression(), gmodel=LogisticRegression(), masker=True
    )
    esd.fit(X, y, X_ood)
    coefs = esd.get_linear_coefs()
    # Assert shape
    assert coefs.shape[1] == X.shape[1]
    # Assert that there is non NaNs
    assert not np.any(np.isnan(coefs))
    # Check when we call the full methods
    coefs = esd.get_coefs()
    # Assert shape
    assert coefs.shape[1] == X.shape[1]
    # Assert that there is non NaNs
    assert not np.any(np.isnan(coefs))


def test_get_coefs_pipeline():
    """
    Check that the coefficients are returned correctly for the linear regression pipeline.
    TODO : add a test for the case of a pipeline for F.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    esd = ExplanationShiftDetector(
        model=LinearRegression(),
        gmodel=Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression())]),
        masker=True,
    )
    esd.fit(X_tr, y_tr, X_ood)
    coefs = esd.get_coefs()
    # Assert shape
    assert coefs.shape[1] == X.shape[1]
    # Assert that there is non NaNs
    assert not np.any(np.isnan(coefs))


def test_get_model_types():
    """
    Check that the model types are returned correctly.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    esd = ExplanationShiftDetector(
        model=LinearRegression(), gmodel=LogisticRegression(), masker=X
    )
    assert esd.get_gmodel_type(), esd.get_model_type() == ("linear", "linear")
    # Case of pipeline
    esd = ExplanationShiftDetector(
        model=Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())]),
        gmodel=Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression())]),
    )
    assert esd.get_gmodel_type(), esd.get_model_type() == ("linear", "linear")


def test_spaces():
    """
    Check that the spaces are returned correctly.
    """

    esd = ExplanationShiftDetector(
        model=LinearRegression(),
        gmodel=LogisticRegression(),
        space="input",
        masker=True,
    )
    esd.fit(X_tr, y_tr, X_ood)
    # Check if returns input space
    # assert esd.get_explanations(X) == X
    np.testing.assert_array_equal(esd.get_explanations(X), X)
    # Check if returns output space
    esd = ExplanationShiftDetector(
        model=LogisticRegression(),
        gmodel=LogisticRegression(),
        space="prediction",
        masker=True,
    )
    esd.fit(X, y, X_ood)
    np.testing.assert_array_equal(
        esd.get_explanations(X),
        # pd.DataFrame(data=esd.model.predict_proba(X)[:, 1], columns=["preds"]),
        pd.DataFrame(data=esd.model.predict(X), columns=["preds"]),
    )

    # Check if returns exp space
    esd = ExplanationShiftDetector(
        model=LinearRegression(),
        gmodel=LogisticRegression(),
        space="explanation",
        masker=True,
    )
    esd.fit(X, y, X_ood)

    np.testing.assert_array_equal(esd.get_explanations(X).shape, X.shape)


def test_tree_shap():
    """
    Check that the shap values are returned correctly for the tree models.
    """
    esd = ExplanationShiftDetector(
        model=XGBRegressor(), gmodel=LogisticRegression(), masker=True
    )
    esd.fit(X, y, X_ood)
    shap_values = esd.get_explanations(X)
    # Assert shape
    assert shap_values.shape[1] == X.shape[1]
    # Assert that there is non NaNs
    assert not np.any(np.isnan(shap_values))

    esd = ExplanationShiftDetector(model=XGBRegressor(), gmodel=LogisticRegression())
    esd.fit(X, y, X_ood)
    shap_values2 = esd.get_explanations(X)
    # Assert shape
    assert shap_values2.shape[1] == X.shape[1]
    # Assert that there is non NaNs
    assert not np.any(np.isnan(shap_values2))
    # Assert that the shap values are different depending on the masker
    assert shap_values2.sum(axis=1).sum(axis=0) != shap_values.sum(axis=1).sum(axis=0)
