from tools.xaiUtils import ShapEstimator


def test_return_shape():
    """
    Test that the return shape is the same as the input shape.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_predict
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=100, centers=3, n_features=5, random_state=0)
    se = ShapEstimator(model=GradientBoostingRegressor())
    shap_pred = cross_val_predict(se, X, y, cv=3)
    assert shap_pred.shape == X.shape
