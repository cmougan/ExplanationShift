from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import pandas as pd
import shap


class ShapEstimator(BaseEstimator, ClassifierMixin):
    """
    A ShapValues estimator based on tree explainer.
    Returns the explanations of the data provided self.predict(X)

    Example:

    import xgboost
    from sklearn.model_selection import cross_val_predict
    X, y = shap.datasets.boston()
    se = ShapEstimator(model=xgboost.XGBRegressor())
    shap_pred = cross_val_predict(se, X, y, cv=3)
    """

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        self.model.fit(self.X_, self.y_)
        return self

    def predict(self, X, dataframe: bool = False):

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        check_array(X)

        explainer = shap.Explainer(self.model)
        shap_values = explainer(X).values
        if dataframe:
            shap_values = pd.DataFrame(shap_values, columns=X.columns)
            shap_values = shap_values.add_suffix("_shap")

        return shap_values
