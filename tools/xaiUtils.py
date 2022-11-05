from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import roc_auc_score
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


class ExplanationShiftDetector(BaseEstimator, ClassifierMixin):
    """
    Given a model, and two datasets (train,test), we want to know if the behaviour of the model is different bt train and test.

    1. Fit the model on train
    2. Get the explanations of the model on train and test.
    3. Fit a classifier (gmodel) on the explanations of train and test, to predict to which distribution it belongs.
    4. Return the AUC.

    Example
    -------
    >>> import pandas as pd
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import make_blobs
    >>> from tools.xaiUtils import ExplanationShiftDetector
    >>> from xgboost import XGBRegressor
    >>> from sklearn.linear_model import LogisticRegression

    >>> X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
    >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
    >>> X_ood,y_ood = make_blobs(n_samples=1000, centers=1, n_features=5, random_state=0)

    >>> detector = ExplanationShiftDetector(model=XGBRegressor(),gmodel=LogisticRegression())
    >>> detector.get_auc(X_tr, y_tr, X_ood)
    # 0.76
    >>> detector.get_auc(X_tr, y_tr, X_te)
    # 0.52
    """

    def __init__(self, model, gmodel):
        self.model = model
        self.gmodel = gmodel
        self.explainer = None

        # Supported F Models
        self.supported_tree_models = ["XGBClassifier", "XGBRegressor"]
        self.supported_linear_models = [
            "LogisticRegression",
            "LinearRegression",
            "Ridge",
            "Lasso",
        ]
        self.supported_models = (
            self.supported_tree_models + self.supported_linear_models
        )
        # Supported detectors
        self.supported_detectors = [
            "LogisticRegression",
            "RandomForestClassifier",
            "XGBClassifier",
        ]

        # Check if models are supported
        if self.model.__class__.__name__ not in self.supported_models:
            raise ValueError(
                "Model not supported. Supported models are: {} got {}".format(
                    self.supported_models, self.model.__class__.__name__
                )
            )
        if self.gmodel.__class__.__name__ not in self.supported_detectors:
            raise ValueError(
                "gmodel not supported. Supported models are: {} got {}".format(
                    self.supported_detectors, self.gmodel.__class__.__name__
                )
            )

    def fit_model(self, X, y):
        self.model.fit(X, y)

    def fit_explanation_shift(self, X, y):
        self.gmodel.fit(X, y)

    def get_explanations(self, X):
        # Determine the type of SHAP explainer to use
        if self.model.__class__.__name__ in self.supported_tree_models:
            self.explainer = shap.Explainer(self.model)
        elif self.model.__class__.__name__ in self.supported_linear_models:
            self.explainer = shap.LinearExplainer(
                self.model, X, feature_dependence="correlation_dependent"
            )
        else:
            raise ValueError(
                "Model not supported. Supported models are: {}, got {}".format(
                    self.supported_models, self.model.__class__.__name__
                )
            )

        shap_values = self.explainer(X)
        # Name columns
        if isinstance(X, pd.DataFrame):
            columns_name = X.columns
        else:
            columns_name = ["Shap%d" % (i + 1) for i in range(X.shape[1])]

        exp = pd.DataFrame(
            data=shap_values.values,
            columns=columns_name,
        )
        return exp

    def get_iid_explanations(self, X, y):
        # Does too many things, getting and setting, not good
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0, test_size=0.5)
        self.fit_model(X_tr, y_tr)
        self.explainer = shap.Explainer(self.model)
        return self.get_explanations(X_te)

    def get_all_explanations(self, X, y, X_ood):
        X_iid = self.get_iid_explanations(X, y)
        X_ood = self.get_explanations(X_ood)
        X_iid["label"] = 0
        X_ood["label"] = 1
        X = pd.concat([X_iid, X_ood])
        return X

    def get_auc(self, X, y, X_ood):
        """
        Determine if the model is behaving differently on the two datasets.
        Receives train and data to test
        Returns AUC of classification bt train and test

        Steps:

        """

        X_shap = self.get_all_explanations(X, y, X_ood)
        X_shap_tr, X_shap_te, y_shap_tr, y_shap_te = train_test_split(
            X_shap.drop(columns="label"), X_shap["label"], random_state=0, test_size=0.5
        )
        self.fit_explanation_shift(X_shap_tr, y_shap_tr)
        return roc_auc_score(y_shap_te, self.gmodel.predict_proba(X_shap_te)[:, 1])
