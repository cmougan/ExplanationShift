from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split
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
    def __init__(self, model, gmodel):
        self.model = model
        self.gmodel = gmodel
        self.explainer = None

        # Supported Tree Models
        self.supported_tree_models = ["XGBClassifier",'XGBRegressor']
        self.supported_linear_models = ['LogisticRegression','LinearRegression','Ridge','Lasso']

        if self.model.__class__.__name__ not in self.supported_tree_models:
            raise ValueError(
                "Model not supported. Supported models are: {}".format(
                    self.supported_tree_models
                )
            )
        if self.gmodel.__class__.__name__ != "LogisticRegression":
            raise ValueError("gmodel must be a LogisticRegression model")
    
    def fit_model(self, X, y):
        self.model.fit(X, y)
    
    def fit_explanation_shift(self, X, y):
        self.gmodel.fit(X, y)
    
    def get_explanations(self, X):
        # Determine the type of SHAP explainer to use
        if self.model.__class__.__name__ == self.supported_tree_models:
            self.explainer = shap.Explainer(self.model)
        elif self.model.__class__.__name__ == self.supported_linear_models:
            self.explainer = shap.LinearExplainer(self.model, X, feature_dependence="correlation_dependent")
        
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
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, random_state=0, test_size=0.5
        )
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
        X_shap = self.get_all_explanations(X, y, X_ood)
        X_shap_tr, X_shap_te, y_shap_tr, y_shap_te = train_test_split(
            X_shap.drop(columns="label"), X_shap["label"], random_state=0, test_size=0.5
        )
        self.fit_explanation_space(X_shap_tr, y_shap_tr)
        return roc_auc_score(y_shap_te, self.gmodel.predict_proba(X_shap_te)[:, 1])
        