from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import shap


class ExplanationShiftDetector(BaseEstimator, ClassifierMixin):
    """
    Given a model, and two datasets (source,test), we want to know if the behaviour of the model is different bt train and test.
    We can do this by computing the shap values of the model on the two datasets, and then train a classifier to distinguish between the two datasets.

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
    >>> detector.fit(X_tr, y_tr, X_ood)
    >>> detector.get_auc_val()
    # 0.76
    >>> detector.fit(X_tr, y_tr, X_te)
    >>> detector.get_auc_val()
    # 0.5
    """

    def __init__(
        self,
        model,
        gmodel,
        space: str = "explanation",
        algorithm: str = "auto",
        masker: bool = False,
    ):
        """
        Parameters
        ----------
        model : sklearn model
            Model to be used to compute the shap values.
        gmodel : sklearn model
            Model to be used to distinguish between the two datasets.
        space : str, optional
            Space in which the gmodel is learned. Can be 'explanation' or 'input' or 'predictions'. Default is 'explanation'.

        algorithm : "auto", "permutation", "partition", "tree", or "linear"
                The algorithm used to estimate the Shapley values. There are many different algorithms that
                can be used to estimate the Shapley values (and the related value for constrained games), each
                of these algorithms have various tradeoffs and are preferrable in different situations. By
                default the "auto" options attempts to make the best choice given the passed model and masker,
                but this choice can always be overriden by passing the name of a specific algorithm. The type of
                algorithm used will determine what type of subclass object is returned by this constructor, and
                you can also build those subclasses directly if you prefer or need more fine grained control over
                their options.

        masker : bool,
                The masker object is used to define the background distribution over which the Shapley values
                are estimated. Is a boolean that indicates if the masker should be used or not. If True, the masker is used.
                If False, the masker is not used. The background distribution is the same distribution as we are calculating the Shapley values.
                TODO Decide which masker distribution is better to use, options are: train data, hold out data, ood data
        """

        self.model = model
        self.gmodel = gmodel
        self.explainer = None
        self.space = space
        self.algorithm = algorithm
        self.masker = masker

        # Check if space is supported
        if self.space not in ["explanation", "input", "prediction"]:
            raise ValueError(
                "space not supported. Supported spaces are: {} got {}".format(
                    ["explanation", "input", "prediction"], self.space
                )
            )

    def get_gmodel_type(self):
        if self.gmodel.__class__.__name__ == "Pipeline":
            return self.gmodel.steps[-1][1].__class__.__name__
        else:
            return self.gmodel.__class__.__name__

    def get_model_type(self):
        if self.model.__class__.__name__ == "Pipeline":
            return self.model.steps[-1][1].__class__.__name__
        else:
            return self.model.__class__.__name__

    def fit(self, X_source, y_source, X_ood):

        # Check that X and y have correct shape
        check_X_y(X_source, y_source)
        self.X_ood = X_ood

        self.X_tr, self.X_val, self.y_tr, self.y_val = train_test_split(
            X_source, y_source, random_state=0, test_size=0.5
        )

        # Fit model F
        self.fit_model(self.X_tr, self.y_tr)

        # Get explanations
        self.S_val = self.get_explanations(self.X_val)
        self.S_ood = self.get_explanations(self.X_ood)

        # Create dataset for  explanation shift detector
        self.S_val["label"] = False
        self.S_ood["label"] = True

        self.S = pd.concat([self.S_val, self.S_ood])

        (
            self.X_shap_tr,
            self.X_shap_te,
            self.y_shap_tr,
            self.y_shap_te,
        ) = train_test_split(
            self.S.drop(columns="label"), self.S["label"], random_state=0, test_size=0.5
        )
        self.fit_explanation_shift(self.X_shap_tr, self.y_shap_tr)

        return self

    def predict(self, X):
        return self.gmodel.predict(self.get_explanations(X))

    def predict_proba(self, X):
        return self.gmodel.predict_proba(self.get_explanations(X))

    def explanation_predict(self, X):
        return self.gmodel.predict(X)

    def explanation_predict_proba(self, X):
        return self.gmodel.predict_proba(X)

    def fit_model(self, X, y):
        self.model.fit(X, y)

    def fit_explanation_shift(self, X, y):
        self.gmodel.fit(X, y)

    def get_explanations(self, X, data_masker=None):
        if data_masker == None:
            data_masker = self.X_tr
        else:
            data_masker = data_masker

        if self.space == "explanation":
            if self.masker:
                self.explainer = shap.Explainer(
                    self.model, algorithm=self.algorithm, masker=data_masker
                )
            else:
                self.explainer = shap.Explainer(self.model, algorithm=self.algorithm)

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
        if self.space == "input":
            shap_values = X
            # Name columns
            if isinstance(X, pd.DataFrame):
                exp = X
            else:
                columns_name = ["Shap%d" % (i + 1) for i in range(X.shape[1])]

                exp = pd.DataFrame(
                    data=shap_values,
                    columns=columns_name,
                )
        if self.space == "prediction":
            # shap_values = self.model.predict_proba(X)[:, 1]
            shap_values = self.model.predict(X)

            # Name columns
            exp = pd.DataFrame(
                data=shap_values,
                columns=["preds"],
            )

        return exp

    def get_auc_val(self):
        """
        Returns the AUC of the explanation shift detector on the validation set of the explanation space
        Example
        -------
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import make_blobs
        from tools.xaiUtils import ExplanationShiftDetector
        from xgboost import XGBRegressor
        from sklearn.linear_model import LogisticRegression

        # Create data
        X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
        X_ood,y_ood = make_blobs(n_samples=1000, centers=1, n_features=5, random_state=0)

        detector = ExplanationShiftDetector(model=XGBRegressor(),gmodel=LogisticRegression())
        detector.fit(X_tr, y_tr, X_ood)
        detector.get_auc_val()
        # 0.76

        """
        return roc_auc_score(
            self.y_shap_te, self.explanation_predict_proba(self.X_shap_te)[:, 1]
        )

    def get_coefs(self):
        if self.gmodel.__class__.__name__ == "Pipeline":
            if "sklearn.linear_model" in self.gmodel.steps[-1][1].__module__:
                return self.gmodel.steps[-1][1].coef_
            else:
                raise ValueError(
                    "Coefficients can not be calculated. Supported models are linear: sklearn.linear_model, got {}".format(
                        self.gmodel.steps[-1][1].__module__
                    )
                )
        else:
            return self.get_linear_coefs()

    def get_linear_coefs(self):
        if "sklearn.linear_model" in self.gmodel.__module__:
            return self.gmodel.coef_
        else:
            raise ValueError(
                "Coefficients can not be calculated. Supported models are linear: sklearn.linear_model, got {}".format(
                    self.gmodel.steps[-1][1].__module__
                )
            )

    def explain_detector(self):
        exp = shap.Explainer(self.model)
        shap_values = exp.shap_values(self.S_ood.drop(columns="label"))
        shap.summary_plot(shap_values, self.S_ood.drop(columns="label"), plot_type="bar", show=False)
    

