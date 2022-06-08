# %%
import warnings

warnings.filterwarnings("ignore")
from folktables import (
    ACSDataSource,
    ACSIncome,
    ACSEmployment,
    ACSMobility,
    ACSPublicCoverage,
    ACSTravelTime,
)

import pandas as pd
from collections import defaultdict
from scipy.stats import kstest, wasserstein_distance
import seaborn as sns

sns.set_style("whitegrid")
import numpy as np
import random
import sys
import matplotlib.pyplot as plt

# Scikit-Learn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Specific packages
from xgboost import XGBRegressor, XGBClassifier
import shap
from tqdm import tqdm


# Home made code
import sys

sys.path.append("../")
from fairtools.utils import loop_estimators_fairness, psi, loop_estimators
from ATC_opt import ATC

# Seeding
np.random.seed(0)
random.seed(0)
# %%
# Load data
data_source = ACSDataSource(survey_year="2014", horizon="1-Year", survey="person")
ca_data = data_source.get_data(states=["CA"], download=True)
# %%
states = [
    "MI",
    "TN",
    "CT",
    "OH",
    "NE",
    "IL",
    "FL",
    "OK",
    "PA",
    "KS",
    "IA",
    "KY",
    "NY",
    "LA",
    "TX",
    "UT",
]

nooo = [
    "OR",
    "ME",
    "NJ",
    "ID",
    "DE",
    "MN",
    "WI",
    "CA",
    "MO",
    "MD",
    "NV",
    "HI",
    "IN",
    "WV",
    "MT",
    "WY",
    "ND",
    "SD",
    "GA",
    "NM",
    "AZ",
    "VA",
    "MA",
    "AA",
    "NC",
    "SC",
    "DC",
    "VT",
    "AR",
    "WA",
    "CO",
    "NH",
    "MS",
    "AK",
    "RI",
    "AL",
    "PR",
]

data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")

# %%
ca_features, ca_labels, ca_group = ACSEmployment.df_to_numpy(ca_data)

## Conver to DF
ca_features = pd.DataFrame(ca_features, columns=ACSEmployment.features)

# %%
# Modeling
# model = XGBClassifier(verbosity=0, silent=True, use_label_encoder=False, njobs=1)
model = LogisticRegression()
# Train on CA data
X = ca_features.copy()
X["group"] = ca_group
preds_ca = cross_val_predict(model, X, ca_labels, cv=3, method="predict_proba")[:, 1]
model.fit(X, ca_labels)

# Fairness in training data
white_tpr = np.mean(preds_ca[(ca_labels == 1) & (ca_group == 1)])
black_tpr = np.mean(preds_ca[(ca_labels == 1) & (ca_group == 2)])
eof_tr = white_tpr - black_tpr

# xAI
# explainer = shap.Explainer(model)
explainer = shap.LinearExplainer(model, X, feature_dependence="correlation_dependent")
shap_ca = explainer(X)
shap_ca = pd.DataFrame(shap_ca.values, columns=X.columns)

# %%
# Other states

data_source = ACSDataSource(survey_year="2016", horizon="1-Year", survey="person")
results = defaultdict()
for state in tqdm(states):
    mi_data = data_source.get_data(states=[state], download=True)
    mi_features, mi_labels, mi_group = ACSEmployment.df_to_numpy(mi_data)
    mi_features = pd.DataFrame(mi_features, columns=ACSEmployment.features)
    X_te = mi_features.copy()
    X_te["group"] = mi_group
    # Test on MI data
    preds_mi = model.predict_proba(X_te)[:, 1]

    # Shap
    shap_mi = explainer(X_te)
    shap_mi = pd.DataFrame(shap_mi.values, columns=X_te.columns)

    # DP
    ks = kstest(preds_mi[mi_group == 1], preds_mi[mi_group == 2]).statistic

    # EOF
    white_tpr = np.mean(preds_mi[(mi_labels == 1) & (mi_group == 1)])
    black_tpr = np.mean(preds_mi[(mi_labels == 1) & (mi_group == 2)])
    eof_mi = white_tpr - black_tpr

    # Shap
    shap_diff = kstest(
        shap_mi["group"][mi_group == 1], shap_mi["group"][mi_group == 2]
    ).statistic
    results[state] = [ks, eof_mi, shap_diff]
# %%

res = pd.DataFrame(results).T
# res = pd.DataFrame(StandardScaler().fit_transform(res))
res.columns = ["DP", "EOF", "SHAP"]
res["SHAP"] = res["SHAP"]
res.sort_values(by="SHAP", ascending=False).plot()
# %%

res.describe()
# %%
