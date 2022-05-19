# %%
from folktables import ACSDataSource, ACSIncome
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
import pandas as pd
from collections import defaultdict
from xgboost import XGBRegressor, XGBClassifier
from scipy.stats import kstest
import shap
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import sys

sys.path.append("../")
from fairtools.xaiUtils import ShapEstimator
import random

random.seed(0)
# %%
# Load data
data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
ca_data = data_source.get_data(states=["CA"], download=True)
mi_data = data_source.get_data(states=["MI"], download=True)
ca_features, ca_labels, ca_group = ACSIncome.df_to_numpy(ca_data)
mi_features, mi_labels, mi_group = ACSIncome.df_to_numpy(mi_data)
## Conver to DF
ca_features = pd.DataFrame(ca_features, columns=ACSIncome.features)
mi_features = pd.DataFrame(mi_features, columns=ACSIncome.features)

# %%
# Modeling
model = XGBClassifier()

# Train on CA data
preds_ca = cross_val_predict(model, ca_features, ca_labels, cv=3)
model.fit(ca_features, ca_labels)

# Test on MI data
preds_mi = model.predict(mi_features)

# %%
##Fairness
white_tpr = np.mean(preds_ca[(ca_labels == 1) & (ca_group == 1)])
black_tpr = np.mean(preds_ca[(ca_labels == 1) & (ca_group == 2)])
print("Train EO", white_tpr - black_tpr)

white_tpr = np.mean(preds_mi[(mi_labels == 1) & (mi_group == 1)])
black_tpr = np.mean(preds_mi[(mi_labels == 1) & (mi_group == 2)])
print("Test EO", white_tpr - black_tpr)

# %%
## Model performance
print(roc_auc_score(preds_ca, ca_labels))
print(roc_auc_score(preds_mi, mi_labels))

# %%
# Input KS
for feat in ca_features.columns:
    pval = kstest(ca_features[feat], mi_features[feat]).pvalue
    if pval < 0.1:
        print(feat, " is distinct ", pval)
    else:
        print(feat, " is equivalent ", pval)
# %%
# %%
# Explainability
explainer = shap.Explainer(model)
shap_values = explainer(ca_features)
ca_shap = pd.DataFrame(shap_values.values, columns=ca_features.columns)
shap_values = explainer(mi_features)
mi_shap = pd.DataFrame(shap_values.values, columns=ca_features.columns)
# %%
# SHAP KS
for feat in ca_features.columns:
    pval = kstest(ca_shap[feat], mi_shap[feat]).pvalue
    if pval < 0.1:
        print(feat, " is distinct ", pval)
    else:
        print(feat, " is equivalent ", pval)
# %%
## Shap Estimator on CA and MI
se = ShapEstimator(model=XGBRegressor())
shap_pred_ca = cross_val_predict(se, ca_features, ca_labels, cv=3)
shap_pred_ca = pd.DataFrame(shap_pred_ca, columns=ca_features.columns)
shap_pred_ca = shap_pred_ca.add_suffix("_shap")

se.fit(ca_features, ca_labels)
error_ca = ca_labels == preds_ca
# %%
# Estimators for the loop
estimators = defaultdict()
estimators["Linear"] = Pipeline(
    [("scaler", StandardScaler()), ("model", LogisticRegression())]
)
estimators["RandomForest"] = RandomForestClassifier(random_state=0)
estimators["XGBoost"] = XGBClassifier(random_state=0)
estimators["MLP"] = MLPClassifier(random_state=0)
# %%
# Loop over different G estimators
for estimator in estimators:
    print(estimator)
    clf = estimators[estimator]

    preds_ca_shap = cross_val_predict(
        clf, shap_pred_ca, error_ca, cv=3, method="predict_proba"
    )[:, 1]
    clf.fit(shap_pred_ca, error_ca)

    shap_pred_mi = se.predict(mi_features)
    shap_pred_mi = pd.DataFrame(shap_pred_mi, columns=ca_features.columns)
    shap_pred_mi = shap_pred_mi.add_suffix("_shap")
    error_mi = mi_labels == preds_mi
    preds_mi_shap = clf.predict_proba(shap_pred_mi)[:, 1]

    ## Only SHAP
    print("Only Shap")
    print(roc_auc_score(error_ca, preds_ca_shap))
    print(roc_auc_score(error_mi, preds_mi_shap))
    ## Only data
    print("Only Data")
    preds_ca_shap = cross_val_predict(
        clf, ca_features, error_ca, cv=3, method="predict_proba"
    )[:, 1]
    clf.fit(ca_features, error_ca)
    preds_mi_shap = clf.predict_proba(mi_features)[:, 1]
    print(roc_auc_score(error_ca, preds_ca_shap))
    print(roc_auc_score(error_mi, preds_mi_shap))

    ## SHAP + Data
    print("Shap + Data")
    ca_full = pd.concat([shap_pred_ca, ca_features], axis=1)
    mi_full = pd.concat([shap_pred_mi, mi_features], axis=1)

    preds_ca_shap = cross_val_predict(
        clf, ca_full, error_ca, cv=3, method="predict_proba"
    )[:, 1]
    clf.fit(ca_full, error_ca)
    preds_mi_shap = clf.predict_proba(mi_full)[:, 1]
    print(roc_auc_score(error_ca, preds_ca_shap))
    print(roc_auc_score(error_mi, preds_mi_shap))

# %%
# Original Error
