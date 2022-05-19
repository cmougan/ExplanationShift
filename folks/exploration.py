# %%
from folktables import (
    ACSDataSource,
    ACSIncome,
    ACSEmployment,
    ACSMobility,
    ACSPublicCoverage,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from scipy.stats import kstest
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np
import random
import shap

random.seed(0)
# %%
# Load TR data
data_source = ACSDataSource(survey_year="2014", horizon="1-Year", survey="person")
data_test = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
ca_data = data_source.get_data(states=["CA"], download=True)
ca_features, ca_labels, ca_group = ACSPublicCoverage.df_to_numpy(ca_data)
## Conver to DF
ca_features = pd.DataFrame(ca_features, columns=ACSPublicCoverage.features)

# %%
# Load TE data
STATES = ["HI", "PR", "MI", "AK", "NY"]
for state in STATES:
    print(state)
    mi_data = data_test.get_data(states=[state], download=True)
    mi_features, mi_labels, mi_group = ACSPublicCoverage.df_to_numpy(mi_data)
    mi_features = pd.DataFrame(mi_features, columns=ACSPublicCoverage.features)
    distinct = 0
    for feat in ca_features.columns:
        pval = kstest(ca_features[feat], mi_features[feat]).pvalue
        if pval < 0.05:
            distinct = distinct + 1
    print("KS: ", distinct)
    # Modeling
    model = GradientBoostingClassifier()

    # Train on CA data
    preds_ca = cross_val_predict(model, ca_features, ca_labels, cv=3)
    model.fit(ca_features, ca_labels)

    # Test on MI data
    preds_mi = model.predict(mi_features)
    # Explainability
    explainer = shap.Explainer(model)
    shap_values = explainer(ca_features)
    ca_shap = pd.DataFrame(shap_values.values, columns=ca_features.columns)
    shap_values = explainer(mi_features)
    mi_shap = pd.DataFrame(shap_values.values, columns=ca_features.columns)

    # SHAP KS
    distinct = 0
    for feat in ca_features.columns:
        pval = kstest(ca_shap[feat], mi_shap[feat]).pvalue
        if pval < 0.1:
            distinct = distinct + 1
    print("SHAP: ", distinct)

    ##Fairness
    white_tpr = np.mean(preds_ca[(ca_labels == 1) & (ca_group == 1)])
    black_tpr = np.mean(preds_ca[(ca_labels == 1) & (ca_group == 2)])
    print("Train EO", white_tpr - black_tpr)

    white_tpr = np.mean(preds_mi[(mi_labels == 1) & (mi_group == 1)])
    black_tpr = np.mean(preds_mi[(mi_labels == 1) & (mi_group == 2)])
    print("Test EO", white_tpr - black_tpr)

    ## Model performance
    print("Train AUC: ", roc_auc_score(preds_ca, ca_labels))
    print("Test AUC: ", roc_auc_score(preds_mi, mi_labels))

# %%
