# %%
from folktables import ACSDataSource, ACSIncome
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
import pandas as pd
from xgboost import XGBClassifier
from scipy.stats import kstest
import shap
import numpy as np
import sys

sys.path.append("../")
import random

random.seed(0)
# %%
# Load data
data_source = ACSDataSource(survey_year="2014", horizon="1-Year", survey="person")
ca_data = data_source.get_data(states=["PR"], download=True)
ca_features, ca_labels, ca_group = ACSIncome.df_to_numpy(ca_data)

# OOD
data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
mi_data = data_source.get_data(states=["CA"], download=True)
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
print('Train')
print(roc_auc_score(preds_ca, ca_labels))
print('Test')
print(roc_auc_score(preds_mi, mi_labels))

# %%
# Input KS
input = 0
for feat in ca_features.columns:
    pval = kstest(ca_features[feat], mi_features[feat]).pvalue
    if pval < 0.1:
        input = input + 1
        print(feat, " is distinct ", pval)
    else:
        print(feat, " is equivalent ", pval)
print(input, "Distinct Features out of ", len(ca_features.columns))
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
sh = 0
for feat in ca_features.columns:
    pval = kstest(ca_shap[feat], mi_shap[feat]).pvalue
    if pval < 0.1:
        sh = sh + 1
        print(feat, " is distinct ", pval)
    else:
        print(feat, " is equivalent ", pval)
print(sh, "Distinct Explanations out of ", len(ca_features.columns))
