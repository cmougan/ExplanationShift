# %%
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from skshift import ExplanationShiftDetector
from xgboost import XGBClassifier
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score
import numpy as np
import shap
import pandas as pd

# Create train, hold and test ID data
X, y = make_blobs(
    n_samples=2000,
    centers=2,
    n_features=5,
    random_state=0,
    cluster_std=1.0,
    center_box=(1.0, 3.0),
)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
X_hold, y_hold = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)

# Create OOD data
X_ood = X + 1
X_ood_te = X_te + 1

# Concatenate Distributions
y_te = np.zeros_like(y_te)
y_ood_te = np.ones_like(y_te)
X_new = np.concatenate([X_te, X_ood_te])
y_new = np.concatenate([y_te, y_ood_te])
# %%
# Option 1: fit the detector when there is a trained model
model = Lasso().fit(X_tr, y_tr)

detector = ExplanationShiftDetector(
    model=model, gmodel=LogisticRegression(), masker=X_te
)

detector.fit_detector(X_te, X_ood)
roc_auc_score(y_new, detector.predict_proba(X_new)[:, 1])
# 0.84

# %%
# Calculate Shap Values of Model
# NDCG
## Shap values in Train
explainer = shap.Explainer(model)
shap_values_tr = explainer(X_tr)
shap_df_tr = pd.DataFrame(shap_values_tr.values)

## Shap values in OOD
explainer = shap.Explainer(model)
shap_values_ood = explainer(X_ood)
shap_df_ood = pd.DataFrame(shap_values_ood.values)

ndcg = ndcg_score(
    np.asarray([shap_df_tr.columns.values]), np.asarray([shap_df_ood.columns.values])
)

res.append([rho, input_ks, classifierDrift, output_ks, wass, unc, esd, ndcg])
