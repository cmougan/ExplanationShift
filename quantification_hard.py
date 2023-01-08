# %%
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})
import pandas as pd
import random

import numpy as np

random.seed(0)
# Scikit Learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tools.explanationShift import ExplanationShiftDetector

plt.style.use("seaborn-whitegrid")
from xgboost import XGBClassifier

from tools.datasets import GetData
from tools.explanationShift import ExplanationShiftDetector

# %%
data = GetData(type="real", datasets="ACSEmployment")
X, y = data.get_state(state="CA", year="2014")

# What is the most important feature?
model = XGBClassifier()
model.fit(X, y)
importances = model.feature_importances_
# %%
# We select the most important feature
most_important = np.argmax(importances)
# Conver feature importance to pandas
importances = pd.DataFrame(importances, index=X.columns, columns=["importance"])
print('Most important feature: "{}"'.format(importances.index[most_important]))
# We sort the data by the most important feature
X["label"] = y
X = X.sort_values(by=X.columns[most_important], ascending=True)
# Split data into first and second half
X_1 = X.iloc[: int(len(X) / 3), :]
X_2 = X.iloc[2 * int(len(X) / 3) :, :]
y_1 = X_1["label"]
y_2 = X_2["label"]
X = X.drop(columns=["label"])
X_1 = X_1.drop(columns=["label"])
X_2 = X_2.drop(columns=["label"])
# Split X_1 into train and val
X_tr, X_val, y_tr, y_val = train_test_split(
    X_1, y_1, test_size=0.5, stratify=y_1, random_state=0
)
# Split X_2 into ood_tr and ood_te
X_ood_tr, X_ood_te, y_ood_tr, y_ood_te = train_test_split(
    X_2, y_2, test_size=0.5, stratify=y_2, random_state=0
)
# Concatenate X_te and X_ood_te
X_val["ood"] = 0
X_ood_te["ood"] = 1
X_hold = pd.concat([X_val, X_ood_te])
y_hold = pd.concat([y_val, y_ood_te])
z_hold = X_hold["ood"]
X_hold = X_hold.drop(columns=["ood"])
X_val = X_val.drop(columns=["ood"])
X_ood_te = X_ood_te.drop(columns=["ood"])

# %%
for space in ["input", "prediction", "explanation"]:
    print("----------------------------------")
    print(space)
    print("----------------------------------")
    detector = ExplanationShiftDetector(
        model=XGBClassifier(max_depth=3, random_state=0),
        gmodel=XGBClassifier(max_depth=3, random_state=0),
        space=space,
    )
    if "label" in X_ood_tr.columns:
        X_ood_tr = X_ood_tr.drop(columns=["label"])

    detector.fit(X_tr, y_tr, X_ood_tr)
    detector.explain_detector()
    # Evaluate the model
    auc_tr = roc_auc_score(y_val, detector.model.predict_proba(X_val)[:, 1])
    print(
        "AUC on test set",
        auc_tr,
    )
    print(
        "AUC on OOD test set",
        roc_auc_score(y_ood_te, detector.model.predict_proba(X_ood_te)[:, 1]),
    )
    print("Auditor", detector.get_auc_val())

    # Two preds
    ## On X_val
    print("VAL")
    aux = X_val.copy()
    aux["real"] = y_val.values
    aux["pred"] = detector.model.predict(X_val)
    aux["pred_proba"] = detector.model.predict_proba(X_val)[:, 1]
    # aux["ood"] = z_val_test.values
    aux["ood_pred_proba"] = detector.predict_proba(X_val)[:, 1]
    # Use the threshold to flag as OOD
    aux["ood_pred"] = detector.predict_proba(X_val)[:, 1] > 0.95
    print("Total flagged as OOD: ", aux[aux["ood_pred"] == 1].shape[0])
    auc_id = roc_auc_score(
        aux[aux["ood_pred"] == 0].real, aux[aux["ood_pred"] == 0].pred_proba.values
    ) - roc_auc_score(
        aux[aux["ood_pred"] == 1].real, aux[aux["ood_pred"] == 1].pred_proba.values
    )

    # On X_ood_te
    print("OOD")
    aux = X_ood_te.copy()
    aux["real"] = y_ood_te.values
    aux["pred"] = detector.model.predict(X_ood_te)
    aux["pred_proba"] = detector.model.predict_proba(X_ood_te)[:, 1]
    # aux["ood"] = z_ood_te_test.values
    # Use the threshold to flag as OOD
    aux["ood_pred"] = detector.predict_proba(X_ood_te)[:, 1] > 0.95
    print("Total flagged as OOD: ", aux[aux["ood_pred"] == 1].shape[0])
    auc_ood = roc_auc_score(
        aux[aux["ood_pred"] == 0].real, aux[aux["ood_pred"] == 0].pred_proba.values
    ) - roc_auc_score(
        aux[aux["ood_pred"] == 1].real, aux[aux["ood_pred"] == 1].pred_proba.values
    )

    aux = X_hold.copy()
    aux["real"] = y_hold.values
    aux["pred"] = detector.model.predict(X_hold)
    aux["pred_proba"] = detector.model.predict_proba(X_hold)[:, 1]
    # aux["ood"] = z_hold_test.values
    aux["ood_pred"] = detector.predict(X_hold)
    # aux["ood_pred_proba"] = detector.predict_proba(X_hold_test)[:, 1]
    print("Total flagged as OOD: ", aux[aux["ood_pred"] == 1].shape[0])
    # auc_ood = roc_auc_score(aux.real, aux.pred_proba.values)

    try:
        decay = auc_id - auc_ood

    except:
        decay = 0
    print("Decay", decay)
    print("AUC ID", auc_id)
    print("AUC OOD", auc_ood)
