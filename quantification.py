# %%
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})
import pandas as pd
import random

from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix

random.seed(0)
# Scikit Learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tools.explanationShift import ExplanationShiftDetector

plt.style.use("seaborn-whitegrid")
from xgboost import XGBClassifier

from tools.datasets import GetData
from tools.explanationShift import ExplanationShiftDetector

# %%
res = []
for datatype in tqdm(
    ["ACSMobility", "ACSPublicCoverage", "ACSTravelTime", "ACSEmployment", "ACSIncome"]
):
    data = GetData(type="real", datasets=datatype)
    X, y = data.get_state(state="CA", year="2014")
    # Split data into train, val and test
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=0.5, stratify=y_tr, random_state=0
    )

    # OOD Data
    for state in ["NY", "TX", "HI", "MN", "WI", "FL"]:
        X_ood, y_ood = data.get_state(state=state, year="2018")
        X_ood_tr, X_ood_te, y_ood_tr, y_ood_te = train_test_split(
            X_ood, y_ood, test_size=0.5, stratify=y_ood, random_state=0
        )
        # Train data for G
        X_val["OOD"] = 0
        X_ood_tr["OOD"] = 1
        X_hold = pd.concat([X_val, X_ood_tr])
        z_hold_tr = X_hold["OOD"]

        # Drop columns to avoid errors
        X_hold = X_hold.drop(columns=["OOD"])
        X_val = X_val.drop(columns=["OOD"])
        X_ood_tr = X_ood_tr.drop(columns=["OOD"])
        # The real y
        y_hold_tr = pd.concat(
            [
                pd.DataFrame(y_val, columns=["real"]),
                pd.DataFrame(y_ood_tr, columns=["real"]),
            ]
        )
        # Test data for G
        X_te["OOD"] = 0
        X_ood_te["OOD"] = 1
        X_hold_test = pd.concat([X_te, X_ood_te])
        z_hold_test = X_hold_test["OOD"]

        # Drop columns to avoid errors
        X_hold_test = X_hold_test.drop(columns=["OOD"])
        X_te = X_te.drop(columns=["OOD"])
        X_ood_te = X_ood_te.drop(columns=["OOD"])
        # The real y
        y_hold_test = pd.concat(
            [
                pd.DataFrame(y_te, columns=["real"]),
                pd.DataFrame(y_ood_te, columns=["real"]),
            ]
        )
        for space in ["explanation", "input", "prediction"]:
            print("----------------------------------")
            print(space)
            print("----------------------------------")
            # Fit our estimator
            detector = ExplanationShiftDetector(
                model=XGBClassifier(max_depth=3, random_state=0),
                gmodel=Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(random_state=0, penalty="l2")),
                    ]
                ),
                space=space,
                masker=False,
            )
            if "label" in X_ood_tr.columns:
                X_ood_tr = X_ood_tr.drop(columns=["label"])

            detector.fit(X_tr, y_tr, X_ood_tr)
            detector.explain_detector()

            # Does the model decay?
            print("Does the model decay?")
            auc_te = np.round(
                roc_auc_score(y_te, detector.model.predict_proba(X_te)[:, 1]), 3
            )
            auc_ood = np.round(
                roc_auc_score(y_ood_te, detector.model.predict_proba(X_ood_te)[:, 1]), 3
            )
            print("AUC TE", auc_te)
            print("AUC OOD", auc_ood)

            # Does the model G have good performance?
            print("Does the model G have good performance?")
            print("g: ", detector.get_auc_val())

            # Two preds
            ## On X_val
            print("VAL")
            aux = X_val.copy()
            aux["real"] = y_val
            aux["pred"] = detector.model.predict(X_val)
            aux["pred_proba"] = detector.model.predict_proba(X_val)[:, 1]
            # aux["ood"] = z_val_test.values

            aux["ood_pred_proba"] = detector.predict_proba(X_val)[:, 1]
            # Use the threshold to flag as OOD
            aux["ood_pred"] = detector.predict_proba(X_val)[:, 1] > 0.95
            print("Total flagged as OOD: ", aux[aux["ood_pred"] == 1].shape[0])
            auc_id = roc_auc_score(
                aux[aux["ood_pred"] == 1].real,
                aux[aux["ood_pred"] == 1].pred_proba.values,
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
                aux[aux["ood_pred"] == 1].real,
                aux[aux["ood_pred"] == 1].pred_proba.values,
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
            print(space, decay)
            res.append([datatype, state, space, decay])

# %%
# Save results
df = pd.DataFrame(res, columns=["data", "state", "space", "decay"])
df.to_csv("results/decay.csv", index=False)
# %%
# Pivot table highlight max
df.pivot_table(
    index=["data", "state"], columns="space", values="decay"
).style.highlight_min(color="lightgreen", axis=1)
# %%
