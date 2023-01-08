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
    [
        "ACSMobility",
        "ACSPublicCoverage",
    ]  # , "ACSTravelTime", "ACSEmployment", "ACSIncome"]
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
    for state in ["NY", "TX", "HI"]:  # , "MN", "WI", "FL"]:
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

            # Analysis on X_ood_te
            aux = X_ood_te.copy()
            aux["real"] = y_ood_te
            aux["pred_proba"] = detector.model.predict_proba(X_ood_te)[:, 1]
            aux["ood"] = 1
            aux["ood_pred_proba"] = detector.predict_proba(X_ood_te)[:, 1]
            # Sort
            aux = aux.sort_values(by="ood_pred_proba", ascending=True)
            # Decay
            for f in [
                100,
                1000,
                int(aux.shape[0] * 0.75),
            ]:  # , int(aux.shape[0]*0.5), int(aux.shape[0]*0.25)]:
                try:
                    decay = roc_auc_score(
                        aux.head(f).real,
                        aux.head(f).pred_proba.values,
                    )
                except Exception as e:
                    print(e)
                    decay = 0.5

                res.append(
                    [
                        datatype,
                        state,
                        f,
                        space,
                        decay,
                    ]
                )

# %%
# Save results
df = pd.DataFrame(res, columns=["data", "state", "samples", "space", "decay"])
df.to_csv("results/decay.csv", index=False)
# %%
# Pivot table highlight max
df.pivot_table(
    index=["data", "state", "samples"], columns="space", values="decay"
).style.highlight_max(color="lightgreen", axis=1)
# %%
