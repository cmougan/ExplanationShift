# %%
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 14})
plt.style.use("seaborn-v0_8-whitegrid")
import pandas as pd
import random

random.seed(0)
# Scikit Learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from tools.explanationShift import ExplanationShiftDetector
from sklearn.metrics import r2_score

# External Libraries
from xgboost import XGBClassifier
from tqdm import tqdm

from tools.datasets import GetData
from tools.explanationShift import ExplanationShiftDetector
from tools.ATC import ATC

# %%
res = []
states = [
    "AK",
    "AL",
    "AR",
    "AZ",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "IA",
    "ID",
    "IL",
    "IN",
    "KS",
    "KY",
    "LA",
    "MA",
    "MD",
    "ME",
    "MI",
    "MN",
    "MO",
    "MS",
    "MT",
    "NC",
    "ND",
    "NE",
    "NH",
    "NJ",
    "NM",
    "NV",
    "NY",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VA",
    "VT",
    "WA",
    "WI",
    "WV",
    "WY",
]
len(states)
# %%
for datatype in tqdm(
    [
        "ACSIncome",
    ]
):
    data = GetData(type="real", datasets=datatype)
    X, y = data.get_state(state="CA", year="2014")
    sc = StandardScaler()
    sc.fit(X)
    X = pd.DataFrame(sc.transform(X), columns=X.columns)
    # Hold out set for CA-14
    X_cal_1, X_cal_2, y_cal_1, y_cal_2 = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=0
    )
    X, y = X_cal_1, y_cal_1
    for year in ["2018"]:
        for state in tqdm(states):
            print(state)
            X_ood, y_ood = data.get_state(state=state[:2], year=year, N=20_000)
            X_ood = pd.DataFrame(sc.transform(X_ood), columns=X_ood.columns)
            X_ood, X_ood_te, y_ood, y_ood_te = train_test_split(
                X_ood, y_ood, test_size=0.5, stratify=y_ood, random_state=0
            )
            # Split the data into a holdout
            X_cal_2["ood"] = 0
            X_ood_te["ood"] = 1
            X_cal_2["label"] = y_cal_2
            X_ood_te["label"] = y_ood_te
            X_hold = pd.concat([X_cal_2, X_ood_te])
            y_hold = X_hold["label"]
            y_hold_ood = X_hold["ood"]
            X_hold = X_hold.drop(columns=["ood", "label"])

            # Build detector
            for space in ["input", "prediction", "explanation"]:
                detector = ExplanationShiftDetector(
                    model=XGBClassifier(max_depth=3, random_state=0, verbosity=0),
                    gmodel=LogisticRegression(random_state=0, max_iter=1000),
                    space=space,
                    masker=True,
                )
                if "label" in X_ood.columns:
                    X_ood = X_ood.drop(columns=["label"])
                detector.fit(X, y, X_ood)

                # Performance of detector on X_ood hold out
                # auc_hold = roc_auc_score(y_hold, detector.model.predict_proba(X_hold)[:, 1])
                auc_hold = accuracy_score(y_hold, detector.model.predict(X_hold))

                # Performance of G
                auc_ood = roc_auc_score(
                    y_hold_ood, detector.predict_proba(X_hold)[:, 1]
                )
                res.append([datatype, state, space, auc_hold, auc_ood])

            # ATC
            atc = ATC()
            atc.fit(detector.model.predict_proba(X), y)
            auc_ood = atc.predict(detector.model.predict_proba(X_hold)) / 100
            res.append([datatype, state, "atc", auc_hold, auc_ood])


# %%
results_ = pd.DataFrame(
    res, columns=["dataset", "state", "space", "auc_hold", "auc_ood"]
)
# %%
plt.figure(figsize=(10, 10))
for space in ["explanation", "input", "prediction", "atc"]:
    results = results_[results_["space"] == space]
    r2 = np.round(r2_score(results["auc_hold"], results["auc_ood"]), decimals=2)
    label = space + " R2: " + str(r2)
    plt.scatter(y=results["auc_hold"], x=results["auc_ood"], label=label)
plt.ylabel("AUC Hold")
plt.xlabel("AUC OOD")

plt.legend()
