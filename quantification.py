# %%
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})
import pandas as pd
import random

from tqdm import tqdm

random.seed(0)
# Scikit Learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tools.explanationShift import ExplanationShiftDetector

plt.style.use("seaborn-whitegrid")
from xgboost import XGBClassifier

from tools.datasets import GetData
from tools.explanationShift import ExplanationShiftDetector

# %%
res = []
states = ["NY18", "TX18", "MI18", "MN18", "WI18", "FL18"]
for datatype in tqdm(
    [
        "ACSMobility",
        # "ACSPublicCoverage",
        # "ACSTravelTime",
        # "ACSEmployment",
        # "ACSIncome",
    ]
):
    data = GetData(type="real", datasets=datatype)
    X, y = data.get_state(state="HI", year="2014")
    # Hold out set for CA-14
    X_cal_1, X_cal_2, y_cal_1, y_cal_2 = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=0
    )
    X, y = X_cal_1, y_cal_1

    for state in tqdm(states):
        X_ood, y_ood = data.get_state(state=state[:2], year="20" + state[2:], N=20_000)
        X_ood, X_ood_te, y_ood, y_ood_te = train_test_split(
            X_ood, y_ood, test_size=0.5, stratify=y_ood, random_state=0
        )
        X_cal_2["OOD"] = 0
        X_ood_te["OOD"] = 1
        X_hold = pd.concat([X_cal_2, X_ood_te])
        y_hold_ood = X_hold["OOD"]
        X_hold = X_hold.drop(columns=["OOD"])
        X_cal_2 = X_cal_2.drop(columns=["OOD"])
        X_ood_te = X_ood_te.drop(columns=["OOD"])
        y_hold = pd.concat(
            [
                pd.DataFrame(y_cal_2, columns=["real"]),
                pd.DataFrame(y_ood_te, columns=["real"]),
            ]
        )

        # Build detector
        for space in ["explanation", "input", "prediction"]:
            detector = ExplanationShiftDetector(
                model=XGBClassifier(max_depth=3, random_state=0),
                # gmodel=Pipeline([("scaler", StandardScaler()),("lr", LogisticRegression(penalty="l1", solver="liblinear")),]),
                gmodel=XGBClassifier(max_depth=3, random_state=0),
                space=space,
                masker=False,
            )
            if "label" in X_ood.columns:
                X_ood = X_ood.drop(columns=["label"])
            detector.fit(X, y, X_ood)

            # Performance of model on X_train hold out
            auc_tr = roc_auc_score(y_cal_2, detector.model.predict_proba(X_cal_2)[:, 1])

            # Performance of detector on X_ood hold out
            auc_hold = roc_auc_score(
                y_ood_te, detector.model.predict_proba(X_ood_te)[:, 1]
            )
            print("Performance of f:", space, datatype, state, auc_hold)
            print("Performance of g:", detector.get_auc_val())
            # Exp Space
            X_hold_ = X_hold.copy()
            X_hold_["y"] = y_hold.real.values
            X_hold_["pred"] = detector.predict_proba(X_hold)[:, 1]

            for sort in [True, False]:
                X_hold_ = X_hold_.sort_values("pred", ascending=sort)
                for N in [20_000, 5_000, 1_000, 500, 100]:
                    try:
                        auc_ood = roc_auc_score(
                            X_hold_.head(N).y,
                            detector.model.predict_proba(
                                X_hold_.head(N).drop(columns=["y", "pred"])
                            )[:, 1],
                        )

                    except Exception as e:
                        print(e)
                        print("Value Error", N, space, datatype, state)
                        auc_ood = 1
                    res.append([datatype, sort, N, space, state, auc_tr - auc_ood])
# %%
results_ = pd.DataFrame(
    res, columns=["dataset", "sort", "N", "space", "state", "auc_diff"]
)
# %%
# Convert results to table with State vs Space
results_ = results_.pivot(
    index=["state", "dataset", "N", "sort"], columns="space", values="auc_diff"
).reset_index()
# %%
results = results_[results_["N"] == 1_000]
# %%
# Closer to 0 is better State
results[results["sort"] == True].groupby(
    ["dataset", "state"]
).mean().reset_index().drop(columns=["sort", "N"]).round(3).to_csv(
    "results/results_low.csv"
)  # .style.highlight_min(color="lightgreen", axis=1, subset=["explanation", "input", "prediction"])
results[results["sort"] == True].groupby(
    ["dataset", "state"]
).mean().reset_index().drop(columns=["sort", "N"]).round(3).style.highlight_min(
    color="lightgreen", axis=1, subset=["explanation", "input", "prediction"]
)

# %%
results = results_[results_["N"] == 1000]
# %%
# Higher is better highlight State
results[results["sort"] == False].groupby(
    ["dataset", "state"]
).mean().reset_index().drop(columns=["sort", "N"]).round(3).to_csv(
    "results/results_high.csv"
)
results[results["sort"] == False].groupby(
    ["dataset", "state"]
).mean().reset_index().drop(columns=["sort", "N"]).round(3).style.highlight_max(
    color="lightgreen", axis=1, subset=["explanation", "input", "prediction"]
)
# %%
import seaborn as sns

sns.kdeplot(detector.predict_proba(X_hold)[:, 1])

# %%
if "label" in X_ood.columns:
    X_ood = X_ood.drop(columns=["label"])
detector = ExplanationShiftDetector(
    model=XGBClassifier(max_depth=3, random_state=0),
    # gmodel=Pipeline([("scaler", StandardScaler()),("lr", LogisticRegression(penalty="l1", solver="liblinear")),]),
    gmodel=XGBClassifier(max_depth=3, random_state=0),
    space="prediction",
    masker=False,
)

detector.fit(X, y, X_ood)
# %%
# Evaluate on hold out
print(
    "f: ", roc_auc_score(y_hold.real.values, detector.model.predict_proba(X_hold)[:, 1])
)
print("g: ", detector.get_auc_val())
print("g2: ", roc_auc_score(y_hold_ood, detector.predict_proba(X_hold)[:, 1]))
# %%
import numpy as np

X_hold_["pred"] = detector.predict_proba(X_hold)[:, 1]
X_hold_["error"] = np.abs(
    detector.model.predict_proba(X_hold)[:, 1] - y_hold.real.values
)
X_hold_ = X_hold_.sort_values("pred", ascending=True).reset_index(drop=True)
# %%
plt.figure(figsize=(10, 10))
plt.scatter(X_hold_.index, X_hold_.error, label="error")
plt.scatter(X_hold_.index, X_hold_.pred, label="pred")
plt.legend()
# %%
detector.predict_proba(X_hold)[:, 1]
# %%
from scipy.stats import pearsonr

# %%
pearsonr(X_hold_.pred, X_hold_.error)
# %%
