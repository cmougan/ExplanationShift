# %%
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")
plt.rcParams.update({"font.size": 14})
import pandas as pd
import random
import numpy as np

random.seed(0)
# Scikit Learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from tools.explanationShift import ExplanationShiftDetector


from xgboost import XGBClassifier, XGBRegressor
from tqdm import tqdm

from tools.datasets import GetData
from tools.explanationShift import ExplanationShiftDetector

# %%
X = pd.read_csv("data/power/train.csv")
X_tr = X.drop(columns=["power"])
y_tr = X["power"]
del X
X_ood_tr = pd.read_csv("data/power/dev_in.csv")
X_ood_te = pd.read_csv("data/power/dev_out.csv")
# Drop and Split label
y_ood_tr = X_ood_tr["power"]
y_ood_te = X_ood_te["power"]
X_ood_tr = X_ood_tr.drop(columns=["power"])
X_ood_te = X_ood_te.drop(columns=["power"])

# %%
datatype = "power"
state = "power"
res = []
for space in ["explanation", "input", "prediction"]:
    print(space)
    detector = ExplanationShiftDetector(
        model=XGBRegressor(max_depth=3, random_state=0),
        gmodel=Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(penalty="l1", solver="liblinear")),
            ]
        ),
        space=space,
        masker=True,
    )
    if "label" in X_ood_tr.columns:
        X_ood_tr = X_ood_tr.drop(columns=["label"])
    detector.fit(X_tr, y_tr, X_ood_tr)

    # Performance of model on X_train hold out
    # auc_tr = roc_auc_score(y_val, detector.model.predict_proba(X_val)[:, 1])

    # Performance of detector on X_ood hold out
    auc_hold = mean_absolute_error(y_ood_te, detector.model.predict(X_ood_te))

    print(space, auc_hold)
    # Analysis
    X_ood_te_ = X_ood_te.copy()
    X_ood_te_["pred"] = detector.predict_proba(X_ood_te)[:, 1]
    X_ood_te_["y"] = y_ood_te

    for sort in [True, False]:
        X_ood_te_ = X_ood_te_.sort_values("pred", ascending=sort)
        for N in [5_000, 1_000, 500, 100]:
            try:
                auc_ood = mean_absolute_error(
                    X_ood_te_.head(N).y,
                    detector.model.predict(
                        X_ood_te_.head(N).drop(columns=["y", "pred"])
                    ),
                )
            except Exception as e:
                print(e)
                print("Value Error", N, space, datatype, state)
                auc_ood = 1
            res.append([datatype, sort, N, space, state, auc_hold - auc_ood])

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
results = results_[results_["N"] == 1000]
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
results_.style.highlight_min(
    color="lightgreen", axis=1, subset=["explanation", "input", "prediction"]
)
# %%
