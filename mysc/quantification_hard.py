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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from tools.explanationShift import ExplanationShiftDetector


from xgboost import XGBClassifier
from tqdm import tqdm

from tools.datasets import GetData
from tools.explanationShift import ExplanationShiftDetector

# %%
res = []
states = ["NY18", "TX18"]  # , "MI18", "MN18", "WI18", "FL18"]
for datatype in tqdm(
    [
        "ACSMobility",
        "ACSPublicCoverage",
        # "ACSTravelTime",
        # "ACSEmployment",
        # "ACSIncome",
    ]
):
    for state in tqdm(states):
        data = GetData(type="real", datasets=datatype)
        X, y = data.get_state(state=state[:2], year="20" + state[2:], N=20_000)

        # What is the most important feature?
        model = XGBClassifier()
        model.fit(X, y)
        importances = model.feature_importances_
        # We select the most important feature
        most_important = np.argmax(importances)
        # Conver feature importance to pandas
        importances = pd.DataFrame(importances, index=X.columns, columns=["importance"])
        # print('Most important feature: "{}"'.format(importances.index[most_important]))
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

        for space in ["explanation", "input", "prediction"]:
            print(space)
            detector = ExplanationShiftDetector(
                model=XGBClassifier(max_depth=3, random_state=0),
                gmodel=Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("lr", LogisticRegression(penalty="l1", solver="liblinear")),
                    ]
                ),
                space=space,
                masker=False,
            )
            if "label" in X_ood_tr.columns:
                X_ood_tr = X_ood_tr.drop(columns=["label"])
            detector.fit(X, y, X_ood_tr)

            # Performance of model on X_train hold out
            auc_tr = roc_auc_score(y_val, detector.model.predict_proba(X_val)[:, 1])

            # Performance of detector on X_ood hold out
            auc_hold = roc_auc_score(
                y_ood_te, detector.model.predict_proba(X_ood_te)[:, 1]
            )

            print(space, auc_hold)
            # Analysis
            X_ood_te_ = X_ood_te.copy()
            X_ood_te_["pred"] = detector.predict_proba(X_ood_te)[:, 1]
            X_ood_te_["y"] = y_ood_te

            for sort in [True, False]:
                X_ood_te_ = X_ood_te_.sort_values("pred", ascending=sort)
                for N in [20_000, 5_000, 1_000, 500, 100]:
                    try:
                        auc_ood = roc_auc_score(
                            X_ood_te_.head(N).y,
                            detector.model.predict_proba(
                                X_ood_te_.head(N).drop(columns=["y", "pred"])
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
