# %%
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})
plt.style.use("seaborn-whitegrid")
import pandas as pd
import random

random.seed(0)
# Scikit Learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss, f1_score
from sklearn.preprocessing import StandardScaler
from tools.explanationShift import ExplanationShiftDetector

# External Libraries
from xgboost import XGBClassifier
from tqdm import tqdm

from tools.datasets import GetData
from tools.explanationShift import ExplanationShiftDetector

# %%
res = []
states = ["NY18", "TX18", "MI18", "MN18", "WI18", "FL18"]
for datatype in tqdm(
    [
        "ACSMobility",
        "ACSPublicCoverage",
        "ACSTravelTime",
        "ACSEmployment",
        "ACSIncome",
    ]
):
    data = GetData(type="real", datasets=datatype)
    X, y = data.get_state(state="CA", year="2014")
    sc = StandardScaler().fit(X)
    X = pd.DataFrame(sc.transform(X), columns=X.columns)
    # Hold out set for CA-14
    X_cal_1, X_cal_2, y_cal_1, y_cal_2 = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=0
    )
    X, y = X_cal_1, y_cal_1

    for state in tqdm(states):
        X_ood, y_ood = data.get_state(state=state[:2], year="20" + state[2:], N=20_000)
        X_ood = pd.DataFrame(sc.transform(X_ood), columns=X_ood.columns)
        X_ood, X_ood_te, y_ood, y_ood_te = train_test_split(
            X_ood, y_ood, test_size=0.5, stratify=y_ood, random_state=0
        )

        # Build detector
        for space in ["explanation", "input", "prediction"]:
            detector = ExplanationShiftDetector(
                gmodel=XGBClassifier(max_depth=3, random_state=0, verbosity=0),
                model=LogisticRegression(random_state=0, max_iter=1000),
                space=space,
                masker=True,
            )
            if "label" in X_ood.columns:
                X_ood = X_ood.drop(columns=["label"])
            detector.fit(X, y, X_ood)

            # Performance of model on X_train hold out
            auc_tr = roc_auc_score(y_cal_2, detector.model.predict_proba(X_cal_2)[:, 1])

            # Performance of detector on X_ood hold out
            auc_hold = roc_auc_score(y_ood_te, detector.model.predict(X_ood_te))

            print(space, datatype, state, auc_hold)
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
