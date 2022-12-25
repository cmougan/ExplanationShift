# %%
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})
import seaborn as sns
import pandas as pd
import random
from scipy.stats import wasserstein_distance
from tqdm import tqdm

random.seed(0)
# Scikit Learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tools.xaiUtils import ExplanationShiftDetector
import seaborn as sns

plt.style.use("seaborn-whitegrid")
from xgboost import XGBClassifier

from tools.datasets import GetData
from tools.xaiUtils import ExplanationShiftDetector

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
    X, y, _, _ = data.get_data()
    # Hold out set for CA-14
    X_cal_1, X_cal_2, y_cal_1, y_cal_2 = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=0
    )
    X, y = X_cal_1, y_cal_1

    for state in tqdm(states):
        X_ood, y_ood = data.get_state(state=state[:2], year="20" + state[2:])
        X_ood, X_ood_te, y_ood, y_ood_te = train_test_split(
            X_ood, y_ood, test_size=0.5, stratify=y_ood, random_state=0
        )

        # Build detector
        for space in ["explanation", "input", "prediction"]:
            detector = ExplanationShiftDetector(
                model=XGBClassifier(),
                gmodel=Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("lr", LogisticRegression(penalty="l1", solver="liblinear")),
                    ]
                ),
                space=space,
            )
            if "label" in X_ood.columns:
                X_ood = X_ood.drop(columns=["label"])
            detector.fit(X, y, X_ood)

            # Performance of model on X_train hold out
            auc_tr = roc_auc_score(y_cal_2, detector.model.predict_proba(X_cal_2)[:, 1])

            # Performance of detector on X_ood hold out
            preds = detector.predict_proba(X_ood_te)

            # Exp Space
            X_ood_te_ = X_ood_te.copy()
            X_ood_te_["pred"] = preds[:, 1]
            X_ood_te_["y"] = y_ood_te
            for sort in [True, False]:
                X_ood_te_ = X_ood_te_.sort_values("pred", ascending=sort)
                N = 10_000
                for N in [20_000, 10_000, 5_000, 1_000]:
                    try:
                        auc_ood = roc_auc_score(
                            X_ood_te_.head(N).y,
                            detector.model.predict_proba(
                                X_ood_te_.head(N).drop(columns=["y", "pred"])
                            )[:, 1],
                        )
                    except:
                        auc_ood = 0.5
                    res.append([datatype, sort, N, space, state, auc_tr - auc_ood])
# %%
results = pd.DataFrame(
    res, columns=["dataset", "sort", "N", "space", "state", "auc_diff"]
)
# results = results[results["N"] == 10_000]
# results1 = results[results['sort']==True]
# False is bigger AUC gap is better
# results2 = results[results['sort']==False]
# %%
# Convert results to table with State vs Space
results = results.pivot(
    index=["state", "dataset", "N", "sort"], columns="space", values="auc_diff"
).reset_index()

# %%
# Lower is better
results[results["sort"] == True].groupby(["dataset"]).mean().reset_index().drop(
    columns=["sort", "N"]
).round(3)
# %%
# Higher is better
results[results["sort"] == False].groupby(["dataset"]).mean().reset_index().drop(
    columns=["sort", "N"]
).round(3)

# %%