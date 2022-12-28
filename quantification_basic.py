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
data = GetData(type="real", datasets="ACSIncome")
X, y = data.get_state(state="HI", year="2014")
# Hold out set for CA-14
X_cal_1, X_cal_2, y_cal_1, y_cal_2 = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=0
)
X, y = X_cal_1, y_cal_1
# OOD Data
X_ood, y_ood = data.get_state(state="CA", year="2018")
X_ood, X_ood_te, y_ood, y_ood_te = train_test_split(
    X_ood, y_ood, test_size=0.5, stratify=y_ood, random_state=0
)
# The y to evaluate OOD detection
X_cal_2["OOD"] = 0
X_ood_te["OOD"] = 1
X_hold = pd.concat([X_cal_2, X_ood_te])
y_hold_ood = X_hold["OOD"]

# Drop columns to avoid errors
X_hold = X_hold.drop(columns=["OOD"])
X_cal_2 = X_cal_2.drop(columns=["OOD"])
X_ood_te = X_ood_te.drop(columns=["OOD"])
# The real y
y_hold = pd.concat(
    [
        pd.DataFrame(y_cal_2, columns=["real"]),
        pd.DataFrame(y_ood_te, columns=["real"]),
    ]
)
# %%
# Fit our estimator
detector = ExplanationShiftDetector(
    model=XGBClassifier(max_depth=3, random_state=0),
    gmodel=Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(penalty="l1", solver="liblinear")),
        ]
    ),
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
aux = X_hold.copy()
aux["OOD"] = detector.predict_proba(X_hold)[:, 1]
aux["error"] = y_hold.real.values - detector.model.predict_proba(X_hold)[:, 1]
# %%
aux = aux.sort_values(by="error", ascending=False).reset_index(drop=True)
# %%
plt.figure(figsize=(10, 10))
plt.scatter(aux.index, aux["error"], label="Error")
plt.scatter(aux.index, aux["OOD"], label="OOD Explanations")
plt.xlabel("Index")
plt.ylabel("Error")
plt.title("Error vs Index")
plt.legend()
plt.show()
# %%
