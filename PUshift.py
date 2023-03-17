# %%
# Import Folktables
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.style.use("seaborn-whitegrid")
rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8
rcParams.update({"font.size": 22})
import seaborn as sns
from sklearn.model_selection import train_test_split
from nobias import ExplanationShiftDetector
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# %%

from tools.datasets import GetData

data = GetData(type="blobs")
X, y, X_ood, y_ood = data.get_data()
# %%
data = GetData(type="real", datasets="ACSIncome")
X, y = data.get_state(state="CA", year="2018", N=20_000)
# %%
df = X.copy()
df["y"] = y


aucs = {}
for r in [1, 8, 6]:
    df_tr = df[df["Race"] != r]
    # Train test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        df_tr.drop("y", axis=1), df_tr["y"], test_size=0.2, random_state=42
    )
    X_ood = df[df["Race"] == r].drop("y", axis=1)

    detector = ExplanationShiftDetector(
        model=XGBClassifier(), gmodel=LogisticRegression()
    )

    # Concatenate the training and validation sets
    params = np.linspace(0.1, 0.99, 10)

    aucs_temp = []
    for i in params:
        X_ = X_ood.sample(frac=i, random_state=42, replace=False)
        X_new = X_te.append(X_).drop(columns=["Race"])

        detector.fit(X_tr.drop(columns=["Race"]), y_tr, X_new)
        aucs_temp.append(detector.get_auc_val())

    aucs[r] = aucs_temp
# %%

# Plot
plt.figure()
for r in [1, 8, 6]:
    if r == 1:
        label = "White"
    elif r == 8:
        label = "Black"
    else:
        label = "Asian"
    plt.plot(params, aucs[r], label=label)
plt.xlabel("Fraction of OOD data")
plt.ylabel("AUC of Explanation Shift Detector")
plt.savefig("images/PUshift.png")
plt.legend()


# %%
