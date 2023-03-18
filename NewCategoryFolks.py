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
for r in [2, 6, 8, 9]:
    df_tr = df[df["Race"] != r]
    # Train test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        df_tr.drop("y", axis=1), df_tr["y"], test_size=0.5, random_state=42
    )
    X_ood = df[df["Race"] == r].drop("y", axis=1)

    detector = ExplanationShiftDetector(
        model=XGBClassifier(), gmodel=LogisticRegression()
    )

    # Concatenate the training and validation sets
    params = np.linspace(0.1, 0.99, 10)

    aucs_temp = []
    for i in params:
        n_samples = X_ood.shape[0] - int(i * X_ood.shape[0])
        n_samples_1 = n_samples

        X_ = X_ood.loc[~X_ood.index.isin(X_ood.sample(10).index)]
        X_new = X_te.sample(n_samples, replace=False).append(X_).drop(columns=["Race"])

        detector.fit(X_tr.drop(columns=["Race"]), y_tr, X_new)
        aucs_temp.append(detector.get_auc_val())

    aucs[r] = aucs_temp
# %%
# Plot
plt.figure()
for r in [2, 6, 8, 9]:
    # TODO: Rename labels with te
    if r == 2:
        label = "Black"
    elif r == 6:
        label = "Asian"
    elif r == 8:
        label = "Other"
    elif r == 9:
        label = "Mixed"

    plt.plot(params, aucs[r], label=label)
plt.xlabel("Fraction of OOD data")
plt.ylabel("AUC of Explanation Shift Detector")
plt.savefig("images/PUshift.png")
plt.legend()


# %%
# Some Fariness metrics
all_preds = detector.model.predict_proba(df.drop(columns=["y", "Race"]))[:, 1]
black_preds = detector.model.predict_proba(
    df[df["Race"] == 2].drop(columns=["y", "Race"])
)[:, 1]
asian_preds = detector.model.predict_proba(
    df[df["Race"] == 5].drop(columns=["y", "Race"])
)[:, 1]
other_preds = detector.model.predict_proba(
    df[df["Race"] == 8].drop(columns=["y", "Race"])
)[:, 1]
sns.kdeplot(all_preds, label="All")
sns.kdeplot(black_preds, label="Black")
sns.kdeplot(asian_preds, label="Asian")
sns.kdeplot(other_preds, label="Other")
# %%
from scipy.stats import wasserstein_distance

# %%
# Demographic Parity
print("Black:", wasserstein_distance(all_preds, black_preds))
print("Asian:", wasserstein_distance(all_preds, asian_preds))
print("Other:", wasserstein_distance(all_preds, other_preds))
