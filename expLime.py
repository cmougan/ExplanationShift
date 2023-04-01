# %%
# Import Folktables
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import wasserstein_distance

plt.style.use("seaborn-whitegrid")
rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8
rcParams.update({"font.size": 16})
import seaborn as sns
from sklearn.model_selection import train_test_split
from nobias import ExplanationShiftDetector
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import lime.lime_tabular
import os
import sys


def blockPrint():
    sys.stdout = open(os.devnull, "w")


blockPrint()

# %%

from tools.datasets import GetData

data = GetData(type="blobs")
X, y, X_ood, y_ood = data.get_data()
# %%
data = GetData(type="real", datasets="ACSIncome")
X, y = data.get_state(state="CA", year="2018", N=2_000)
# %%
model = XGBRegressor()
model.fit(X, y)
# %%
explainer = lime.lime_tabular.LimeTabularExplainer(
    X.values,
    feature_names=X.columns,
    class_names=["y"],
    discretize_continuous=True,
    verbose=True,
    mode="regression",
)


# %%
def create_explanation(X, model):
    exp = X.copy()[:0]

    for i, _ in enumerate(X.iterrows()):

        ex = explainer.explain_instance(X.iloc[i], model.predict)
        exx = pd.DataFrame(ex.local_exp[0], columns=["feature", "weight"]).sort_values(
            "feature"
        )
        exx.feature = X.columns
        exx = exx.T
        # Make header first row
        new_header = exx.iloc[0]  # grab the first row for the header
        exx = exx[1:]  # take the data less the header row
        exx.columns = new_header

        exp = exp.append(exx)
    return exp.reset_index(drop=True)


def train_esd(X, X_ood, model, detector):
    aux = create_explanation(X, model)
    aux["y"] = 0
    aux_ood = create_explanation(X_ood, model)
    aux_ood["y"] = 1
    df = aux.append(aux_ood)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df.drop("y", axis=1), df["y"], test_size=0.5, random_state=42
    )
    detector.fit(X_tr, y_tr)

    # return auc
    return roc_auc_score(y_te, detector.predict_proba(X_te)[:, 1])


# %%
df = X.copy()
df["y"] = y


aucs = {}
aucs_lime = {}
for r in [6]:
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
    params = np.linspace(0.05, 0.99, 50)

    aucs_temp = []
    aucs_lime_temp = []
    for i in tqdm(params):
        n_samples = X_ood.shape[0] - int(i * X_ood.shape[0])
        n_samples_1 = n_samples

        X_ = X_ood.loc[~X_ood.index.isin(X_ood.sample(n_samples).index)]
        X_new = X_te.sample(n_samples, replace=False).append(X_).drop(columns=["Race"])

        detector.fit(X_tr.drop(columns=["Race"]), y_tr, X_new)
        aucs_temp.append(detector.get_auc_val())

        # LIME
        aucs_lime_temp.append(
            train_esd(
                X_te, X_ood, XGBClassifier().fit(X_tr, y_tr), LogisticRegression()
            )
        )

    aucs[r] = aucs_temp
# %%
# Plot
plt.figure(figsize=(10, 6))
for r in [6]:
    # TODO: Rename labels with te
    if r == 2:
        label = "Black"
    elif r == 3:
        label = "Am-Indian"
    elif r == 6:
        label = "Asian"
    elif r == 8:
        label = "Other"
    elif r == 9:
        label = "Mixed"

    plt.plot(params, aucs[r], label=label)
    ci = 1.96 * np.std(aucs[r]) / np.sqrt(len(params))

    plt.fill_between(params, (aucs[r] - ci), (aucs[r] + ci), alpha=0.1)
plt.xlabel("Fraction of OOD data")
plt.ylabel("AUC of Explanation Shift Detector")
plt.legend()
plt.savefig("images/NewCategory.pdf", bbox_inches="tight")


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
mixed_preds = detector.model.predict_proba(
    df[df["Race"] == 9].drop(columns=["y", "Race"])
)[:, 1]
# Demographic Parity
print("Black:", wasserstein_distance(all_preds, black_preds))
print("Asian:", wasserstein_distance(all_preds, asian_preds))
print("Other:", wasserstein_distance(all_preds, other_preds))
print("Mixed:", wasserstein_distance(all_preds, mixed_preds))

# %%
