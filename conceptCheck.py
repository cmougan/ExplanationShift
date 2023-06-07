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


def c2st(x, y, model=XGBClassifier()):
    # Convert to dataframes
    # Dinamic columns depending on lenght of x
    try:
        columns = [f"var_{i}" for i in range(x.shape[1])]
    except:
        columns = ["var_0"]
    if isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x.values, columns=columns)
        x["label"] = 0
        y = pd.DataFrame(y.values, columns=columns)
        y["label"] = 1
    else:
        x = pd.DataFrame(x, columns=columns)
        x["label"] = 0
        y = pd.DataFrame(y, columns=columns)
        y["label"] = 1

    # Concatenate
    df = pd.concat([x, y], axis=0)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(["label"], axis=1), df["label"], test_size=0.5, random_state=42
    )

    # Train classifier
    clf = model.fit(X_train, y_train)

    # Evaluate AUC
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    return auc


# %%

from tools.datasets import GetData

data = GetData(type="blobs")
X, y, X_ood, y_ood = data.get_data()
# %%
data = GetData(type="real", datasets="ACSIncome")
X, y = data.get_state(state="CA", year="2018", N=20_000)
# %%
us_states = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
]
# %%
res = []
for state in us_states:
    X_ood, y_ood = data.get_state(state=state, year="2018", N=20_000)

    # Concept shift if there is AUC>0.5
    concept = c2st(X[y == True], X_ood[y_ood == True])

    # Covariate shift if there is AUC>0.5
    covariate = c2st(X, X_ood)
    res.append([state, concept, covariate])

res = pd.DataFrame(res, columns=["state", "concept", "covariate"])
# %%

res
# %%

res["concept"] - res["covariate"]
# %%
