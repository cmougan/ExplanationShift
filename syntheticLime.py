# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import seaborn as sns
import pandas as pd
import random
from collections import defaultdict
from scipy.stats import ks_2samp, wasserstein_distance
from doubt import Boot
from skshift import ExplanationShiftDetector

random.seed(0)
# Scikit Learn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.ensemble import (
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)

plt.style.use("seaborn-v0_8-whitegrid")
from matplotlib import rcParams

rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8
rcParams.update({"font.size": 22})

from xgboost import XGBRegressor, XGBClassifier
import shap
from alibi_detect.cd import ChiSquareDrift, TabularDrift, ClassifierDrift
from tqdm import tqdm
import lime.lime_tabular
import os
import sys


def blockPrint():
    sys.stdout = open(os.devnull, "w")


blockPrint()
# %%


# %%
def create_explanation(X, model):
    exp = X.copy()[:0]

    for i, _ in tqdm(enumerate(X.iterrows())):
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
        exx.reset_index(inplace=True)
        exp = pd.concat([exp, exx])

    return exp


def train_esd(X, X_ood, model, detector):
    aux = create_explanation(X, model)
    aux["y"] = 0
    aux_ood = create_explanation(X_ood, model)
    aux_ood["y"] = 1
    df = pd.concat([aux,aux_ood]).drop(columns=["index"])
    # df = aux.append(aux_ood).drop(columns=["index"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        df.drop("y", axis=1), df["y"], test_size=0.5, random_state=42
    )
    detector.fit(X_tr, y_tr)

    # return auc
    return roc_auc_score(y_te, detector.predict_proba(X_te)[:, 1])


# %%
res = []
for i in np.linspace(0, 1, 11):
    rho = i
    ## Sensitivity experiment
    sigma = 1

    mean = [0, 0]
    cov = [[sigma, 0], [0, sigma]]
    samples = 5_000
    x1, x2 = np.random.multivariate_normal(mean, cov, samples).T
    x3 = np.random.normal(0, sigma, samples)
    # Different values
    mean = [0, 0]
    cov = [[sigma, rho], [rho, sigma]]
    x11, x22 = np.random.multivariate_normal(mean, cov, samples).T
    x33 = np.random.normal(0, sigma, samples)

    # Create Data
    df = pd.DataFrame(data=[x1, x2, x3]).T
    df.columns = ["Var%d" % (i + 1) for i in range(df.shape[1])]
    # df["target"] = np.where(df["Var1"] * df["Var2"] > 0, 1, 0)
    df["target"] = (
        df["Var1"] * df["Var2"] + df["Var3"] + np.random.normal(0, 0.1, samples)
    )
    df["target"] = np.where(df["target"] > df["target"].mean(), 1, 0)
    X_ood = pd.DataFrame(data=[x11, x22, x33]).T
    X_ood.columns = ["Var%d" % (i + 1) for i in range(X_ood.shape[1])]

    ## Split Data
    X_tr, X_te, y_tr, y_te = train_test_split(df.drop(columns="target"), df["target"])
    ## Fit our ML model
    model = GradientBoostingClassifier()
    model_r = GradientBoostingRegressor()
    # model = LinearRegression()
    model.fit(X_tr, y_tr)
    model_r.fit(X_tr, y_tr)

    # Input KS Test
    input_ks = 1
    # Classifier Drift
    classifierDrift = 1
    # Output test
    output_ks = 1
    wass = 1
    # Uncertainty
    unc = 1

    # Explanation Shift
    ESD = ExplanationShiftDetector(
        model=XGBClassifier(),
        gmodel=Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(penalty="l1", solver="liblinear")),
            ]
        ),
    )
    ESD.fit(X_tr, y_tr, X_ood)
    esd = ESD.get_auc_val()

    # Lime
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_tr.values,
        feature_names=X_tr.columns,
        class_names=["y"],
        discretize_continuous=True,
        verbose=True,
        mode="regression",
    )
    auc_lime = train_esd(
        X_te, X_ood, XGBClassifier().fit(X_tr, y_tr), LogisticRegression()
    )

    res.append([rho, input_ks, classifierDrift, output_ks, wass, unc, esd, auc_lime])

# %%
results = pd.DataFrame(
    res,
    columns=[
        "rho",
        "input_ks",
        "classifierDrift",
        "output_ks",
        "wass",
        "unc",
        "esd",
        "lime",
    ],
)
# %%
plt.figure()
plt.plot(results["rho"], results["esd"], label="Explanation Shift - SHAP")
ci = 1.96 * np.std(results["esd"]) / np.sqrt(len(results["rho"]))
plt.fill_between(
    results["rho"], (results["esd"] - ci), (results["esd"] + ci), alpha=0.1
)


plt.plot(results["rho"], results["lime"], label="Explanation Shift - Lime")
ci = 1.96 * np.std(results["lime"]) / np.sqrt(len(results["rho"]))
plt.fill_between(
    results["rho"], (results["lime"] - ci), (results["lime"] + ci), alpha=0.1
)


plt.legend()
plt.xlabel("Correlation coefficient")
plt.ylabel("AUC Explanation Shift Detector")
plt.title("Sensitivity to Multicovariate Shift")
plt.tight_layout()
plt.savefig("images/SOTAsensitivityLime.pdf", bbox_inches="tight")
plt.show()

# %%
