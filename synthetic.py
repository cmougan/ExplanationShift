# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import pandas as pd
import random
from scipy.stats import ks_2samp, wasserstein_distance
from doubt import Boot
from nobias import ExplanationShiftDetector

random.seed(0)
# Scikit Learn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)

plt.style.use("seaborn-whitegrid")
from matplotlib import rcParams

rcParams.update({"font.size": 12})

from xgboost import XGBClassifier
import shap
from alibi_detect.cd import TabularDrift, ClassifierDrift

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
    cd = TabularDrift(X_tr.values, p_val=0.05)
    input_ks = cd.predict(X_ood.values)["data"]["is_drift"]

    # Classifier Drift
    detector = ClassifierDrift(
        x_ref=X_te.values,
        model=model,
        backend="sklearn",
        n_folds=3,
    )
    classifierDrift = 1 - detector.predict(x=X_ood.values)["data"]["p_val"]

    # Output test
    output_ks = 1 - ks_2samp(model_r.predict(X_te), model_r.predict(X_ood)).pvalue
    wass = wasserstein_distance(model_r.predict(X_te), model_r.predict(X_ood))
    # Uncertainty
    bmodel = Boot(GradientBoostingRegressor(), random_seed=4242)
    bmodel.fit(X_tr.values, y_tr.values)
    preds, intervals = bmodel.predict(X_ood.values, uncertainty=0.95)
    unc = np.mean(intervals[:, 1] - intervals[:, 0])

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
    esd = 2 * (ESD.get_auc_val() - 0.5)

    res.append([rho, input_ks, classifierDrift, output_ks, wass, unc, esd])

# %%
results = pd.DataFrame(
    res,
    columns=["rho", "input_ks", "classifierDrift", "output_ks", "wass", "unc", "esd"],
)
# %%
plt.figure()
plt.plot(
    results["rho"],
    results["esd"],
    label="Explanation Shift",
    marker=".",
)
plt.plot(
    results["rho"],
    results["input_ks"],
    label="Input KS",
    marker="o",
)
plt.plot(
    results["rho"],
    results["classifierDrift"],
    label="Classifier Drift",
    marker="*",
)
plt.plot(
    results["rho"],
    results["output_ks"],
    label="Output KS",
    marker="s",
)
plt.plot(
    results["rho"],
    results["unc"],
    label="Uncertainty by CP",
    marker="x",
)
plt.plot(
    results["rho"],
    results["wass"],
    label="Wasserstein Ouput",
    marker="v",
)
plt.legend()
plt.xlabel(r"Correlation coefficient $\rho$")
plt.ylabel("OOD behaviour")
plt.title("Prediction of OOD behaviour under increasing covariate shift")
plt.tight_layout()
plt.savefig("images/SOTAsensitivity.pdf", bbox_inches="tight")
plt.show()

# %%
results["input_ks"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# %%
