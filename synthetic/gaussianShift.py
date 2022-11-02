# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import seaborn as sns
import pandas as pd
import random
from collections import defaultdict

random.seed(0)
# Scikit Learn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

plt.style.use("seaborn-whitegrid")
from xgboost import XGBRegressor, XGBClassifier
import shap

import sys

sys.path.append("../")
from fairtools.xaiUtils import ShapEstimator

# %%
## Sensitivity experiment
res_exp = []
res_out = []
res_inp = []
iters = np.linspace(0, 1, 20)
for rho in iters:
    ## Create variables
    ### Normal
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
    X_ood = pd.DataFrame(data=[x11, x22, x33]).T
    X_ood.columns = ["Var%d" % (i + 1) for i in range(X_ood.shape[1])]

    ## Split Data
    X_tr, X_te, y_tr, y_te = train_test_split(df.drop(columns="target"), df["target"])
    ## Fit our ML model
    model = XGBRegressor(random_state=0)
    # model = LinearRegression()
    model.fit(X_tr, y_tr)

    ## Real explanation
    explainer = shap.Explainer(model)
    # explainer = shap.LinearExplainer(model, X_te, feature_dependence="correlation_dependent")
    # shap.KernelExplainer(model.predict,X_te,nsamples=100)
    shap_values = explainer(X_te)
    exp = pd.DataFrame(
        data=shap_values.values, columns=["Shap%d" % (i + 1) for i in range(3)]
    )
    ##  OODexplanation
    preds_ood = model.predict(X_ood)
    preds_te = model.predict(X_te)
    shap_values = explainer(X_ood)
    exp_ood = pd.DataFrame(
        data=shap_values.values, columns=["Shap%d" % (i + 1) for i in range(3)]
    )
    y_ood = X_ood["Var1"] * X_ood["Var2"] + np.random.normal(0, 0.1, samples)

    ## Does xAI help to solve this issue?
    ## Shap Estimator
    exp_ood["label"] = 1
    exp["label"] = 0
    exp_space = pd.concat([exp, exp_ood])

    input_space = pd.concat([X_te, X_ood])
    input_space["label"] = exp_space["label"].values
    out_space = np.concatenate([preds_te, preds_ood])

    ## Model to be used
    # logreg = XGBClassifier(random_state=0)
    logreg = LogisticRegression()
    # logreg = Pipeline((("scaler", StandardScaler()), ("logreg", LogisticRegression())))
    # Explanation Space
    S_tr, S_te, yy_tr, yy_te = train_test_split(
        exp_space.drop(columns="label"),
        exp_space[["label"]],
        random_state=0,
        test_size=0.5,
        stratify=exp_space[["label"]],
    )
    logreg.fit(S_tr, yy_tr)
    # Evaluation
    preds = logreg.predict_proba(S_te)[:, 1]
    res_exp.append(roc_auc_score(yy_te, preds))

    # Input Space
    S_tr, S_te, yy_tr, yy_te = train_test_split(
        input_space.drop(columns="label"),
        input_space[["label"]],
        random_state=0,
        test_size=0.5,
        stratify=input_space[["label"]],
    )
    logreg.fit(S_tr, yy_tr)
    ## Evaluation
    preds = logreg.predict_proba(S_te)[:, 1]
    res_inp.append(roc_auc_score(yy_te, preds))

    # Output Space
    S_tr, S_te, yy_tr, yy_te = train_test_split(
        pd.DataFrame(out_space),
        exp_space[["label"]],
        random_state=0,
        test_size=0.5,
        stratify=exp_space[["label"]],
    )
    logreg.fit(S_tr, yy_tr)
    ## Evaluation
    preds = logreg.predict_proba(S_te)[:, 1]
    res_out.append(roc_auc_score(yy_te, preds))
    ## Plotting
    """
    plt.figure()
    plt.scatter(x1, x2, label="X")
    plt.scatter(x11, x22, label="X*")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()

    ## Plots of the spaces
    print("----------")
    print("PARAMETER:  ", rho)
    plt.figure()
    sns.kdeplot(preds_te, label="In")
    sns.kdeplot(preds_ood, label="Out")
    plt.vlines(logreg.coef_[0][0], 0, 1, label="Coef")
    plt.show()

    plt.figure()
    plt.scatter(
        exp_space[exp_space["label"] == 1].Shap1,
        exp_space[exp_space["label"] == 1].Shap2,
        label="X",
    )
    plt.scatter(
        exp_space[exp_space["label"] == 0].Shap1,
        exp_space[exp_space["label"] == 0].Shap2,
        label="X*",
    )
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    """
# %%
plt.figure()
plt.title("Sensitivity to multivariate shift correlation")
plt.plot(iters, res_exp, label="Explanation Space")
plt.plot(iters, res_inp, label="Input Space")
plt.plot(iters, res_out, label="Output Space")
plt.xlabel(r"Correlation coefficient $\rho$")
plt.ylabel(r"$g_\psi$ AUC")
plt.legend()
plt.tight_layout()
plt.savefig("images/sensivity.png")
plt.show()
# %%
