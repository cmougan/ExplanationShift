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
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

plt.style.use("seaborn-whitegrid")
from xgboost import XGBRegressor, XGBClassifier
import shap

import sys

sys.path.append("../")
from fairtools.xaiUtils import ShapEstimator

# %%
## Create variables
### Normal
sigma = 5
mean = [0, 0]
cov = [[sigma, 0], [0, sigma]]
samples = 5_000
x1, x2 = np.random.multivariate_normal(mean, cov, samples).T
# Different values
mean = [0, 0]
rho = 3
cov = [[sigma, out], [out, sigma]]
x11, x22 = np.random.multivariate_normal(mean, cov, samples).T
# %%
## Plotting
plt.figure()
sns.histplot(x1, color="r")
sns.histplot(x11)
# %%
plt.figure()
plt.scatter(x1, x2, label="X")
plt.scatter(x11, x22, label="X*")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()

# %%
df = pd.DataFrame(data=[x1, x2]).T
df.columns = ["Var%d" % (i + 1) for i in range(df.shape[1])]
# df["target"] = np.where(df["Var1"] * df["Var2"] > 0, 1, 0)
df["target"] = df["Var1"] * df["Var2"] + np.random.normal(0, 0.1, samples)
# %%
## Fit our ML model
X_tr, X_te, y_tr, y_te = train_test_split(df.drop(columns="target"), df[["target"]])
# %%
model = XGBRegressor(random_state=0)
preds_val = cross_val_predict(model, X_tr, y_tr, cv=3)
model.fit(X_tr, y_tr)
preds_test = model.predict(X_te)
# %%
## Real explanation
explainer = shap.Explainer(model)
shap_values = explainer(X_te)
exp = pd.DataFrame(
    data=shap_values.values, columns=["Shap%d" % (i + 1) for i in range(2)]
)
# %%
## OOD Data + OODexplanation
X_ood = pd.DataFrame(data=[x11, x22]).T
X_ood.columns = ["Var%d" % (i + 1) for i in range(X_ood.shape[1])]
preds_ood = model.predict(X_ood)
shap_values = explainer(X_ood)
exp_ood = pd.DataFrame(
    data=X_ood.values, columns=["Shap%d" % (i + 1) for i in range(2)]
)
y_ood = X_ood["Var1"] * X_ood["Var2"] + np.random.normal(0, 0.1, samples)
# %%
# Original error
print("Train (val) error:", np.round(mean_squared_error(y_tr, preds_val), decimals=3))
print("Test error:", np.round(mean_squared_error(y_te, preds_test), decimals=3))
print("OOD error:", np.round(mean_squared_error(y_ood, preds_ood), decimals=3))
# %%
print("Feat 1")
print("Explanation Shift:", ks_2samp(exp_ood["Shap1"], exp["Shap1"]))
print("Distribution Shift:", ks_2samp(x11, x1))
# %%
print("Feat 2")
print("Explanation Shift: ", ks_2samp(exp_ood["Shap2"], exp["Shap2"]))
print("Distribution Shift:", ks_2samp(x22, x2))
# %%
print("Target")
print("Original label change: ", ks_2samp(y_te["target"], y_ood))
print("Prediction shift:", ks_2samp(preds_test, preds_ood))

# %%
## Does xAI help to solve this issue?
## Shap Estimator
exp_ood["label"] = 1
exp["label"] = 0
exp_space = pd.concat([exp, exp_ood])
# %%
exp_space.label.value_counts()
# %%
S_tr, S_te, y_tr, y_te = train_test_split(
    exp_space.drop(columns="label"), exp_space[["label"]], random_state=0, test_size=0.5
)
logreg = LogisticRegression(random_state=0)
logreg = XGBClassifier(random_state=0)
logreg.fit(S_tr, y_tr)
# %%
preds = logreg.predict_proba(S_te)[:, 1]
print(roc_auc_score(y_te, preds))
# %%
## Acountability on G
explainer = shap.Explainer(logreg)
shap_values = explainer(S_te)
shap.plots.bar(shap_values)
# %%
## Sensitivity experiment
res_exp = []
res_out = []
res_inp = []
iters = np.linspace(0, 1, 10)
for rho in iters:
    ## Create variables
    ### Normal
    sigma = 2
    mean = [0, 0]
    cov = [[sigma, 0], [0, sigma]]
    samples = 5_000
    x1, x2 = np.random.multivariate_normal(mean, cov, samples).T
    # Different values
    mean = [0, 0]
    cov = [[sigma, rho], [rho, sigma]]
    x11, x22 = np.random.multivariate_normal(mean, cov, samples).T

    ## Plotting
    plt.figure()
    plt.scatter(x1, x2, label="X")
    plt.scatter(x11, x22, label="X*")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()

    # Create Data
    df = pd.DataFrame(data=[x1, x2]).T
    df.columns = ["Var%d" % (i + 1) for i in range(df.shape[1])]
    # df["target"] = np.where(df["Var1"] * df["Var2"] > 0, 1, 0)
    df["target"] = df["Var1"] * df["Var2"] + np.random.normal(0, 0.1, samples)

    ## Split Data
    X_tr, X_te, y_tr, y_te = train_test_split(df.drop(columns="target"), df["target"])
    ## Fit our ML model
    model = XGBRegressor(random_state=0)
    model.fit(X_tr, y_tr)

    ## Real explanation
    explainer = shap.Explainer(model)
    shap_values = explainer(X_te)
    exp = pd.DataFrame(
        data=shap_values.values, columns=["Shap%d" % (i + 1) for i in range(2)]
    )
    ## OOD Data + OODexplanation
    X_ood = pd.DataFrame(data=[x11, x22]).T
    X_ood.columns = ["Var%d" % (i + 1) for i in range(X_ood.shape[1])]
    preds_ood = model.predict(X_ood)
    preds_te = model.predict(X_te)
    shap_values = explainer(X_ood)
    exp_ood = pd.DataFrame(
        data=X_ood.values, columns=["Shap%d" % (i + 1) for i in range(2)]
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

    # Explanation Space
    S_tr, S_te, yy_tr, yy_te = train_test_split(
        exp_space.drop(columns="label"),
        exp_space[["label"]],
        random_state=0,
        test_size=0.5,
    )
    logreg = XGBClassifier(random_state=0)
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
    )
    logreg = XGBClassifier(random_state=0)
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
    )
    logreg = XGBClassifier(random_state=0)
    logreg.fit(S_tr, yy_tr)
    ## Evaluation
    preds = logreg.predict_proba(S_te)[:, 1]
    res_out.append(roc_auc_score(yy_te, preds))


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
plt.savefig("images/sensivity.pdf", bbox_inches="tight")
plt.show()
# %%
