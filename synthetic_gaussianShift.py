# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

random.seed(0)
# Scikit Learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

plt.style.use("seaborn-whitegrid")
from matplotlib import rcParams

rcParams.update({"font.size": 12})
from xgboost import XGBRegressor
import shap

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
plt.title("Sensitivity to Covariate Shift of Classifier Two Sample Test")
plt.plot(iters, res_exp, label=r"Explanation Distribution $g_\psi$ (ours)", marker="o")
ci = 1.96 * np.std(res_exp) / np.sqrt(len(iters))
plt.fill_between(iters, (res_exp - ci), (res_exp + ci), alpha=0.1)

plt.plot(
    iters, res_inp, label=r"Input Distribution $g_\phi$ (B6)", marker="*", color="red"
)
ci = 1.96 * np.std(res_inp) / np.sqrt(len(iters))
plt.fill_between(iters, (res_inp - 2 * ci), (res_inp + 2 * ci), alpha=0.1, color="red")

plt.plot(
    iters,
    res_out,
    label=r"Prediction Distribution $g_\Upsilon$ (B7)",
    marker="x",
    color="black",
)
ci = 1.96 * np.std(res_out) / np.sqrt(len(iters))
plt.fill_between(iters, (res_out - ci), (res_out + ci), alpha=0.1, color="black")

plt.xlabel(r"Correlation coefficient $\rho$")
plt.ylabel(r"AUC")
plt.legend()
plt.tight_layout()
plt.savefig("images/sensivity.pdf", bbox_inches="tight")
plt.show()
# %%
# Accountability Explanation Space
S_tr, S_te, yy_tr, yy_te = train_test_split(
    exp_space.drop(columns="label"),
    exp_space[["label"]],
    random_state=0,
    test_size=0.5,
    stratify=exp_space[["label"]],
)
logreg.fit(S_tr, yy_tr)
# %%
explainer = shap.LinearExplainer(
    logreg, S_te, feature_dependence="correlation_dependent"
)
shap_values = explainer(S_te)
plt.figure()
shap.plots.bar(shap_values, show=False)
plt.savefig("images/xAIMultivariate.pdf", bbox_inches="tight")
plt.show()

# %%
## Correlation coefficient
print('Correlation explanation',np.corrcoef(iters, res_exp)[0,1])
print('Correlation input',np.corrcoef(iters, res_inp)[0,1])
print('Correlation output',np.corrcoef(iters, res_out)[0,1])

# %%
#Correlation explanation 0.9971629548286309
#Correlation input 0.18054841217224993
#Correlation output 0.9694638213466882