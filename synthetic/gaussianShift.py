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

sys.path.append("../")
from fairtools.xaiUtils import ShapEstimator

# %%
## Create variables
### Normal
sigma = 5
mean = [0, 0]
cov = [[sigma, 0], [0, sigma]]
samples = 50_000
x1, x2 = np.random.multivariate_normal(mean, cov, samples).T
# Different values
mean = [0, 0]
out = 3
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
print(mean_squared_error(y_tr, preds_val))
print(mean_squared_error(y_te, preds_test))
print(mean_squared_error(y_ood, preds_ood))
# %%
print("Feat 1")
print(ks_2samp(exp_ood["Shap1"], exp["Shap1"]))
print(ks_2samp(x11, x1))
# %%
print("Feat 2")
print(ks_2samp(exp_ood["Shap2"], exp["Shap2"]))
print(ks_2samp(x22, x2))
# %%
print("Target")
print(ks_2samp(y_te["target"], y_ood))
print(ks_2samp(preds_test, preds_ood))

# %%
## Does xAI help to solve this issue?
## Shap Estimator
se = ShapEstimator(model=model)
shap_pred_tr = cross_val_predict(se, X_tr, y_tr, cv=3)
shap_pred_tr = pd.DataFrame(shap_pred_tr, columns=X_tr.columns)
shap_pred_tr = shap_pred_tr.add_suffix("_shap")

se.fit(X_tr, y_tr)
# %%
# Estimators to use
estimators = defaultdict()
estimators["Linear"] = Pipeline([("scaler", StandardScaler()), ("model", Lasso())])
estimators["RandomForest"] = RandomForestRegressor(random_state=0)
estimators["XGBoost"] = XGBRegressor(random_state=0)
estimators["MLP"] = MLPRegressor(random_state=0)
# %%
for estimator in estimators:
    print(estimator)
    clf = estimators[estimator]
    error_tr = y_tr.target.values - preds_val
    preds_tr_shap = cross_val_predict(clf, shap_pred_tr, error_tr, cv=3)
    clf.fit(shap_pred_tr, error_tr)

    ## Test Set
    shap_pred_te = se.predict(X_te)
    shap_pred_te = pd.DataFrame(shap_pred_te, columns=X_te.columns)
    shap_pred_te = shap_pred_te.add_suffix("_shap")

    error_te = y_te.target.values - preds_test
    preds_te_shap = clf.predict(shap_pred_te)

    ## OOD Set
    shap_pred_ood = se.predict(X_ood)
    shap_pred_ood = pd.DataFrame(shap_pred_ood, columns=X_te.columns)
    shap_pred_ood = shap_pred_ood.add_suffix("_shap")

    error_ood = y_ood - preds_ood
    preds_ood_shap = clf.predict(shap_pred_ood)
    # Original error
    print("Original error, should remain invariant to diff estimators")
    print(mean_squared_error(y_tr, preds_val))
    print(mean_squared_error(y_te, preds_test))
    print(mean_squared_error(y_ood, preds_ood))

    ## Only SHAP
    print("Only SHAP")
    print(mean_squared_error(error_tr, preds_tr_shap))
    print(mean_squared_error(error_te, preds_te_shap))
    print(mean_squared_error(error_ood, preds_ood_shap))

    ## Only data
    print("Only data")
    preds_tr_shap = cross_val_predict(clf, X_tr, error_tr, cv=3)
    clf.fit(X_tr, error_tr)
    preds_te_shap = clf.predict(X_te)
    preds_ood_shap = clf.predict(X_ood)
    print(mean_squared_error(error_tr, preds_tr_shap))
    print(mean_squared_error(error_te, preds_te_shap))
    print(mean_squared_error(error_ood, preds_ood_shap))

    print("Shap + Data")
    ## SHAP + Data
    tr_full = pd.concat(
        [shap_pred_tr.reset_index(drop=True), X_tr.reset_index(drop=True)], axis=1
    )
    te_full = pd.concat(
        [shap_pred_te.reset_index(drop=True), X_te.reset_index(drop=True)], axis=1
    )
    ood_full = pd.concat(
        [shap_pred_ood.reset_index(drop=True), X_ood.reset_index(drop=True)], axis=1
    )

    preds_tr_shap = cross_val_predict(clf, tr_full, error_tr, cv=3)
    clf.fit(tr_full, error_tr)
    preds_te_shap = clf.predict(te_full)
    preds_ood_shap = clf.predict(ood_full)
    print(mean_squared_error(error_tr, preds_tr_shap))
    print(mean_squared_error(error_te, preds_te_shap))
    print(mean_squared_error(error_ood, preds_ood_shap))

# %%
# %%
model = MLPRegressor(random_state=0)
preds_val = cross_val_predict(model, X_tr, y_tr, cv=3)
model.fit(X_tr, y_tr)
preds_test = model.predict(X_te)
print(mean_squared_error(preds_test, y_te))
print(mean_squared_error(model.predict(X_ood), y_ood))
# That would be the best result possible
