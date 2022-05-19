# %%
# General imports
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")
from scipy.stats import ks_2samp, kstest, wasserstein_distance
import seaborn as sns
import pandas as pd
import random
from collections import defaultdict

# Scikit Learn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR

# Specific packages
from xgboost import XGBRFClassifier, XGBRegressor, XGBClassifier
import shap
from tqdm import tqdm

# Home made code
import sys

sys.path.append("../")
from fairtools.xaiUtils import ShapEstimator
from fairtools.utils import psi

# Seeding
np.random.seed(0)
random.seed(0)
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
print("Train:", mean_squared_error(y_tr, preds_val))
print("Test:", mean_squared_error(y_te, preds_test))
print("OOD:", mean_squared_error(y_ood, preds_ood))

print("Feat 1")
print(ks_2samp(exp_ood["Shap1"], exp["Shap1"]))
print(ks_2samp(x11, x1))

print("Feat 2")
print(ks_2samp(exp_ood["Shap2"], exp["Shap2"]))
print(ks_2samp(x22, x2))

print("Target")
print(ks_2samp(y_te["target"], y_ood))
print(ks_2samp(preds_test, preds_ood))

# %%
## Can we learn xAI help to solve this issue?
################################
####### PARAMETERS #############
SAMPLE_FRAC = 1_000
ITERS = 2_000
# Init
train = defaultdict()
performance = defaultdict()
train_shap = defaultdict()

explainer = shap.Explainer(model)
shap_test = explainer(X_te)
shap_test = pd.DataFrame(shap_test.values, columns=X_tr.columns)

full = X_ood.copy()
full["target"] = y_ood
train_error = mean_squared_error(y_tr, preds_val)
# %%
# Loop to creat training data
for i in tqdm(range(0, ITERS), leave=False):
    row = []
    row_shap = []

    # Sampling
    aux = full.sample(n=SAMPLE_FRAC, replace=True)

    # Performance calculation
    preds = model.predict(aux.drop(columns="target"))
    preds = train_error - preds  # How much the preds differ from train
    performance[i] = mean_absolute_error(aux.target.values, preds)

    # Shap values calculation
    shap_values = explainer(aux.drop(columns="target"))
    shap_values = pd.DataFrame(shap_values.values, columns=X_tr.columns)

    for feat in X_tr.columns:
        ks = psi(X_te[feat], aux[feat])
        sh = psi(shap_test[feat], shap_values[feat])
        row.append(ks)
        row_shap.append(sh)

    train_shap[i] = row_shap
    train[i] = row

# %%
# Save results
train_df = pd.DataFrame(train).T
train_df.columns = X_tr.columns

train_shap_df = pd.DataFrame(train_shap).T
train_shap_df.columns = X_tr.columns
train_shap_df = train_shap_df.add_suffix("_shap")

# %%
# Estimators for the loop
estimators = defaultdict()
estimators["Dummy"] = DummyRegressor()
estimators["Linear"] = Pipeline(
    [("scaler", StandardScaler()), ("model", LinearRegression())]
)
estimators["Lasso"] = Pipeline([("scaler", StandardScaler()), ("model", Lasso())])
estimators["RandomForest"] = RandomForestRegressor(random_state=0)
estimators["XGBoost"] = XGBRegressor(verbose=0, random_state=0)
estimators["SVM"] = SVR()
estimators["MLP"] = MLPRegressor(random_state=0)
# %%
# Loop over different G estimators
for estimator in estimators:
    print(estimator)
    ## ONLY DATA
    X_train, X_test, y_train, y_test = train_test_split(
        train_df, performance, test_size=0.33, random_state=42
    )
    estimators[estimator].fit(X_train, y_train)
    print(
        "ONLY DATA", mean_absolute_error(estimators[estimator].predict(X_test), y_test)
    )

    #### ONLY SHAP
    X_train, X_test, y_train, y_test = train_test_split(
        train_shap_df, performance, test_size=0.33, random_state=42
    )
    estimators[estimator].fit(X_train, y_train)
    print(
        "ONLY SHAP", mean_absolute_error(estimators[estimator].predict(X_test), y_test)
    )

    ### SHAP + DATA
    X_train, X_test, y_train, y_test = train_test_split(
        pd.concat([train_shap_df, train_df], axis=1),
        performance,
        test_size=0.33,
        random_state=42,
    )
    estimators[estimator].fit(X_train, y_train)
    print(
        "SHAP + DATA",
        mean_absolute_error(estimators[estimator].predict(X_test), y_test),
    )
