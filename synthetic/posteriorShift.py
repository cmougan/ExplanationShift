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

# %%
df = pd.DataFrame(data=[x1, x2]).T
df.columns = ["Var%d" % (i + 1) for i in range(df.shape[1])]
# df["target"] = np.where(df["Var1"] * df["Var2"] > 0, 1, 0)
df["target"] = 2 * df["Var1"] + df["Var2"] + np.random.normal(0, 0.1, samples)
df["target2"] = df["Var1"] + 2 * df["Var2"] + np.random.normal(0, 0.1, samples)
# %%
## Fit our ML model
X_tr, X_te, y_tr, y_te = train_test_split(
    df.drop(columns=["target", "target2"]), df[["target"]]
)
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    df.drop(columns=["target", "target2"]), df[["target2"]]
)
# %%
# Model F
model = XGBRegressor(random_state=0)
preds_val = cross_val_predict(model, X_tr, y_tr, cv=3)
model.fit(X_tr, y_tr)
preds_test = model.predict(X_te)
# Model G
model2 = XGBRegressor(random_state=0)
preds_val2 = cross_val_predict(model2, X_tr2, y_tr2, cv=3)
model2.fit(X_tr2, y_tr2)
preds_test2 = model2.predict(X_te2)
# %%
## Real explanation F
explainer = shap.Explainer(model)
shap_values = explainer(X_te)
exp = pd.DataFrame(
    data=shap_values.values, columns=["Shap%d" % (i + 1) for i in range(2)]
)
# %%
## OOD Data + OODexplanation
explainer2 = shap.Explainer(model2)
shap_values2 = explainer2(X_te2)
exp2 = pd.DataFrame(
    data=shap_values2.values, columns=["Shap%d" % (i + 1) for i in range(2)]
)
# %%
# Original error
print("Model F")
print("Train:", mean_squared_error(y_tr, preds_val))
print("Test:", mean_squared_error(y_te, preds_test))

print("Model G")
print("Train:", mean_squared_error(y_tr2, preds_val2))
print("Test:", mean_squared_error(y_te2, preds_test2))

print("Feat 1")
print("Shap", ks_2samp(exp2["Shap1"], exp["Shap1"]))
print(ks_2samp(x1, x1))

print("Feat 2")
print("Shap", ks_2samp(exp2["Shap2"], exp["Shap2"]))
print(ks_2samp(x2, x2))

print("Target")
print(ks_2samp(y_te["target"], y_te2["target2"]))

# %%
