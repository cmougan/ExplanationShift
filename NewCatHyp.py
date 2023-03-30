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
rcParams.update({"font.size": 12})
import seaborn as sns
from sklearn.model_selection import train_test_split
from nobias import ExplanationShiftDetector
from xgboost import XGBRegressor, XGBClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from tools.datasets import GetData
from alibi_detect.cd import ChiSquareDrift, TabularDrift, ClassifierDrift

# %%
data = GetData(type="real", datasets="ACSIncome")
X, y = data.get_state(state="CA", year="2018", N=20_000)
# %%
df = X.copy()
df["y"] = y

r = 8
df_tr = df[df["Race"] != r]
# Train test split
X_tr, X_te, y_tr, y_te = train_test_split(
    df_tr.drop("y", axis=1), df_tr["y"], test_size=0.5, random_state=42
)
X_ood = df[df["Race"] == r].drop("y", axis=1)

detector = ExplanationShiftDetector(model=XGBClassifier(), gmodel=LogisticRegression())


i = 0.5
n_samples = X_ood.shape[0] - int(i * X_ood.shape[0])
n_samples_1 = n_samples

X_ = X_ood.loc[~X_ood.index.isin(X_ood.sample(n_samples).index)]
X_new = X_te.sample(n_samples, replace=False).append(X_).drop(columns=["Race"])
# %%
# XGB
xgb_list = []
xgb_param = np.linspace(1, 50, 10)
for i in xgb_param:
    detector = ExplanationShiftDetector(
        model=XGBClassifier(max_depth=int(i)), gmodel=LogisticRegression()
    )
    detector.fit(X_tr.drop(columns=["Race"]), y_tr, X_new)

    value = detector.get_auc_val()

    xgb_list.append(value)
# %%
# Decision Tree
dt_list = []
dt_param = np.linspace(1, 10, 9)
for i in dt_param:
    detector = detector = ExplanationShiftDetector(
        model=DecisionTreeRegressor(max_depth=int(i)), gmodel=LogisticRegression()
    )
    detector.fit(X_tr.drop(columns=["Race"]), y_tr, X_new)

    value = detector.get_auc_val()
    dt_list.append(value)
# %%
# Plot
plt.figure(figsize=(10, 8))
plt.plot(xgb_param, xgb_list, label="XGB")
plt.plot(dt_param, dt_list, label="Decision Tree")
plt.xlabel("Max Depth")
plt.ylabel("AUC")
plt.legend()
plt.show()


# %%
