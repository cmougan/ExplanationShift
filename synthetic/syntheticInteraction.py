# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

random.seed(0)
from xgboost import XGBRegressor, XGBClassifier
from scipy.stats import bernoulli, cauchy
import math
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

from category_encoders.target_encoder import TargetEncoder
from fairtools.utils import auc_group, metric_calculator, calculate_cm
from collections import defaultdict


# %%
## Create Variables
### Normal
sigma = 5
mean = [0, 0]
cov = [[sigma, 0], [0, sigma]]
samples = 50_000
x1, x2 = np.random.multivariate_normal(mean, cov, samples).T
xP = np.random.choice([1, 2], size=samples, p=[0.5, 0.5])
# %%
df = pd.DataFrame(data=[x1, x2]).T
df.columns = ["Var%d" % (i + 1) for i in range(df.shape[1])]
df["cat"] = xP
# %%
df["target"] = df["Var1"] * df["cat"] + np.random.normal(0, 0.1, samples)
df["target"] = np.where(df["target"] > np.mean(df["Var1"] * df["cat"]), 1, 0)
# %%
X_tr, X_te, y_tr, y_te = train_test_split(df.drop(columns="target"), df[["target"]])
# %%
pipe = Pipeline(
    [
        ("scaler", TargetEncoder()),
        (
            "model",
            XGBClassifier(
                verbosity=0, silent=True, random_state=42, use_label_encoder=False
            ),
        ),
    ]
)
pred_train = cross_val_predict(pipe, X_tr, y_tr.values.ravel(), cv=5)
pipe.fit(X_tr, y_tr.values.ravel())
# %%
# Extract metrics for each group
res = defaultdict()
for cat, num in X_te["cat"].value_counts().items():
    COL = "cat"
    GROUP1 = 1
    GROUP2 = cat
    res[cat] = [
        metric_calculator(
            modelo=pipe, data=X_te, truth=y_te, col=COL, group1=GROUP1, group2=GROUP2
        ),
        num,
    ]
res
# %%
# Fairnes Should Change
X_te_aux = X_te.copy()
X_te_aux["Var1"] = X_te["Var1"] + 1
y_te_aux = np.where(
    y_te["target"] > np.mean(X_te_aux["Var1"] * X_te_aux["cat"] + X_te_aux["Var2"]),
    1,
    0,
)
res = defaultdict()
for cat, num in X_te["cat"].value_counts().items():
    COL = "cat"
    GROUP1 = 1
    GROUP2 = cat
    res[cat] = [
        metric_calculator(
            modelo=pipe,
            data=X_te_aux,
            truth=y_te,
            col=COL,
            group1=GROUP1,
            group2=GROUP2,
        ),
        num,
    ]
res
# %%
# Fairnes should not change
X_te_aux = X_te.copy()
X_te_aux["Var2"] = X_te["Var2"] + 1
y_te_aux = np.where(y_te["target"] > np.mean(X_te_aux["Var1"] * X_te_aux["cat"]), 1, 0)
res = defaultdict()
for cat, num in X_te["cat"].value_counts().items():
    COL = "cat"
    GROUP1 = 1
    GROUP2 = cat
    res[cat] = [
        metric_calculator(
            modelo=pipe,
            data=X_te_aux,
            truth=y_te,
            col=COL,
            group1=GROUP1,
            group2=GROUP2,
        ),
        num,
    ]
res

# %%
plt.plot()
plt.xlabel(r"$ \it{\bf{\mu_{0}}}$ H  (mT)", fontsize=12, fontweight="bold")
# %%
