# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.stats import ks_2samp
from tqdm import tqdm
from xgboost import XGBRegressor
import shap

from matplotlib import rcParams


rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8
rcParams.update({"font.size": 16})
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.grid"] = True
sns.set(font_scale=2)
sns.set_style("whitegrid")
plt.style.use("seaborn-whitegrid")
# %%
N = 5_000
gamma = 1
epsilon = 1
fair = []
for gamma in tqdm(np.linspace(0, 5, 50)):
    x11 = np.random.normal(0, 1, N)
    x12 = np.random.normal(0, epsilon, N)
    x21 = np.random.normal(0, 1, N)
    x22 = np.random.normal(0, epsilon, N)

    a1 = np.repeat(1, N)
    a2 = np.repeat(-1, N)
    y1 = np.where(x11 + x21 + np.random.normal(0, 0.1, N) > 0, 1, 0)
    y2 = np.where(gamma + x21 + x22 + np.random.normal(0, 0.1, N) < 0, 1, 0)
    x1 = np.append(x11, x12)
    x2 = np.append(x21, x22)
    a = np.append(a1, a2)
    y = np.append(y1, y2)

    X = pd.DataFrame([x1, x2, a]).T
    X.columns = ["Var%d" % (i + 1) for i in range(X.shape[1])]
    X["target"] = y

    X_tr, X_te, y_tr, y_te = train_test_split(
        X.drop(columns="target"), X["target"], test_size=0.33, random_state=42
    )
    A_tr = X_tr["Var3"]
    A_te = X_te["Var3"]

    model = LogisticRegression()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    # xAI
    explainer = shap.LinearExplainer(
        model, X_tr, feature_perturbation="correlation_dependent"
    )
    shap_test = explainer(X_te)
    shap_test = pd.DataFrame(shap_test.values, columns=X_te.columns)

    # EOF
    white_tpr = np.mean(preds[(y_te == 1) & (A_te == 1)])
    black_tpr = np.mean(preds[(y_te == 1) & (A_te == -1)])

    # DP
    dp = ks_2samp(preds[A_te == 1], preds[A_te == -1]).statistic
    fair.append([gamma, white_tpr - black_tpr, dp, shap_test["Var3"].mean()])
# %%
# sns.set_style("whitegrid")
res = pd.DataFrame(fair, columns=["Gamma", "Fairness", "DP", "SHAP"])
plt.figure()
sns.scatterplot(x="Gamma", y="Fairness", data=res, label="EOF")
sns.scatterplot(x="Gamma", y="SHAP", data=res, label="SHAP")
sns.scatterplot(x="Gamma", y="DP", data=res, label="DP")
plt.ylabel("")
plt.legend()
plt.savefig("images/syntheticFairnes.png")
plt.show()
