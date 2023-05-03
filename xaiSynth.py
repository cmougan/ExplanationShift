# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier, XGBRegressor
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from skshift import ExplanationShiftDetector

plt.style.use("seaborn-whitegrid")
from matplotlib import rcParams

rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8
rcParams.update({"font.size": 22})

# %%
# %%
# Data
N = 5_000
x1 = np.random.normal(1, 1, size=N)
x2 = np.random.normal(1, 1, size=N)
x3 = np.random.normal(1, 1, size=N)
X_tr_id = pd.DataFrame([x2, x3]).T
X_tr_id.columns = ["var%d" % (i + 1) for i in range(X_tr_id.shape[1])]

CASE_A = True
if CASE_A:
    y = x2 + x3
else:
    y = x3
model = XGBRegressor().fit(X_tr_id, y)


dp = []
res = []
exp_evolution = pd.DataFrame()
coefs = []
params = np.linspace(0, 1, 20)
for gamma in params:
    x3 = np.random.normal(1, 1, size=N)
    x4 = np.random.normal(1, 1, size=N)

    x12 = np.random.multivariate_normal([1, 1], [[1, gamma], [gamma, 1]], size=N)
    x1 = x12[:, 0]
    x2 = x12[:, 1]

    X_ood = pd.DataFrame([x2, x3]).T
    X_ood.columns = ["var%d" % (i + 1) for i in range(X_ood.shape[1])]

    # Inspector
    inspector = ExplanationShiftDetector(
        model, gmodel=LogisticRegression(penalty="l1", solver="saga")
    )
    inspector.fit_detector(X_tr_id, X_ood)

    # Concat X_tr and X_ood
    X_val = pd.concat([X_tr_id, X_ood], axis=0)
    Z_val = np.concatenate([np.zeros(X_tr_id.shape[0]), np.ones(X_ood.shape[0])])

    print(roc_auc_score(Z_val, model.predict(X_val)))

    coefs.append(inspector.detector.coef_)


# %%
# convert to dataframe
coefs_df = pd.DataFrame(coefs[0])

for i in range(len(coefs)):
    if i > 0:
        coefs_df = pd.concat([coefs_df, pd.DataFrame(coefs[i])], axis=0)
coefs_df.columns = ["var2", "var3"]
# %%
# Plot three coefficients
plt.plot(np.linspace(0, 1, 20), coefs_df["var2"], label="var2")
plt.fill_between(
    np.linspace(0, 1, 20),
    coefs_df["var2"] - 0.5 * coefs_df["var2"].std(),
    coefs_df["var2"] + 0.5 * coefs_df["var2"].std(),
    alpha=0.2,
)


plt.plot(np.linspace(0, 1, 20), coefs_df["var3"], label="var3")
plt.fill_between(
    np.linspace(0, 1, 20),
    coefs_df["var3"] - 0.5 * coefs_df["var3"].std(),
    coefs_df["var3"] + 0.5 * coefs_df["var3"].std(),
    alpha=0.2,
)


plt.xlabel("Correlation")
plt.legend()

plt.ylabel("Coefficients of the detector")
plt.title(
    "Coefficient evolution with correlation for case {0}".format(
        "Indirect" if CASE_A else "Uninformative"
    )
)
plt.ylim(-0.5, 35)
plt.savefig(
    "images/coef_evolution{0}.pdf".format("A" if CASE_A else "B"), bbox_inches="tight"
)
plt.show()

# %%
