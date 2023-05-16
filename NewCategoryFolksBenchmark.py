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
rcParams.update({"font.size": 16})
import seaborn as sns
from sklearn.model_selection import train_test_split
from nobias import ExplanationShiftDetector
from xgboost import XGBRegressor, XGBClassifier
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

# Concatenate the training and validation sets
params = np.linspace(0.05, 0.99, 10)

aucs_xgb = []
aucs_log = []
input_ks = []
input_classDrift = []


for i in tqdm(params):
    n_samples = X_ood.shape[0] - int(i * X_ood.shape[0])
    n_samples_1 = n_samples

    X_ = X_ood.loc[~X_ood.index.isin(X_ood.sample(n_samples).index)]
    X_new = X_te.sample(n_samples, replace=False).append(X_).drop(columns=["Race"])

    # Explanation Shift XGB
    detector = ExplanationShiftDetector(
        model=XGBClassifier(), gmodel=LogisticRegression()
    )
    detector.fit(X_tr.drop(columns=["Race"]), y_tr, X_new)
    aucs_xgb.append(2 * detector.get_auc_val() - 1)

    # Explanation Shift Log
    detector = ExplanationShiftDetector(
        model=LogisticRegression(), gmodel=XGBClassifier(), masker=True
    )
    detector.fit(X_tr.drop(columns=["Race"]), y_tr, X_new)
    aucs_log.append(2 * detector.get_auc_val() - 1)

    # Classifier Drift
    classDrift = ClassifierDrift(
        x_ref=X_te.drop(columns="Race").values,
        model=detector.model,
        backend="sklearn",
        n_folds=3,
    )
    input_classDrift.append(classDrift.predict(x=X_new.values)["data"]["distance"])

    # Input KS Test
    cd = TabularDrift(X_tr.drop(columns="Race").values, p_val=0.05)
    input_ks.append(cd.predict(X_new.values)["data"]["distance"].mean())

# %%
# Plot
plt.figure(figsize=(10, 6))

# XGB AUC
plt.plot(params, aucs_xgb, label="Exp. Shift XGB", color="darkblue", linewidth=2)
# ci = 1.96 * np.std(aucs_xgb) / np.sqrt(len(params))
# plt.fill_between(params, (aucs_xgb - ci), (aucs_xgb + ci), alpha=0.1)

# Log AUC
linewidth = 3
alpha = 0.2
plt.plot(
    params,
    aucs_log,
    label="Exp. Shift Log",
    color="lightblue",
    linestyle=":",
    marker="o",
    linewidth=2,
)

# Input KS Test
plt.plot(
    params,
    input_ks,
    label="KS-Test XGB",
    color="darkgreen",
    linewidth=linewidth,
    alpha=alpha,
)
plt.plot(
    params,
    input_ks,
    label="KS-Test Log",
    color="lightgreen",
    linestyle="None",
    marker="o",
    linewidth=linewidth,
)

# Classifier Drift
plt.plot(
    params,
    input_classDrift,
    label="Classifier Drift - XGB",
    color="crimson",
    linewidth=linewidth,
    alpha=alpha,
)

# Classifier Drift
plt.plot(
    params,
    input_classDrift,
    label="Classifier Drift - Log",
    color="lightcoral",
    linestyle="None",
    marker="o",
    linewidth=linewidth,
)


plt.xlabel("Fraction of data from previously unseen group")
plt.ylabel("Impact of the Distribution Shift on the Model")
plt.legend()
plt.savefig("images/NewCategoryBenchmark.pdf", bbox_inches="tight")
# %%

# %%
# Some Fariness metrics
all_preds = detector.model.predict_proba(df.drop(columns=["y", "Race"]))[:, 1]
black_preds = detector.model.predict_proba(
    df[df["Race"] == 2].drop(columns=["y", "Race"])
)[:, 1]
asian_preds = detector.model.predict_proba(
    df[df["Race"] == 5].drop(columns=["y", "Race"])
)[:, 1]
other_preds = detector.model.predict_proba(
    df[df["Race"] == 8].drop(columns=["y", "Race"])
)[:, 1]
mixed_preds = detector.model.predict_proba(
    df[df["Race"] == 9].drop(columns=["y", "Race"])
)[:, 1]
# Demographic Parity
print("Black:", wasserstein_distance(all_preds, black_preds))
print("Asian:", wasserstein_distance(all_preds, asian_preds))
print("Other:", wasserstein_distance(all_preds, other_preds))
print("Mixed:", wasserstein_distance(all_preds, mixed_preds))

# %%
# Large Benchmark


i = 0.5
n_samples = X_ood.shape[0] - int(i * X_ood.shape[0])
n_samples_1 = n_samples
params = np.linspace(0.05, 0.99, 5)

X_ = X_ood.loc[~X_ood.index.isin(X_ood.sample(n_samples).index)]
X_new = X_te.sample(n_samples, replace=False).append(X_).drop(columns=["Race"])

# Explanation Shift XGB
# Loop over all estimators
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier

list_estimator = [
    XGBClassifier(),
    LogisticRegression(),
    Lasso(),
    Ridge(),
    RandomForestRegressor(),
    DecisionTreeRegressor(),
    MLPRegressor(),
]
list_detector = [
    XGBClassifier(),
    LogisticRegression(),
    SVC(probability=True),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    MLPClassifier(),
]

res = pd.DataFrame(index=range(len(list_estimator)), columns=range(len(list_estimator)))
for i, estimator in tqdm(enumerate(list_estimator)):
    for j, gmodel in enumerate(list_detector):
        detector = ExplanationShiftDetector(model=estimator, gmodel=gmodel, masker=True)
        try:
            detector.fit(X_tr.drop(columns=["Race"]), y_tr, X_new)

            value = detector.get_auc_val()
            res.at[i, j] = value
        # Catch errors and print
        except Exception as e:
            print(e)
            res.at[i, j] = np.nan
            print("Error")
            print("Estimator: ", estimator.__class__.__name__)
            print("Detector: ", gmodel.__class__.__name__)

# %%
res.index = [estimator.__class__.__name__ for estimator in list_estimator]
res.columns = [estimator.__class__.__name__ for estimator in list_estimator]
# %%
res.dropna().astype(float).round(3).to_csv("results/ExplanationShift.csv")
# %%
