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
import pdb
import shap
from sklearn.metrics import ndcg_score

# %%
data = GetData(type="real", datasets="ACSMobility")
X, y = data.get_state(state="CA", year="2018", N=20_000)


# %%
def c2st(x, y):
    # Convert to dataframes
    # Dinamic columns depending on lenght of x
    try:
        columns = [f"var_{i}" for i in range(x.shape[1])]
    except:
        columns = ["var_0"]
    if isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x.values, columns=columns)
        x["label"] = 0
        y = pd.DataFrame(y.values, columns=columns)
        y["label"] = 1
    else:
        x = pd.DataFrame(x, columns=columns)
        x["label"] = 0
        y = pd.DataFrame(y, columns=columns)
        y["label"] = 1

    # Concatenate
    df = pd.concat([x, y], axis=0)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(["label"], axis=1), df["label"], test_size=0.5, random_state=42
    )

    # Train classifier
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)

    # Evaluate AUC
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    return auc


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
preds_xgb = []
preds_log = []
input_classDrift = []
ndcgs = []


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
    aucs_xgb.append(detector.get_auc_val())

    # Prediction Shift - XGB
    preds_tr = detector.model.predict_proba(X_tr.drop(columns=["Race"]))[:, 1]
    preds_te = detector.model.predict_proba(X_new)[:, 1]
    preds_xgb.append(c2st(preds_tr, preds_te))

    # Explanation Shift Log
    detector = ExplanationShiftDetector(
        model=LogisticRegression(), gmodel=XGBClassifier(), masker=True
    )
    detector.fit(X_tr.drop(columns=["Race"]), y_tr, X_new)
    aucs_log.append(detector.get_auc_val())

    # Prediction Shift - Log
    preds_tr = detector.model.predict_proba(X_tr.drop(columns=["Race"]))[:, 1]
    preds_te = detector.model.predict_proba(X_new)[:, 1]
    preds_log.append(c2st(preds_tr, preds_te))

    # Classifier Drift
    input_classDrift.append(c2st(X_te.drop(columns="Race"), X_new))

    # Input KS Test
    cd = TabularDrift(X_tr.drop(columns="Race").values, p_val=0.05)
    input_ks.append(cd.predict(X_new.values)["data"]["distance"].mean())

    # NDCG - XGB
    ## Shap values in Train
    model = XGBClassifier().fit(X_tr.drop(columns="Race").values, y_tr)
    explainer = shap.Explainer(model)
    shap_values_tr = explainer(X_tr.drop(columns="Race").values)
    shap_df_tr = pd.DataFrame(shap_values_tr.values)

    ## Shap values in OOD
    explainer = shap.Explainer(model)
    shap_values_ood = explainer(X_new)
    shap_df_ood = pd.DataFrame(shap_values_ood.values)

    id = shap_df_tr.mean().sort_values(ascending=False).index.values
    nid = shap_df_ood.mean().sort_values(ascending=False).index.values
    ndcg = (
        1
        - ndcg_score(
            np.asarray([id]),
            np.asarray([nid]),
        )
        + 0.5
    )
    ndcgs.append(ndcg)


# %%
# Plot
plt.figure(figsize=(10, 6))

# XGB AUC
plt.plot(
    params,
    aucs_xgb,
    label=r"$g_\psi$ = Explanations, $f_\theta$ = XGB",
    color="darkblue",
    linewidth=2,
)
# ci = 1.96 * np.std(aucs_xgb) / np.sqrt(len(params))
# plt.fill_between(params, (aucs_xgb - ci), (aucs_xgb + ci), alpha=0.1)

# Log AUC
linewidth = 3
alpha = 0.2
plt.plot(
    params,
    aucs_log,
    label=r"$g_\psi$ = Explanations, $f_\theta$ = Log",
    color="lightblue",
    linestyle=":",
    marker="o",
    linewidth=2,
)

# Preds xgb
plt.plot(
    params,
    preds_xgb,
    label=r"$g_\Upsilon$ = Predictions, $f_\theta$ = XGB",
    color="darkgreen",
    linewidth=linewidth,
    alpha=alpha,
)
plt.plot(
    params,
    preds_log,
    label=r"$g_\Upsilon$ = Predictions, $f_\theta$ = Log",
    color="lightgreen",
    marker="o",
    linestyle=":",
    linewidth=linewidth,
)

# Classifier Drift
plt.plot(
    params,
    input_classDrift,
    label=r"$g_\phi$ = Input, $f_\theta$ = XGB",
    color="crimson",
    linewidth=linewidth,
    alpha=alpha,
)

# Classifier Drift
plt.plot(
    params,
    input_classDrift,
    label=r"$g_\phi$ = Input, $f_\theta$ = Log",
    color="lightcoral",
    linestyle="None",
    marker="o",
    linewidth=linewidth,
)
# NDCG XGB
plt.plot(
    params,
    ndcgs,
    label=r"$f_\theta$ = XGB, Exp. NDCG",
    color="darkorange",
    linewidth=linewidth,
    alpha=alpha,
)


plt.xlabel("Fraction of data from previously unseen group", fontsize=20)
# plt.ylabel("AUC")
plt.legend(fontsize=18)
plt.savefig("images/NewCategoryBenchmark.pdf", bbox_inches="tight")
# %%
## Correlation Coefficient
print("Correlation Coefficient")
print("XGB AUC:", np.corrcoef(params, aucs_xgb)[0, 1])
print("Log AUC:", np.corrcoef(params, aucs_log)[0, 1])
print("XGB Preds:", np.corrcoef(params, preds_xgb)[0, 1])
print("Log Preds:", np.corrcoef(params, preds_log)[0, 1])
print("Input KS:", np.corrcoef(params, input_ks)[0, 1])
print("Input Classifier Drift:", np.corrcoef(params, input_classDrift)[0, 1])

# %%
kkkk
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

c2st(X_te, X_new)
# %%
1 - np.array(ndcgs) + 0.5
# %%
