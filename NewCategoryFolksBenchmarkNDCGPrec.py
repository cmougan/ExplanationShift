# %%
# Import Folktables
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.style.use("seaborn-whitegrid")
rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8
rcParams.update({"font.size": 22})
import seaborn as sns
from sklearn.model_selection import train_test_split
from skshift import ExplanationShiftDetector
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from tqdm import tqdm
from tools.datasets import GetData
from alibi_detect.cd import TabularDrift
import shap
from sklearn.metrics import ndcg_score

# %%
# dataset = "ACSEmployment"
# dataset = "ACSIncome"
# dataset = "ACSTravelTime"
dataset = "ACSIncome"
data = GetData(type="real", datasets=dataset)
X, y = data.get_state(state="CA", year="2018", N=20_000)
# %%
df = X.copy()
df["y"] = y

aucs_xgb = []
aucs_log = []
input_ks = []
preds_xgb = []
preds_log = []
input_classDrift = []
ndcg_xgb = []
ndcg_log = []


for r in df["Race"].unique():
    df_tr = df[df["Race"] != r]
    # Train test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        df_tr.drop("y", axis=1), df_tr["y"], test_size=0.5, random_state=42
    )
    X_ood = df[df["Race"] == r].drop("y", axis=1)

    detector = ExplanationShiftDetector(
        model=XGBClassifier(), gmodel=LogisticRegression()
    )

    # Concatenate the training and validation sets
    n_samples = X_ood.shape[0] - int(0.8 * X_ood.shape[0])
    n_samples_1 = n_samples

    X_ = X_ood.loc[~X_ood.index.isin(X_ood.sample(n_samples).index)]
    X_new = X_te.sample(n_samples, replace=False).append(X_).drop(columns=["Race"])

    # Explanation Shift XGB
    detector = ExplanationShiftDetector(
        model=XGBClassifier(), gmodel=LogisticRegression()
    )
    detector.fit(X_tr.drop(columns=["Race"]), y_tr, X_new)
    aucs_xgb.append(detector.get_auc_val())

    # Explanation Shift Log
    detector = ExplanationShiftDetector(
        model=LogisticRegression(), gmodel=XGBClassifier(), masker=True
    )
    detector.fit(X_tr.drop(columns=["Race"]), y_tr, X_new)
    aucs_log.append(detector.get_auc_val())

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
    ndcg_xgb.append(ndcg)

    # NDCG - Log
    ## Shap values in Train
    model = LogisticRegression().fit(X_tr.drop(columns="Race").values, y_tr)
    explainer = shap.Explainer(model, X_tr.drop(columns="Race"))
    shap_values_tr = explainer(X_tr.drop(columns="Race").values)
    shap_df_tr = pd.DataFrame(shap_values_tr.values)

    ## Shap values in OOD
    explainer = shap.Explainer(model, X_tr.drop(columns="Race"))
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
    ndcg_log.append(ndcg)


# %%
# Plot
plt.figure(figsize=(10, 6))

# XGB AUC
sns.lineplot(
    x=params,
    y=aucs_xgb,
    label=r"Explanation Shift $f_\theta$ = XGB",
    color="darkblue",
    marker="o",
)

# Log AUC
sns.lineplot(
    x=params,
    y=aucs_log,
    label=r"Explanation Shift $f_\theta$ = Log",
    color="blue",
    linestyle="--",
)

# NDCG XGB
plt.plot(
    params,
    ndcg_xgb,
    label=r"(B3)Exp. NDCG $f_\theta$ = XGB",
    color="grey",
    marker="o",
)


# NDGC Log
sns.lineplot(
    x=params,
    y=ndcg_log,
    label=r"(B3)Exp. NDCG $f_\theta$ = Log",
    color="black",
    linestyle="--",
)

plt.xlabel("Fraction of data from previously unseen group", fontsize=22)
plt.ylabel("AUC")
plt.title("New Category Benchmark - {}".format(dataset))
plt.legend(fontsize=22)
plt.savefig(
    "images/NewCategoryBenchmarkNDCG{}.pdf".format(dataset), bbox_inches="tight"
)

# %%
