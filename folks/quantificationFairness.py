# %%
import warnings

warnings.filterwarnings("ignore")
from folktables import (
    ACSDataSource,
    ACSIncome,
    ACSEmployment,
    ACSMobility,
    ACSPublicCoverage,
    ACSTravelTime,
)
import pandas as pd
from collections import defaultdict
import seaborn as sns
from scipy.stats import wasserstein_distance, kstest

sns.set_style("whitegrid")
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import kstest

# Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
)

from sklearn.model_selection import train_test_split

# Specific packages
from xgboost import XGBRegressor, XGBClassifier
import shap
from tqdm import tqdm


# Home made code
import sys
import os
import traceback

sys.path.append("../")
sys.path.append("/home/jupyter/ExplanationShift")

# Seeding
np.random.seed(0)
random.seed(0)
# %%
# Load data
data_source = ACSDataSource(survey_year="2014", horizon="1-Year", survey="person")
ca_data = data_source.get_data(states=["HI"], download=True)
ca_features, ca_labels, ca_group = ACSIncome.df_to_numpy(ca_data)
## Conver to DF
ca_features = pd.DataFrame(ca_features, columns=ACSIncome.features)
ca_features["group"] = ca_group
# %%
states = [
    "CA",
    "OK",
    "TN",
    "MI",
    "PR",
    "TN",
    "CT",
    "OH",
    "NE",
    "IL",
    "FL",
    "ID",
]
nooo = [
    "PA",
    "KS",
    "IA",
    "KY",
    "NY",
    "LA",
    "TX",
    "UT",
    "OR",
    "ME",
    "NJ",
    "DE",
    "MN",
    "WI",
    "CA",
    "MO",
    "MD",
    "NV",
    "HI",
    "IN",
    "WV",
    "MT",
    "WY",
    "ND",
    "SD",
    "GA",
    "NM",
    "AZ",
    "VA",
    "MA",
    "NC",
    "SC",
    "VT",
    "AR",
    "WA",
    "CO",
    "NH",
    "MS",
    "AK",
    "RI",
    "AL",
    "PR",
]


# %%
# Modeling
# model = XGBClassifier(verbosity=0, silent=True, use_label_encoder=False, njobs=1)
model = LogisticRegression()
# Train on CA data
preds_ca = cross_val_predict(
    model, ca_features, ca_labels, cv=3, method="predict_proba"
)[:, 1]
model.fit(ca_features, ca_labels)

# Fairness in training data
white_tpr = np.mean(preds_ca[(ca_labels == 1) & (ca_group == 1)])
black_tpr = np.mean(preds_ca[(ca_labels == 1) & (ca_group == 2)])
eof_tr = white_tpr - black_tpr

# %%
## Can we learn to solve this issue?
################################
####### PARAMETERS #############
SAMPLE_FRAC = 1000
ITERS = 5_000
THRES = 0.05
GROUP = 1
# Init
train_error = accuracy_score(ca_labels, np.round(preds_ca))
train_error_acc = accuracy_score(ca_labels, np.round(preds_ca))

# xAI Train
# explainer = shap.Explainer(model)
explainer = shap.LinearExplainer(
    model, ca_features, feature_dependence="correlation_dependent"
)
shap_test = explainer(ca_features)
shap_test = pd.DataFrame(shap_test.values, columns=ca_features.columns)


def my_explode(data):
    """
    Explode a dataframe with list in columns into a dataframe.
    Loses the name columns
    """
    aux = pd.DataFrame()
    for col in data.columns:
        aux = pd.concat([aux, pd.DataFrame(data[col].to_list())], axis=1)
    return aux


## Meta data function
def create_meta_data(test, samples, boots):
    # Init
    train = defaultdict()
    train_target_shift = defaultdict()
    performance = defaultdict()
    train_shap = defaultdict()

    for i in tqdm(range(0, boots), leave=False, desc="Test Bootstrap", position=1):
        # Initiate
        row = []
        row_target_shift = []
        row_shap = []

        # Sampling
        aux = test.sample(n=SAMPLE_FRAC, replace=True)

        # Performance calculation
        preds = model.predict_proba(aux.drop(columns=["target"]))[:, 1]

        performance[i] = eof_tr - (
            np.mean(preds[(aux.target == 1) & (aux.group == 1)])
            - np.mean(preds[(aux.target == 1) & (aux.group == 2)])
        )

        # Shap values calculation
        shap_values = explainer(aux.drop(columns=["target"]))
        shap_values = pd.DataFrame(shap_values.values, columns=ca_features.columns)

        for feat in ca_features.columns:
            ks = [
                np.mean(ca_features[feat][ca_features["group"] == 1])
                - np.mean(aux[feat][aux["group"] == 1]),
                np.mean(ca_features[feat][ca_features["group"] == 2])
                - np.mean(aux[feat][aux["group"] == 2]),
            ]

            sh = [
                np.mean(shap_test[feat][ca_features.reset_index()["group"] == 1])
                - np.mean(shap_values[feat][aux.reset_index()["group"] == 1].mean()),
                np.mean(shap_test[feat][ca_features.reset_index()["group"] == 2])
                - np.mean(shap_values[feat][aux.reset_index()["group"] == 2]),
            ]

            row.append(ks)
            row_shap.append(sh)
        # Target shift
        ks_target_shift = [
            kstest(
                preds_ca[ca_features.reset_index()["group"] == 1],
                preds[mi_features.reset_index()["group"] == 1],
            ).statistic,
            kstest(
                preds_ca[ca_features.reset_index()["group"] == 1],
                preds[mi_features.reset_index()["group"] == 1],
            ).statistic,
        ]
        row_target_shift.append(ks_target_shift)
        # Save results
        train_shap[i] = row_shap
        train[i] = row
        train_target_shift[i] = row_target_shift

    ## Train (previous test)
    train_df = pd.DataFrame(train).T
    train_df.columns = ca_features.columns

    train_shap_df = pd.DataFrame(train_shap).T
    train_shap_df.columns = ca_features.columns
    train_shap_df = train_shap_df.add_suffix("_shap")

    train_target_shift_df = pd.DataFrame(train_target_shift, index=[0]).T
    train_target_shift_df.columns = ["target"]

    # On the target
    performance = pd.DataFrame(performance, index=[0]).T.values
    return (
        train_df,
        train_shap_df,
        train_target_shift_df,
        performance.squeeze(),
    )


# %%
res = defaultdict(list)
# Loop throug each state
for state in tqdm(states):
    print(state)
    try:
        ## Lets add the target to ease the sampling
        data_source = ACSDataSource(
            survey_year="2018", horizon="1-Year", survey="person"
        )
        mi_data = data_source.get_data(states=[state], download=True)
        mi_features, mi_labels, mi_group = ACSIncome.df_to_numpy(mi_data)
        mi_features = pd.DataFrame(mi_features, columns=ACSIncome.features)
        mi_full = mi_features.copy()
        mi_full["group"] = mi_group
        mi_full["target"] = mi_labels
        # mi_full = mi_full[mi_full["group"] == GROUP]

        input_tr, shap_tr, output_tr, model_error_tr_ = create_meta_data(
            mi_full, SAMPLE_FRAC, ITERS
        )
        input_tr = my_explode(input_tr)
        shap_tr = my_explode(shap_tr)
        output_tr = my_explode(output_tr)

        # Convert in classification

        model_error_tr = np.where(
            model_error_tr_ < np.quantile(model_error_tr_, q=THRES), 1, 0
        )
        # Input
        X_tr, X_te, y_tr, y_te = train_test_split(
            input_tr, model_error_tr, test_size=0.3, random_state=42
        )
        clf = LogisticRegression()
        clf.fit(X_tr, y_tr)
        input_results = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
        # Shap
        X_tr, X_te, y_tr, y_te = train_test_split(
            shap_tr, model_error_tr, test_size=0.3, random_state=42
        )
        clf.fit(X_tr, y_tr)
        shap_results = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
        # Output
        X_tr, X_te, y_tr, y_te = train_test_split(
            output_tr, model_error_tr, test_size=0.3, random_state=42
        )
        clf.fit(X_tr, y_tr)
        output_results = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])

        res[state] = [input_results, shap_results, output_results]
    except Exception as e:
        print(state, "failed")
        print(traceback.format_exc())
# %%
df = pd.DataFrame(data=res).T
df.columns = ["Input Shift", "Explanation Shift", "Output Shift"]
df.to_csv("results_fairOne.csv")
# %%
plt.figure()
sns.barplot(y=df.mean().values, x=df.columns, ci=0.1, capsize=0.2, palette="RdBu_r")
plt.axhline(0.5, color="black", linestyle="--")
plt.ylim(0.4, 0.9)
plt.ylabel("AUC")
plt.savefig("images/shap_shift_fairOne.png")
plt.show()
# %%
aux = df.copy()
best = []
for state in df.index.unique():
    aux_state = aux[aux.index == state]
    # Estimators
    input = aux_state["Input Shift"].values
    output = aux_state["Output Shift"].values
    exp = aux_state["Explanation Shift"].values

    d = {
        "Distribution Shift": input,
        "Prediction Shift": output,
        "Explanation Shift": exp,
    }

    best.append([state, max(d, key=d.get)])

best = pd.DataFrame(best, columns=["state", "data"])
# %%
import plotly.express as px

fig = px.choropleth(
    best,
    locations="state",
    locationmode="USA-states",
    color="data",
    # color_continuous_scale="Reds",
    scope="usa",
    # hover_name="state",
    # hover_data=["error_ood"],
)
fig.show()
fig.write_image("images/best_method_fairOne.png")
# %%
print(df.mean())
print(df.median())
print(df.std())
