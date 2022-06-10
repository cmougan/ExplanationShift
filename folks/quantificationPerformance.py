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
from scipy.stats import kstest, wasserstein_distance
import seaborn as sns

sns.set_style("whitegrid")
import numpy as np
import random
import sys
import matplotlib.pyplot as plt

# Scikit-Learn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# Specific packages
from xgboost import XGBRegressor, XGBClassifier
import shap
from tqdm import tqdm


# Home made code
import sys

sys.path.append("../")
from fairtools.utils import loop_estimators_fairness, psi, loop_estimators
from ATC_opt import ATC

# Seeding
np.random.seed(0)
random.seed(0)
# %%
# Load data
data_source = ACSDataSource(survey_year="2014", horizon="1-Year", survey="person")
ca_data = data_source.get_data(states=["CA"], download=True)
data_source = ACSDataSource(survey_year="2016", horizon="1-Year", survey="person")
mi_data = data_source.get_data(states=["HI"], download=True)
# %%
states = [
    "MI",
    "TN",
    "CT",
    "OH",
    "NE",
    "IL",
    "FL",
]

nooo = [
    "OK",
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
    "ID",
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
    "AA",
    "NC",
    "SC",
    "DC",
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

data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")

# %%
ca_features, ca_labels, ca_group = ACSEmployment.df_to_numpy(ca_data)
mi_features, mi_labels, mi_group = ACSEmployment.df_to_numpy(mi_data)

## Conver to DF
ca_features = pd.DataFrame(ca_features, columns=ACSEmployment.features)
mi_features = pd.DataFrame(mi_features, columns=ACSEmployment.features)
# %%
# Modeling
# model = XGBClassifier(verbosity=0, silent=True, use_label_encoder=False, njobs=1)
model = LogisticRegression()
# Train on CA data
preds_ca = cross_val_predict(
    model, ca_features, ca_labels, cv=3, method="predict_proba"
)[:, 1]
model.fit(ca_features, ca_labels)
# Test on MI data
preds_mi = model.predict_proba(mi_features)[:, 1]
# Threshold classifier
atc = ATC()
atc.fit(model.predict_proba(ca_features), ca_labels)

# %%
## Can we learn to solve this issue?
################################
####### PARAMETERS #############
SAMPLE_FRAC = 1_000
ITERS = 5_00
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

# Lets add the target to ease the sampling
mi_full = mi_features.copy()
mi_full["group"] = mi_group
mi_full["target"] = mi_labels


def create_meta_data(test, samples, boots):
    # Init
    train = defaultdict()
    train_target_shift = defaultdict()
    performance = defaultdict()
    train_shap = defaultdict()
    atc_scores = defaultdict()
    for i in tqdm(range(0, boots), leave=False, desc="Test Bootstrap", position=1):
        # Initiate
        row = []
        row_target_shift = []
        row_shap = []

        # Sampling
        aux = test.sample(n=samples, replace=True)

        # Performance calculation
        preds = model.predict(aux.drop(columns=["target", "group"]))
        performance[i] = train_error - accuracy_score(aux.target, preds)

        # ATC
        atc_scores[i] = (
            atc.predict(model.predict_proba(aux.drop(columns=["target", "group"])))
            / 100
            - train_error_acc
        )
        # Shap values calculation
        shap_values = explainer(aux.drop(columns=["target", "group"]))
        shap_values = pd.DataFrame(shap_values.values, columns=ca_features.columns)

        for feat in ca_features.columns:
            # Michigan
            ks = ca_features[feat].mean() - aux[feat].mean()
            sh = shap_test[feat].mean() - shap_values[feat].mean()

            row.append(ks)
            row_shap.append(sh)
        # Target shift
        ks_target_shift = preds_ca.mean() - preds.mean()
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
        atc_scores,
    )


input_tr, shap_tr, output_tr, model_error_tr_, atc_scores = create_meta_data(
    mi_full, SAMPLE_FRAC, ITERS
)
# %%
# Convert in classification
model_error_tr = np.where(model_error_tr_ < -0.02, 1, 0)
sns.kdeplot(model_error_tr)
# %%
# Input
X_tr, X_te, y_tr, y_te = train_test_split(
        input_tr, model_error_tr, test_size=0.3, random_state=42
    )
dummy = DummyRegressor(strategy="constant", constant=0).fit(X_tr, y_tr)
print("Dummy", roc_auc_score(y_te, dummy.predict(X_te)))
clf = LogisticRegression()

clf.fit(X_tr, y_tr)
print("Input", roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))

# Shap
X_tr, X_te, y_tr, y_te = train_test_split(
        shap_tr, model_error_tr, test_size=0.3, random_state=42)
clf = LogisticRegression()
clf.fit(X_tr, y_tr)
print("Shap", roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))

# Output
X_tr, X_te, y_tr, y_te = train_test_split(
        output_tr, model_error_tr, test_size=0.3, random_state=42
    )
clf = LogisticRegression()
clf.fit(X_tr, y_tr)
print("Output", roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))
# ATC
print("ATC", roc_auc_score(model_error_tr,np.where(pd.DataFrame(atc_scores.values(),columns=['values']).values<-0.02,1,0)))

# %%
estimators = defaultdict()
input_data = defaultdict()
shap_data = defaultdict()
output_data = defaultdict()
model_error_data = defaultdict()
atc_scores_data = defaultdict()
for i, state in tqdm(enumerate(states), total=len(states)):
    # Load and process data
    tx_data = data_source.get_data(states=["HI"], download=True)
    tx_features, tx_labels, tx_group = ACSEmployment.df_to_numpy(tx_data)
    tx_features = pd.DataFrame(tx_features, columns=ACSEmployment.features)
    tx_full = tx_features.copy()
    tx_full["group"] = tx_group
    tx_full["target"] = tx_labels

    (
        input_data[state],
        shap_data[state],
        output_data[state],
        model_error_data[state],
        atc_scores_data[state],
    ) = create_meta_data(tx_full, SAMPLE_FRAC, int(ITERS / 10))
# %%
plt.plot()
for k in input_data.keys():
    sns.kdeplot(model_error_data[k], label=k)
plt.legend()
plt.show()
# Evaluating stage
## Input Data
X_tr, X_te, y_tr, y_te = train_test_split(
    input_tr, model_error_tr, test_size=0.33, random_state=42
)
dummy = DummyRegressor(strategy="constant", constant=0).fit(input_tr, model_error_tr)

clf_input = XGBClassifier().fit(input_tr, model_error_tr)
clf_output = XGBClassifier().fit(output_tr, model_error_tr)
clf_shap = XGBClassifier().fit(shap_tr, model_error_tr)
# %%
input_results = []
output_results = []
shap_results = []
dummy_results = []
state_res = []
atc_res = []
for state in input_data.keys():
    state_res.append(str(state))
    input_results.append(
        roc_auc_score(np.where(model_error_data[state]<-0.02,1,0),clf_input.predict(input_data[state]))
    )
    output_results.append(
        roc_auc_score(np.where(model_error_data[state]<-0.02,1,0), clf_output.predict(output_data[state]))
    )
    shap_results.append(
        roc_auc_score(np.where(model_error_data[state]<-0.02,1,0), clf_shap.predict(shap_data[state]))
    )
    dummy_results.append(
        roc_auc_score(np.where(model_error_data[state]<-0.02,1,0), dummy.predict(input_data[state]))
    )
    atc_res.append(roc_auc_score(np.where(model_error_data[state]<-0.02,1,0),np.where(pd.DataFrame(atc_scores_data[state].values(),columns=['values']).values<-0.02,1,0)))
    #atc_res.append(roc_auc_score(pd.DataFrame(np.where(model_error_data[state]<-0.02,1,0), atc_scores_data[state].values())))


# %%
res = pd.DataFrame(
    [input_results, output_results, shap_results, dummy_results, atc_res, state_res]
).T
res = res.rename(
    columns={0: "input", 1: "output", 2: "shap", 3: "dummy", 4: "atc", 5: "state"}
)
# %%
aux = res.mean()
sns.barplot(x=aux.index, y=aux.values)
# %%
plt.figure()
aux.index
plt.show()
# %%
