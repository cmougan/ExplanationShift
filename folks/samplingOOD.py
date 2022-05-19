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
import numpy as np
import random
import sys

# Scikit-Learn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR

# Specific packages
from xgboost import XGBRegressor, XGBClassifier
import shap
from tqdm import tqdm


# Home made code
import sys

sys.path.append("../")
from fairtools.utils import loop_estimators_fairness, psi, loop_estimators

# Seeding
np.random.seed(0)
random.seed(0)

# Load data
data_source = ACSDataSource(survey_year="2014", horizon="1-Year", survey="person")
ca_data = data_source.get_data(states=["CA"], download=True)
data_source = ACSDataSource(survey_year="2016", horizon="1-Year", survey="person")
mi_data = data_source.get_data(states=["MI"], download=True)
# %%
states = [
    "MI",
    "TN",
    "CT",
    "OH",
    "NE",
    "IL",
    "FL",
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


ca_features, ca_labels, ca_group = ACSEmployment.df_to_numpy(ca_data)
mi_features, mi_labels, mi_group = ACSEmployment.df_to_numpy(mi_data)

## Conver to DF
ca_features = pd.DataFrame(ca_features, columns=ACSEmployment.features)
mi_features = pd.DataFrame(mi_features, columns=ACSEmployment.features)

# Modeling
model = XGBClassifier(verbosity=0, silent=True, use_label_encoder=False, njobs=1)

# Train on CA data
preds_ca = cross_val_predict(
    model, ca_features, ca_labels, cv=3, method="predict_proba"
)[:, 1]
model.fit(ca_features, ca_labels)
# Test on MI data
preds_mi = model.predict_proba(mi_features)[:, 1]

# %%
## Can we learn to solve this issue?
################################
####### PARAMETERS #############
SAMPLE_FRAC = 1_000
ITERS = 2000
# Init
train = defaultdict()
train_target_shift = defaultdict()
train_target_shift_ood = defaultdict()
train_ood = defaultdict()
performance = defaultdict()

train_shap = defaultdict()
train_shap_ood = defaultdict()
train_error = roc_auc_score(ca_labels, preds_ca)

# xAI Train
explainer = shap.Explainer(model)
shap_test = explainer(ca_features)
shap_test = pd.DataFrame(shap_test.values, columns=ca_features.columns)

# Lets add the target to ease the sampling
mi_full = mi_features.copy()
mi_full["group"] = mi_group
mi_full["target"] = mi_labels
# Trainning set
for i in tqdm(range(0, ITERS), leave=False, desc="Test Bootstrap", position=1):
    # Initiate
    row = []
    row_target_shift = []
    row_shap = []

    # Sampling
    aux = mi_full.sample(n=SAMPLE_FRAC, replace=True)

    # Performance calculation
    preds = model.predict_proba(aux.drop(columns=["target", "group"]))[:, 1]
    performance[i] = train_error - roc_auc_score(aux.target, preds)

    # Shap values calculation
    shap_values = explainer(aux.drop(columns=["target", "group"]))
    shap_values = pd.DataFrame(shap_values.values, columns=ca_features.columns)

    for feat in ca_features.columns:
        # Michigan
        ks = psi(ca_features[feat], aux[feat])
        sh = psi(shap_test[feat], shap_values[feat])

        row.append(ks)
        row_shap.append(sh)
    # Target shift
    ks_target_shift = wasserstein_distance(preds_ca, preds)
    row_target_shift.append(ks_target_shift)
    # Save test

    train_shap[i] = row_shap
    train[i] = row
    train_target_shift[i] = row_target_shift

# Save results
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
## OOD State loop
for state in tqdm(states, desc="States", position=0):
    print(state)
    # Initialize some lists
    performance_ood = defaultdict()

    # Load and process data
    tx_data = data_source.get_data(states=["HI"], download=True)
    tx_features, tx_labels, tx_group = ACSEmployment.df_to_numpy(tx_data)
    tx_features = pd.DataFrame(tx_features, columns=ACSEmployment.features)

    # Lets add the target to ease the sampling
    tx_full = tx_features.copy()
    tx_full["group"] = tx_group
    tx_full["target"] = tx_labels

    # Loop to create training data
    for i in tqdm(range(0, int(ITERS/20)), leave=False, desc="Bootstrap", position=1):
        row_ood = []
        row_shap_ood = []
        row_target_shift_ood = []

        # Sampling
        aux_ood = tx_full.sample(n=SAMPLE_FRAC, replace=True)

        # OOD performance calculation
        preds_ood = model.predict_proba(aux_ood.drop(columns=["target", "group"]))[:, 1]
        performance_ood[i] = train_error - roc_auc_score(
            aux_ood.target.values, preds_ood
        )

        # Shap values calculation OOD
        shap_values_ood = explainer(aux_ood.drop(columns=["target", "group"]))
        shap_values_ood = pd.DataFrame(
            shap_values_ood.values, columns=ca_features.columns
        )

        for feat in ca_features.columns:
            # OOD
            ks_ood = psi(ca_features[feat], aux_ood[feat])
            sh_ood = psi(shap_test[feat], shap_values_ood[feat])
            row_ood.append(ks_ood)
            row_shap_ood.append(sh_ood)

        # Target shift
        ks_target_shift_ood = wasserstein_distance(preds_ca, preds_ood)
        row_target_shift_ood = ks_target_shift_ood
        # Save OOD
        train_shap_ood[i] = row_shap_ood
        train_ood[i] = row_ood
        train_target_shift_ood[i] = row_target_shift_ood

    # Save results
    ## Test (previous OOD)
    train_df_ood = pd.DataFrame(train_ood).T
    train_df_ood.columns = ca_features.columns

    train_shap_df_ood = pd.DataFrame(train_shap_ood).T
    train_shap_df_ood.columns = ca_features.columns
    train_shap_df_ood = train_shap_df_ood.add_suffix("_shap")

    train_target_shift_df_ood = pd.DataFrame(train_target_shift_ood, index=[0]).T
    train_target_shift_df_ood.columns = ["target"]

    # Estimators for the loop
    estimators = defaultdict()
    estimators["Dummy"] = DummyRegressor()
    estimators["Linear"] = Pipeline(
        [("scaler", StandardScaler()), ("model", LinearRegression())]
    )
    estimators["RandomForest"] = RandomForestRegressor(random_state=0)
    estimators["XGBoost"] = XGBRegressor(
        verbosity=0, verbose=0, silent=True, random_state=0
    )
    estimators["SVM"] = Pipeline([("scaler", StandardScaler()), ("model", SVR(C=0.01))])
    estimators["MLP"] = Pipeline(
        [("scaler", StandardScaler()), ("model", MLPRegressor(random_state=0))]
    )

    ## Loop over different G estimators
    # Some target transformations
    performance_ood = pd.DataFrame(performance_ood, index=[0]).T.values

    # Performance
    loop_estimators_fairness(
        estimator_set=estimators,
        normal_data=train_df,
        normal_data_ood=train_df_ood,
        shap_data=train_shap_df,
        shap_data_ood=train_shap_df_ood,
        target_shift=train_target_shift_df,
        target_shift_ood=train_target_shift_df_ood,
        performance_ood=performance_ood,
        target=performance,
        state=state,
        error_type="performance",
    )


# %%
