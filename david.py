# %%
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

random.seed(0)
from sklearn.linear_model import LogisticRegression
# Scikit Learn
from sklearn.model_selection import train_test_split

plt.style.use("seaborn-whitegrid")
import shap
from folktables import ACSDataSource, ACSEmployment
from xgboost import XGBRegressor

# %%
# Do we want synthetic or real data?
## Synthetic data
### Normal
synthetic_data = False
if synthetic_data:
    ## Synthetic data
    ### Normal
    sigma = 1
    rho = 0.5
    mean = [0, 0]
    cov = [[sigma, 0], [0, sigma]]
    samples = 5_000
    x1, x2 = np.random.multivariate_normal(mean, cov, samples).T
    x3 = np.random.normal(0, sigma, samples)
    # Different values
    mean = [0, 0]
    cov = [[sigma, rho], [rho, sigma]]
    x11, x22 = np.random.multivariate_normal(mean, cov, samples).T
    x33 = np.random.normal(0, sigma, samples)

    # Create Data
    df = pd.DataFrame(data=[x1, x2, x3]).T
    df.columns = ["Var%d" % (i + 1) for i in range(df.shape[1])]
    df["target"] = (
        df["Var1"] * df["Var2"] + df["Var3"] + np.random.normal(0, 0.1, samples)
    )
    X_ood = pd.DataFrame(data=[x11, x22, x33]).T
    X_ood.columns = ["Var%d" % (i + 1) for i in range(X_ood.shape[1])]
    y_ood = (
        X_ood["Var1"] * X_ood["Var2"]
        + X_ood["Var3"]
        + np.random.normal(0, 0.1, samples)
    )
    X = df.drop(columns="target")
    y = df["target"]

else:
    ## Real data based on US census data
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    acs_data = data_source.get_data(states=["CA"], download=True)
    X, y, group = ACSEmployment.df_to_numpy(acs_data)
    X = pd.DataFrame(X, columns=ACSEmployment.features)
    # Lets make smaller data for computational reasons
    X = X.head(10_000)
    y = y[:10_000]
    # OOD data
    acs_data = data_source.get_data(states=["NY"], download=True)
    X_ood, y_ood, group = ACSEmployment.df_to_numpy(acs_data)
    X_ood = pd.DataFrame(X_ood, columns=ACSEmployment.features)
    X_ood = X_ood.head(5_000)
    y_ood = y_ood[:5_000]


# %%
## Fit our ML model
model = XGBRegressor(random_state=0)
gmodel = LogisticRegression()

# %%
class ShiftDetector:
    def __init__(self, model, gmodel):
        self.model = model
        self.gmodel = gmodel
        self.explainer = None
    
    def fit_model(self, X, y):
        self.model.fit(X, y)
    
    def fit_explanation_space(self, X, y):
        self.gmodel.fit(X, y)
    
    def get_explanations(self, X):
        shap_values = self.explainer(X)
        exp = pd.DataFrame(
            data=shap_values.values,
            columns=["Shap%d" % (i + 1) for i in range(X.shape[1])],
        )
        return exp
    
    def get_iid_explanations(self, X, y):
        # Does too many things, getting and setting, not good
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, random_state=0, test_size=0.5
        )
        self.fit_model(X_tr, y_tr)
        self.explainer = shap.Explainer(self.model)
        return self.get_explanations(X_te)
    
    def get_all_explanations(self, X, y, X_ood):
        X_iid = self.get_iid_explanations(X, y)
        X_ood = self.get_explanations(X_ood)
        X_iid["label"] = 0
        X_ood["label"] = 1
        X = pd.concat([X_iid, X_ood])
        return X
    
    def get_auc(self, X, y, X_ood):
        X_shap = self.get_all_explanations(X, y, X_ood)
        X_shap_tr, X_shap_te, y_shap_tr, y_shap_te = train_test_split(
            X_shap.drop(columns="label"), X_shap["label"], random_state=0, test_size=0.5
        )
        self.fit_explanation_space(X_shap_tr, y_shap_tr)
        return roc_auc_score(y_shap_te, self.gmodel.predict_proba(X_shap_te)[:, 1])
        

# %%
ShiftDetector(model, gmodel).get_auc(X, y, X_ood)
# %% Build AUC interval
aucs = []
for _ in tqdm(range(100)):
    X_train, X_test, y_tr, y_te = train_test_split(
        X, y, test_size=0.5
    )
    auc = ShiftDetector(model, gmodel).get_auc(X_train, y_tr, X_test)
    aucs.append(auc)
    

# %%
ood_auc = ShiftDetector(model, gmodel).get_auc(X, y, X_ood)
# %%
aucs = np.array(aucs)

# %% This is a p-value
np.mean(aucs > ood_auc)
# %% no-shift confidence interval
lower = np.quantile(aucs, 0.025)
upper = np.quantile(aucs, 0.975)
lower, upper
# %% For a random in-distribution sample, we get a low p-value
np.mean(aucs > aucs[5])

# %% See, for each distribution sample, if we would reject with alpha = 0.05
# 95% of the times we wouldn't reject with alpha = 0.05
np.mean((aucs >= lower) * (aucs <= upper))

# TODO: Think if we should do one-sided or two-sided test


# %%
