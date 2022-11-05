# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random

random.seed(0)
# Scikit Learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tools.xaiUtils import ExplanationShiftDetector
from sklearn.datasets import make_blobs

plt.style.use("seaborn-whitegrid")
from xgboost import XGBRegressor, XGBClassifier
import shap
from folktables import ACSDataSource, ACSEmployment

# %%
# Do we want synthetic or real data?
synthetic_data = "blobs"
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

    ## Split Data
    X_tr, X_te, y_tr, y_te = train_test_split(
        df.drop(columns="target"), df["target"], test_size=0.5, random_state=0
    )
elif synthetic_data == "blobs":
    X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
    X = pd.DataFrame(X, columns=["a", "b", "c", "d", "e"])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
    X_ood = make_blobs(n_samples=1000, centers=1, n_features=5, random_state=0)


else:
    ## Real data based on US census data
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    acs_data = data_source.get_data(states=["CA"], download=True)
    X, y, group = ACSEmployment.df_to_numpy(acs_data)
    X = pd.DataFrame(X, columns=ACSEmployment.features)
    # Lets make smaller data for computational reasons
    X = X.head(10_000)
    y = y[:10_000]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
    # OOD data
    acs_data = data_source.get_data(states=["NY"], download=True)
    X_ood, y_ood, group = ACSEmployment.df_to_numpy(acs_data)
    X_ood = pd.DataFrame(X_ood, columns=ACSEmployment.features)
    X_ood = X_ood.head(5_000)
    y_ood = y_ood[:5_000]
# %%

# %%
detector = ExplanationShiftDetector(model=XGBRegressor(), gmodel=LogisticRegression())

detector.get_auc(X_tr, y_tr, X_ood)


# %%# %%
# %%
# %%
kk
## Fit our ML model
model = XGBRegressor(random_state=0)
# model = LinearRegression()
model.fit(X_tr, y_tr)

# Build explanation space
## Real explanation
### Type of shap values
explainer = shap.Explainer(model)
# explainer = shap.LinearExplainer(model, X_te, feature_dependence="correlation_dependent")
# shap.KernelExplainer(model.predict,X_te,nsamples=100)
# %%
## In distribution explanation
shap_values = explainer(X_te)
### Created a dataframe
if synthetic_data:
    exp = pd.DataFrame(
        data=shap_values.values, columns=["Shap%d" % (i + 1) for i in range(3)]
    )
else:
    exp = pd.DataFrame(data=shap_values.values, columns=ACSEmployment.features)
# %%
##  OODexplanation
preds_ood = model.predict(X_ood)
preds_te = model.predict(X_te)
shap_values = explainer(X_ood)
if synthetic_data:
    exp_ood = pd.DataFrame(
        data=shap_values.values, columns=["Shap%d" % (i + 1) for i in range(3)]
    )

else:
    exp_ood = pd.DataFrame(data=shap_values.values, columns=ACSEmployment.features)

# %%
## xAI on the model G
## Shap Estimator
exp_ood["label"] = 1
exp["label"] = 0
exp_space = pd.concat([exp, exp_ood])


## Model to be used
# gmodel = XGBClassifier(random_state=0)
gmodel = LogisticRegression()
# gmodel = Pipeline((("scaler", StandardScaler()), ("gmodel", LogisticRegression())))
# Explanation Space
S_tr, S_te, yy_tr, yy_te = train_test_split(
    exp_space.drop(columns="label"),
    exp_space[["label"]],
    random_state=0,
    test_size=0.5,
    stratify=exp_space[["label"]],
)
gmodel.fit(S_tr, yy_tr)

# %%
explainer = shap.LinearExplainer(
    gmodel, S_te, feature_dependence="correlation_dependent"
)
shap_values = explainer(S_te)
plt.figure()
shap.plots.bar(shap_values, show=False)
plt.show()

# %%
m = XGBClassifier(random_state=0)

# %%
m.__class__.__name__
# %%
a = pd.DataFrame(data=[1, 2, 3], columns=["AA"])
# %%
