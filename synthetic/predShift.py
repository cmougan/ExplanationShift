# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

random.seed(0)

# Import kstest
from scipy.stats import kstest
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import shap

# %%
# Create the data
n = 5_000
x1 = np.random.uniform(1, 2, n) + np.random.normal(0, 0.1, n)
x2 = np.random.uniform(0, 1, n) + np.random.normal(0, 0.1, n)
# Dataframe
X = pd.DataFrame({"x1": x1, "x2": x2})
y = x1 + x2 + np.random.normal(0, 0.1, n)
x11 = np.random.uniform(0, 1, n) + np.random.normal(0, 0.1, n)
x22 = np.random.uniform(1, 2, n) + np.random.normal(0, 0.1, n)
# Dataframe
X_ood = pd.DataFrame({"x1": x11, "x2": x22})
# %%
# Fit model
model = LinearRegression()
model.fit(X, y)
# Calculate SHAP values
explainer = shap.Explainer(model, X)
shap_values = pd.DataFrame(explainer(X).values, columns=X.columns)
shap_values_ood = pd.DataFrame(explainer(X_ood).values, columns=X.columns)

preds_id = model.predict(X)
preds_ood = model.predict(X_ood)
# %%
print(kstest(shap_values["x1"], shap_values_ood["x1"]))
print(kstest(shap_values["x1"], shap_values_ood["x1"]))
print(kstest(preds_id, preds_ood))
# %%
# plot the two distributions
plt.figure()
plt.hist(preds_id, bins=100, alpha=0.5, label="id")
plt.hist(preds_ood, bins=100, alpha=0.5, label="ood")
plt.legend()
plt.show()

# %%
plt.figure()
plt.hist(shap_values["x1"], bins=100, alpha=0.5, label="id")
plt.hist(shap_values_ood["x1"], bins=100, alpha=0.5, label="ood")
plt.legend()
plt.show()

# %%
