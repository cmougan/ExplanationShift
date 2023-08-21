# %%
from folktables import ACSDataSource, ACSEmployment
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
import shap
import numpy as np

# %%
data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
acs_data = data_source.get_data(states=["AL"], download=True)
features, label, group = ACSEmployment.df_to_numpy(acs_data)
# Convert to dataframe
df = pd.DataFrame(features, columns=ACSEmployment.features).drop(columns="RAC1P")
# There are 2 labels, y and z
df["y"] = label
df["z"] = group
df = df[(df["z"] == 1) | (df["z"] == 2)]
# Convert to binary classification
df["z"] = df["z"].apply(lambda x: True if x == 1 else False)
# %%
# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=["y", "z"]), df[["y", "z"]], test_size=0.2, random_state=42
)
# %%
# Train two models
model1 = XGBClassifier()
model1.fit(X_train, y_train["y"])

model2 = XGBClassifier()
model2.fit(X_train, y_train["z"])

# Score AUC models
print("AUC model1(y): ", roc_auc_score(y_test["y"], model1.predict_proba(X_test)[:, 1]))
print("AUC model2(z): ", roc_auc_score(y_test["z"], model2.predict_proba(X_test)[:, 1]))

# %%
# Explanations of the first model
explainer = shap.Explainer(model1)
shap_values1 = explainer(X_test)
shap.plots.bar(shap_values1)

# Explanations of the second model
explainer = shap.Explainer(model2)
shap_values2 = explainer(X_test)
shap.plots.bar(shap_values2)

# We can see how explanations are different are different

# %%
# Lets save the explanations distributions
exp1 = pd.DataFrame(shap_values1.data, columns=X_test.columns)
exp2 = pd.DataFrame(shap_values2.data, columns=X_test.columns)
res = X_test.copy()  # We will save the results here
# %%
# Anomaly detection
# On Raw Data
iso = IsolationForest()
iso.fit(X_test)
res["iso"] = iso.predict(X_test)

# On Model 1
iso = IsolationForest()
iso.fit(exp1)
res["iso1"] = iso.predict(exp1)

# On Model 2
iso = IsolationForest()
iso.fit(exp2)
res["iso2"] = iso.predict(exp2)

# %%
# We can see how there are discrepancies between the methods
print(np.sum(res["iso"] != res["iso1"]))
print(np.sum(res["iso"] != res["iso2"]))
print(np.sum(res["iso1"] != res["iso2"]))
# %%
