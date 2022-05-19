# %%
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")
# %%
files = os.listdir()

data = pd.DataFrame()
for file in files:
    if file.endswith(".csv"):
        if file != "all_results.csv":
            if file != "all_results_fairness.csv":
                data = data.append(pd.read_csv(file))
data.to_csv("all_results.csv", index=False)
"""

df = pd.DataFrame()
for STATE in data.state.unique():
    aux = data[(data["state"] == STATE) & (data["error_type"] == "fairness_one")]
    y2_ood = data[
        (data["state"] == STATE) & (data["error_type"] == "fairness_two")
    ].error_ood
    y2_te = data[
        (data["state"] == STATE) & (data["error_type"] == "fairness_two")
    ].error_te

    aux["y2_te"] = y2_te
    aux["y2_ood"] = y2_ood

    aux.columns = [
        "state",
        "error_type",
        "estimator",
        "data",
        "y1_te",
        "y1_ood",
        "y2_te",
        "y2_ood",
    ]
    aux["fair_te"] = aux["y1_te"] - aux["y2_te"]
    aux["fair_ood"] = aux["y1_ood"] - aux["y2_ood"]
    df = df.append(aux)


# %%
df = df.drop(columns=["y1_te", "y2_te", "y1_ood", "y2_ood"])
# %%
df.to_csv("all_results_fairness.csv", index=False)

# %%
"""
