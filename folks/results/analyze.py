# %%
import pandas as pd
import numpy as np

# %%
df = pd.read_csv("all_results.csv")
df.data = df.data.replace("Only Data", "Data")
df.data = df.data.replace("Only Shap", "Shap")
df.data = df.data.replace("Data + Shap", "Shap + Data")

df = df[df["error_type"] == "fairness"]
df = df.drop(columns="error_te")
# df = df[df['state']=='PR']
df = df[
    (df["estimator"] == "Linear")
    | (df["estimator"] == "Dummy")
    | (df["estimator"] == "XGBoost")
]

pd.pivot_table(df, columns=["estimator", "data",], index=["state"],).T.sort_values(
    by=["estimator", "data"], ascending=True
).round(
    decimals=4
).style.highlight_min()  # .to_csv("state_fairness.csv")
# %%
pd.pivot_table(
    df,
    index=[
        "estimator",
        "data",
    ],
    aggfunc=[np.mean, np.std, np.median],
).sort_values(by=["estimator", "data"], ascending=True).round(
    decimals=6
).style.highlight_min()
# .to_csv("total_mean_fairness.csv")#.style.highlight_min()

# %%
