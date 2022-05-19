# %%
from cProfile import label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

sns.set_theme(style="whitegrid")
plt.style.use("seaborn-whitegrid")
rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8
rcParams.update({"font.size": 22})

# %%
df = pd.read_csv("folks/results/all_results_xx.csv")

df.data = df.data.replace("Only Data", "Distribution Shift")
df.data = df.data.replace("Data+Target", "Dist+Pred Shift")
df.data = df.data.replace("Only Target", "Prediction Shift")
df.data = df.data.replace("Only Shap", " Explanation Shift")
df.data = df.data.replace("Data+Target+Shap", " Dist+Pred+Exp Shift")


df = df[df["error_type"] == "fairness_two"]
# df = df[df["error_type"] == "fairness_one"]
# df = df[df["error_type"] == "performance"]
df = df.drop(columns="error_te")
# df = df[df['state']=='WA']
df = df[
    (df["estimator"] == "Linear")
    | (df["estimator"] == "Dummy")
    | (df["estimator"] == "RandomForest")
    # | (df["estimator"] == "SVM")
    | (df["estimator"] == "XGBoost")
]

pd.pivot_table(df, columns=["estimator", "data",], index=["state"],).T.sort_values(
    by=["estimator", "data"], ascending=True
).round(
    decimals=4
).style.highlight_min()  # .to_csv("state_fairness.csv")
# %%
aux = (
    pd.pivot_table(
        df,
        index=[
            "estimator",
            "data",
        ],
        aggfunc=[np.mean, np.std, np.median],
    )
    .sort_values(by=["estimator", "data"], ascending=True)
    .round(decimals=6)
)  # .style.highlight_min()
# .to_csv("total_mean_fairness.csv")#.style.highlight_min()
aux = aux.reset_index()

aux = pd.DataFrame(aux.values, columns=["Estimator", "Data", "Mean", "Std", "Median"])
aux = aux.drop(columns=["Std", "Median"])
base = aux[aux.Estimator == "Dummy"].Mean.mean()
aux = aux[aux.Estimator != "Dummy"]
aux = aux.sort_values(by="Data", ascending=False)
# %%
plt.figure()
sns.barplot(x="Estimator", y="Mean", hue="Data", data=aux)
plt.axhline(base, color="black", linestyle="--", label="Baseline")
plt.ylabel("Error on quantification of model performance")
plt.legend()
plt.show()
# %%

# %%
