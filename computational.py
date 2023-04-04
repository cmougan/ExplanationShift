# %%
from sklearn.datasets import make_regression
import pandas as pd
import tqdm
import lime.lime_tabular
import shap
from xgboost import XGBRegressor
import time
import matplotlib.pyplot as plt
import seaborn as sns

# %%
time_tree_shap = []
time_lime = []
time_kernel_shap = []
n_samples = [100, 200, 500, 1000, 5_000]
for n_sample in n_samples:
    X, y = make_regression(n_samples=n_sample, n_features=5, noise=1, random_state=42)
    # Convert X to dataframe
    X = pd.DataFrame(X)
    # Fit model
    model = XGBRegressor().fit(X, y)
    # Name columns with loop
    for i in range(X.shape[1]):
        X.rename(columns={i: "x" + str(i)}, inplace=True)

    # Lime
    tic = time.time()
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X.values,
        feature_names=X.columns,
        verbose=False,
        mode="regression",
    )
    for i, _ in tqdm.tqdm(enumerate(X.iterrows())):
        explainer.explain_instance(X.iloc[i], model.predict)
    toc = time.time()
    time_lime.append(toc - tic)

    # Tree Shap
    tic = time.time()
    explainer = shap.TreeExplainer(model)
    explainer.shap_values(X)
    toc = time.time()
    time_tree_shap.append(toc - tic)

    # Kernel Shap
    tic = time.time()
    explainer = shap.KernelExplainer(model.predict, X)
    explainer.shap_values(X)
    toc = time.time()
    time_kernel_shap.append(toc - tic)


# %%
# Grouped Bart Plot
df = pd.DataFrame(
    {
        "time": time_tree_shap + time_lime + time_kernel_shap,
        "method": ["Tree Shap"] * len(time_tree_shap)
        + ["Lime"] * len(time_lime)
        + ["Kernel Shap"] * len(time_kernel_shap),
        "n_samples": n_samples * 3,
    }
)
sns.barplot(x="n_samples", y="time", hue="method", data=df)
plt.title("Time to explain 1 instance")
plt.xlabel("Number of samples")
plt.ylabel("Time (s)")
plt.savefig("computational_samples.pdf", bbox_inches="tight")
plt.show()

# %%
