# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import shap
from collections import defaultdict
from tqdm import tqdm

sns.set_style(style="whitegrid")
from matplotlib import rcParams


plt.style.use("seaborn-whitegrid")
rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8


# %%
def experiment(
    out: int = 1, samples: int = 50_000, sigma: int = 5, mean: list = [0, 0]
):
    cov = [[sigma, 0], [0, sigma]]
    x1, x2 = np.random.multivariate_normal(mean, cov, samples).T

    # Different values
    cov = [[sigma, out], [out, sigma]]
    x11, x22 = np.random.multivariate_normal(mean, cov, samples).T

    fig, axs = plt.subplots(ncols=1)
    plt.title(out)
    plt.scatter(x1, x2, alpha=0.1, label="NonInteraction")
    plt.scatter(x11, x22, alpha=0.1, label="Interaction")
    plt.legend()
    plt.show()

    df = pd.DataFrame(data=[x1, x2]).T
    df.columns = ["Var%d" % (i + 1) for i in range(df.shape[1])]
    # df["target"] = np.where(df["Var1"] * df["Var2"] > 0, 1, 0)
    df["target"] = df["Var1"] * df["Var2"] + np.random.normal(0, 0.1, samples)

    ## Fit our ML model
    X_tr, X_te, y_tr, y_te = train_test_split(df.drop(columns="target"), df[["target"]])

    model = GradientBoostingRegressor()
    model.fit(X_tr, y_tr)

    ## Real explanation
    explainer = shap.Explainer(model)
    shap_values = explainer(X_te)
    exp = pd.DataFrame(
        data=shap_values.values, columns=["Shap%d" % (i + 1) for i in range(2)]
    )
    ## Fake explanation
    fake = pd.DataFrame(data=[x11, x22]).T
    shap_values = explainer(fake)
    exp_fake = pd.DataFrame(
        data=fake.values, columns=["Shap%d" % (i + 1) for i in range(2)]
    )
    return (
        ks_2samp(exp_fake["Shap1"], exp["Shap1"]),
        ks_2samp(exp_fake["Shap2"], exp["Shap2"]),
        ks_2samp(x11, x1),
        ks_2samp(x22, x2),
    )


# %%
pvalShap = []
ksShap = []
pval = []
ks = []
xx = np.linspace(0, 10, 50)
for i in tqdm(xx):
    s1, s2, x1, x2 = experiment(out=i)
    pval.append(np.mean([x1.pvalue, x2.pvalue]))
    ks.append(np.mean([x1.statistic, x2.statistic]))
    pvalShap.append(np.mean([s1.pvalue, s2.pvalue]))
    ksShap.append(np.mean([s1.statistic, s2.statistic]))

# %%

plt.figure()
plt.title(
    "Pvalue of the Kolmogorov-Smirnov test between detecting input data and model explanations"
)
plt.plot(xx, pval, label="Pvalue on input data")
plt.plot(xx, pvalShap, label="Pvalue on SHAP explanations")
plt.xlabel("Interaction parameter")
plt.ylabel("P value of the Kolmogorov-Smirnov test")
plt.legend()
plt.savefig("images/gaussianpvalue.png")
plt.show()

# %%
plt.figure()
plt.title(
    "Kolmogorov-Smirnov statistic between detecting input data and model explanations"
)
plt.plot(xx, ks, label="Input KS statistic")
plt.plot(xx, ksShap, label="SHAP Kolmogorov-Smirnov statistic")
plt.xlabel("Interaction parameter")
plt.ylabel("Kolmogorov-Smirov statistic")
plt.legend()
plt.savefig("images/gaussianKS.png")
plt.show()

# %%
