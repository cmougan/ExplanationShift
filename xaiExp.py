# %%
import matplotlib.pyplot as plt
import sys

plt.rcParams.update({"font.size": 14})
import seaborn as sns
import pandas as pd
import random
from scipy.stats import wasserstein_distance
from tqdm import tqdm

random.seed(0)
# Scikit Learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tools.explanationShift import ExplanationShiftDetector
import seaborn as sns

plt.style.use("seaborn-whitegrid")
from xgboost import XGBClassifier


from tools.datasets import GetData
from tools.explanationShift import ExplanationShiftDetector

# %%
# Number of bootstrap samples
N_b = 20
datasets = sys.argv[1]
data = GetData(datasets=datasets)
X, y = data.get_state(year=2014, state="CA", verbose=True)
print("Dataset:", datasets)


# Hold out set for CA-14
X_cal_1, X_cal_2, y_cal_1, y_cal_2 = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=0
)
X, y = X_cal_1, y_cal_1
# %%
detector = ExplanationShiftDetector(
    model=XGBClassifier(),
    gmodel=Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(penalty="l1", solver="liblinear")),
        ]
    ),
)
# %% Build AUC interval
aucs = []
cofs = []
for i in tqdm(range(N_b)):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=i
    )
    detector.fit(X_tr, y_tr, X_te)
    aucs.append(detector.get_auc_val())
    cofs.append(detector.get_coefs()[0])
# %%
## OOD AUC
ood_auc = {}
ood_coefs = {}
# states = ["NY14", "TX14", "HI14", "NY18", "TX18", "HI18", "CA18", "CA14"]
states = ["NY18", "TX18", "HI18", "KS18", "MN18", "PR18", "CA18"]
for state in tqdm(states):
    X_ood_, _ = data.get_state(state=state[:2], year="20" + state[2:])
    ood_temp = []
    ood_coefs_temp = pd.DataFrame(columns=X.columns)
    for i in range(N_b):
        X_ood = X_ood_.sample(frac=0.632, replace=True)
        detector.fit(X, y, X_ood)
        ood_temp.append(detector.get_auc_val())
        ood_coefs_temp = ood_coefs_temp.append(
            pd.DataFrame(detector.get_coefs(), columns=X.columns)
        )
    ood_auc[state] = ood_temp
    ood_coefs[state] = ood_coefs_temp

# %%
# Lets add the CA-14 indistribution hold out set
detector.fit(X, y, X_cal_2)
hold_auc = detector.get_auc_val()
hold_coefs = detector.get_coefs()
# %%
# Plot AUC
plt.figure(figsize=(10, 6))
plt.title("AUC OOD performance of the Explanation Shift detector")
plt.ylabel("AUC")
sns.kdeplot(aucs, fill=True, label="In-Distribution (CA14)")
colors = ["#00BFFF", "#C68E17", "#7DFDFE", "#6F4E37", "#EB5406", "r", "g", "k"]
for i, state in enumerate(states):
    # plt.axvline(np.mean(ood_auc[state]), label=state, color=colors[i])
    print(state, colors[i])
    sns.kdeplot(ood_auc[state], label=state, color=colors[i], fill=True)
# plt.axvline(hold_auc, label="CA-14 (Hold Out)")
plt.legend()
plt.tight_layout()
plt.savefig("images/AUC_OOD_{}.png".format(datasets))
plt.close()
# %%
# Analysis of performance of G
#  This is a p-value
# print("Pvalue: ", np.mean(aucs > ood_auc))
# no-shift confidence interval
# lower = np.quantile(aucs, 0.025)
# upper = np.quantile(aucs, 0.975)
# print("No-shift confidence interval: {:.2f},{:.2f} ".format(lower, upper))
# print("For a random in-distribution sample, we get a low p-value {:.2f}".format(np.mean(aucs > aucs[5])))
# print("See, for each distribution sample, if we would reject with alpha = 0.05",np.mean((aucs >= lower) * (aucs <= upper)),)
# print("{} of the times we wouldnt reject with alpha = 0.05".format(np.mean((aucs >= lower) * (aucs <= upper))))
# %%
# Analysis of coeficients
coefs = pd.DataFrame(cofs, columns=X.columns)
coefs_res = pd.DataFrame(index=coefs.columns)
# %%
# Strength of the feature importance
for i, state in enumerate(states):
    for col in coefs.columns:
        coefs_res.loc[col, state] = wasserstein_distance(
            ood_coefs[state][col], coefs[col]
        )

# %%
# Means on both axis
coefs_res["mean"] = coefs_res.mean(axis=1)
coefs_res.loc["mean"] = coefs_res.mean(axis=0)
coefs_res.sort_values(by="mean", ascending=True)
# %%
from matplotlib.colors import LogNorm

plt.figure(figsize=(10, 6))
plt.title("Feature importance of the Explanation Shift detector (Wasserstein)")
sns.heatmap(
    coefs_res.sort_values(by="mean", ascending=False, axis=0)
    .sort_values(by="mean", ascending=False, axis=1)
    .drop(["mean"], axis=1)
    .drop(["mean"], axis=0),
    annot=True,
    norm=LogNorm(),
)
plt.tight_layout()
plt.savefig("images/feature_importance_{}.png".format(datasets))
plt.close()