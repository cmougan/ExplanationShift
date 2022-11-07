# %%
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})
import seaborn as sns
import pandas as pd
import random
from tqdm import tqdm

random.seed(0)
# Scikit Learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tools.xaiUtils import ExplanationShiftDetector
import seaborn as sns

plt.style.use("seaborn-whitegrid")
from xgboost import XGBClassifier


from tools.datasets import GetData
from tools.xaiUtils import ExplanationShiftDetector

# %%
data = GetData(type="real")
X, y, X_ood, y_ood = data.get_data()
# Hold out set for CA-14
X_cal_1, X_cal_2, y_cal_1, y_cal_2 = train_test_split(X, y, test_size=0.2)
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
for _ in tqdm(range(100)):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5)
    detector.fit(X_tr, y_tr, X_te)
    aucs.append(detector.get_auc_val())
    cofs.append(detector.get_coefs()[0])
# %%
## OOD AUC
ood_auc = []
ood_coefs = []
# states = ["NY14", "TX14", "HI14", "NY18", "TX18", "HI18", "CA18", "CA14"]
states = ["NY18", "TX18", "HI18", "CA18"]

for state in states:
    X_ood, _ = data.get_state(state=state[:2], year="20" + state[2:])
    detector.fit(X, y, X_ood)
    ood_auc.append(detector.get_auc_val())
    ood_coefs.append(detector.get_coefs())
# %%
# Lets add the CA-14 indistribution hold out set
detector.fit(X, y, X_cal_2)
ood_auc.append(detector.get_auc_val())
ood_coefs.append(detector.get_coefs())
states = states + ["CA14"]
# %%
# Plot AUC
plt.figure(figsize=(10, 6))
plt.title("AUC OOD performance of the Explanation Shift detector")
sns.kdeplot(aucs, fill=True, label="In-Distribution (CA14)")
plt.axvline(ood_auc[0], label=states[0], color="#00BFFF")
plt.axvline(ood_auc[1], label=states[1], color="#C68E17")
plt.axvline(ood_auc[2], label=states[2], color="#7DFDFE")
plt.axvline(ood_auc[3], label=states[3], color="#6F4E37")
plt.axvline(ood_auc[4], label="CA-14 (Hold Out)")
# plt.axvline(ood_auc[4], label=states[4], color="#EB5406")
# plt.axvline(ood_auc[5], label=states[5], color="#8E7618")
# plt.axvline(ood_auc[6], label=states[6], color="r")
# plt.axvline(ood_auc[7], label=states[7])
plt.legend()
plt.tight_layout()
plt.savefig("images/AUC_OOD.png")
plt.show()
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
for i in range(len(ood_coefs)):
    coefs_res[states[i]] = np.mean(coefs < ood_coefs[i])

# %%
coefs_res["mean"] = coefs_res.mean(axis=1)
coefs_res.sort_values(by="mean", ascending=True)
# %%
coefs_res.sort_values(by="mean", ascending=True).shape
# %%
plt.figure(figsize=(10, 6))
plt.title("Feature importance of the Explanation Shift detector (p-values)")
sns.heatmap(coefs_res.sort_values(by="mean", ascending=True), annot=True)
plt.tight_layout()
plt.savefig("images/feature_importance.png")
plt.show()
