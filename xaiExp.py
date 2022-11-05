# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
from tqdm import tqdm

random.seed(0)
# Scikit Learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tools.xaiUtils import ExplanationShiftDetector
import seaborn as sns

plt.style.use("seaborn-whitegrid")
from xgboost import XGBClassifier


from tools.datasets import GetData
from tools.xaiUtils import ExplanationShiftDetector

# %%
data = GetData(type="real")
X, y, X_ood, y_ood = data.get_data()
# %%
detector = ExplanationShiftDetector(model=XGBClassifier(), gmodel=LogisticRegression())
# %% Build AUC interval
aucs = []
cofs = []
for _ in tqdm(range(10)):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5)
    detector.fit(X_tr, y_tr, X_te)
    aucs.append(detector.get_auc_val())
    cofs.append(detector.get_coefs()[0])
# %%
## OOD AUC
ood_auc = []
ood_coefs = []
states = ["NY", "CA", "TX", "FL", "IL"]
for state in states:
    X_ood, _ = data.get_state(state=state)
    detector.fit(X, y, X_ood)
    ood_auc.append(detector.get_auc_val())
    ood_coefs.append(detector.get_coefs())
# %%
# Plot AUC
plt.figure(figsize=(10, 6))
plt.title("AUC distribution of the performance of the Explanation Shift detector")
sns.kdeplot(aucs, fill=True, label="In-Distribution AUC")
for d in range(len(ood_auc)):
    plt.axvline(ood_auc[d], label=states[d], color="red")
plt.legend()
plt.show()
# %%
# Analysis of performance of G
#  This is a p-value
print("Pvalue: ", np.mean(aucs > ood_auc))

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
