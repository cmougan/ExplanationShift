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
for _ in tqdm(range(100)):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5)
    detector.fit(X_tr, y_tr, X_te)
    aucs.append(detector.get_auc_val())
# %%
## OOD AUC
detector.fit(X, y, X_ood)
ood_auc = detector.get_auc_val()
# %%
# Plot AUC
plt.figure(figsize=(10, 6))
sns.kdeplot(aucs)
plt.axvline(ood_auc, color="red", label="OOD AUC")
plt.show()
# %% This is a p-value
np.mean(aucs > ood_auc)
# %% no-shift confidence interval
lower = np.quantile(aucs, 0.025)
upper = np.quantile(aucs, 0.975)
lower, upper
# %% For a random in-distribution sample, we get a low p-value
np.mean(aucs > aucs[5])

# %% See, for each distribution sample, if we would reject with alpha = 0.05
# 95% of the times we wouldn't reject with alpha = 0.05
np.mean((aucs >= lower) * (aucs <= upper))
# %%
