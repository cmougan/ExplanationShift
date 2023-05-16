# %%
from alibi_detect.datasets import fetch_ecg,fetch_kdd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor,XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Lasso + Logistic Regression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression

from skshift import ExplanationShiftDetector

# %%
(X_train, y_train), (X_test, y_test) = fetch_kdd(return_X_y=True)
# %%
a,b = fetch_kdd(return_X_y=True)
# %%
# Data
(X_train, y_train), (X_test, y_test) = fetch_ecg(return_X_y=True)
# To dataframe
X_train = pd.DataFrame(X_train, columns=["X" + str(i) for i in range(X_train.shape[1])])
X_test = pd.DataFrame(X_test, columns=["X" + str(i) for i in range(X_test.shape[1])])

# %%
X_train = pd.DataFrame(X_train, columns=["X" + str(i) for i in range(X_train.shape[1])])
X_test = pd.DataFrame(X_test, columns=["X" + str(i) for i in range(X_test.shape[1])])


X_train['ood']=1
X_test['ood']=0
X_train['label']=y_train
X_test['label']=y_test
# Concat
X = pd.concat([X_train,X_test])
# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X.drop('ood',axis=1), X['ood'], test_size=0.5, random_state=0
)
# IS OOD?
m = XGBClassifier().fit(X_train,y_train)
# AUC
roc_auc_score(y_test,m.predict_proba(X_test)[:,1])

#  Pefromance
X_train, X_test, y_train, y_test = train_test_split(
    X.drop('label',axis=1), X['label'], test_size=0.5, random_state=0
)
# IS OOD?
m = XGBClassifier().fit(X_train,y_train)
# AUC
roc_auc_score(y_test,m.predict_proba(X_test)[:,1])
# %%
# %%
# %%
# %%
# %%
# %%
# %%

















# train val split
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.5, random_state=0
)
# train holdout split
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X_train, y_train, test_size=0.5, random_state=0
)

# OOD train
X_ood_tr, X_ood_te, y_ood_tr, y_ood_te = train_test_split(
    X_test, y_test, test_size=0.5, random_state=0
)

# Eval data
OOD_test = pd.concat([X_holdout, X_ood_te])
z_test = np.concatenate([np.zeros_like(y_holdout), np.ones_like(y_ood_te)])
OOD_train = pd.concat([X_train, X_ood_tr])
z_train = np.concatenate([np.zeros_like(y_train), np.ones_like(y_ood_tr)])

# %%
model = XGBRegressor().fit(X_train, y_train)
# %%
detector = ExplanationShiftDetector(
    model=model, gmodel=LogisticRegression(penalty="l1", solver="liblinear")
)
detector.fit_detector(X_val,X_ood_tr)
# %%
roc_auc_score(z_test,detector.predict_proba(OOD_test)[:,1])

# %%
# Input space

m = LogisticRegression().fit(OOD_train,z_train)
roc_auc_score(z_test,m.predict_proba(OOD_test)[:,1])

# %%

from scipy.stats import ks_2samp
import seaborn as sns
vals = []
for col in X_train.columns:
    vals.append(ks_2samp(X_train[col],X_test[col]).pvalue)
sns.kdeplot(vals)
# %%

(X_train, y_train), (X_test, y_test) = fetch_ecg(return_X_y=True)
# To dataframe
X_train = pd.DataFrame(X_train, columns=["X" + str(i) for i in range(X_train.shape[1])])
X_test = pd.DataFrame(X_test, columns=["X" + str(i) for i in range(X_test.shape[1])])

# %%
m 