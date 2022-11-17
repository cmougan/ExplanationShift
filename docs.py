# %%
from sklearn.datasets import make_blobs
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from tools.xaiUtils import ExplanationShiftDetector

X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
X_ood, _ = make_blobs(n_samples=1000, centers=1, n_features=5, random_state=0)
# %%
detector = ExplanationShiftDetector(model=XGBRegressor(), gmodel=LogisticRegression())
detector.fit(X, y,X_ood)
# %%
detector.get_auc_val()
# %%
