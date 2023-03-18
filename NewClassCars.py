# %%
# Import Folktables
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.style.use("seaborn-whitegrid")
rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8
rcParams.update({"font.size": 22})
import seaborn as sns
from sklearn.model_selection import train_test_split
from nobias import ExplanationShiftDetector
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder

# %%
df = pd.read_csv(
    "dataCars/Automobile_customers.csv", index_col="Customer ID"
).reset_index(drop=True)
# %%
# Convert categorical variables to interget
df["On_Call_Offer"] = df["On_Call_Offer"].astype("category").cat.codes.astype(int)
# %%
aucs = {}
for cat in [0, 1, 2, 3]:
    print(cat)
    X = df[df["On_Call_Offer"] != cat].drop("On_Call_Offer", axis=1)._get_numeric_data()
    y = df[df["On_Call_Offer"] != cat]["On_Call_Offer"]
    y = LabelEncoder().fit_transform(y)
    X_ood = (
        df[df["On_Call_Offer"] == cat].drop("On_Call_Offer", axis=1)._get_numeric_data()
    )
    y_ood = df[df["On_Call_Offer"] == cat]["On_Call_Offer"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=42)

    detector = ExplanationShiftDetector(
        model=XGBRegressor(), gmodel=LogisticRegression()
    )

    # Concatenate the training and validation sets
    params = np.linspace(0.1, 0.99, 10)

    aucs_temp = []
    for i in params:
        n_samples = X_ood.shape[0] - int(i * X_ood.shape[0])
        n_samples_1 = n_samples

        X_ = X_ood.loc[~X_ood.index.isin(X_ood.sample(10).index)]
        X_new = X_te.sample(n_samples, replace=False).append(X_)

        detector.fit(X_tr, y_tr, X_new)
        aucs_temp.append(detector.get_auc_val())

        aucs[cat] = aucs_temp
# %%
# Plot
plt.figure()
for cat in [0, 1, 2, 3]:
    # TODO: Rename labels with te
    if cat == 0:
        label = "No Offer"
    elif cat == 1:
        label = "Offer 1"
    elif cat == 2:
        label = "Offer 2"
    elif cat == 3:
        label = "Offer 3"
    elif cat == 4:
        label = "Offer 4"
    else:
        label = "Offer 5"

    plt.plot(params, aucs[cat], label=label)
plt.xlabel("Fraction of OOD data")
plt.ylabel("AUC of Explanation Shift Detector")
plt.savefig("images/PUshift.png")
plt.legend()

# %%
