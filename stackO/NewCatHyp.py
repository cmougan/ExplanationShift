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
rcParams.update({"font.size": 12})
import seaborn as sns
from sklearn.model_selection import train_test_split
from nobias import ExplanationShiftDetector
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from alibi_detect.cd import ChiSquareDrift, TabularDrift, ClassifierDrift
from category_encoders import TargetEncoder

# %%
data = pd.read_csv("stackO/survey_results_public.csv")
# %%
data = data[data.ConvertedCompYearly.isna() == False]
data = data.fillna(0)
data = data.drop(
    columns=[
        "CompFreq",
        "CompTotal",
        "SOPartFreq",
        "PlatformWantToWorkWith",
        "SurveyEase",
        "OfficeStackAsyncWantToWorkWith",
        "OfficeStackSyncWantToWorkWith",
        "Knowledge_3",
        "Accessibility",
        "Age",
        "VCInteraction",
        "TimeSearching",
        "Trans",
        "MentalHealth",
        "Onboarding",
        "TimeAnswering",
        "TrueFalse_3",
        "Knowledge_4",
        "NEWSOSites",
        "Knowledge_1",
        "VersionControlSystem",
        "Knowledge_5",
        "Frequency_1",
        "Knowledge_6",
        "Blockchain",
        "Frequency_3",
        "Gender",
        "TBranch",
        "SOAccount",
        "Frequency_2",
        "TrueFalse_2",
        "VCHostingPersonal use",
        "VCHostingProfessional use",
    ]
)
# %%
te = TargetEncoder()


# %%
r = "India"
df_tr = data[data["Country"] != r]
# Train test split
X_tr, X_te, y_tr, y_te = train_test_split(
    df_tr.drop(columns=["ConvertedCompYearly"], axis=1),
    df_tr["ConvertedCompYearly"],
    test_size=0.5,
    random_state=42,
)
X_ood = data[data["Country"] == r].drop("ConvertedCompYearly", axis=1)


te = TargetEncoder()
X_tr = te.fit_transform(X_tr, y_tr)
X_te = te.transform(X_te)
X_ood = te.transform(X_ood)


i = 0.5
n_samples = X_ood.shape[0] - int(i * X_ood.shape[0])
n_samples_1 = n_samples

X_ = X_ood.loc[~X_ood.index.isin(X_ood.sample(n_samples).index)]
X_new = X_te.sample(n_samples, replace=False).append(X_).drop(columns=["Country"])
# %%
# XGB
xgb_list = []
xgb_param = np.linspace(1, 10, 10)
for i in xgb_param:
    detector = ExplanationShiftDetector(
        model=XGBRegressor(n_estimators=int(i), max_depth=int(i)),
        gmodel=LogisticRegression(),
    )
    detector.fit(X_tr.drop(columns=["Country"]), y_tr, X_new)

    value = detector.get_auc_val()

    xgb_list.append(value)
# %%
# Decision Tree
dt_list = []
dt_param = np.linspace(1, 10, 9)
for i in dt_param:
    detector = ExplanationShiftDetector(
        model=DecisionTreeRegressor(max_depth=int(i)), gmodel=LogisticRegression()
    )
    detector.fit(X_tr.drop(columns=["Country"]), y_tr, X_new)

    value = detector.get_auc_val()
    dt_list.append(value)
# %%
# Random Forest
rf_list = []
rf_param = np.linspace(1, 10, 10)
for i in rf_param:
    detector = ExplanationShiftDetector(
        model=RandomForestRegressor(n_estimators=int(i), max_depth=int(i)),
        gmodel=LogisticRegression(),
    )

    detector.fit(X_tr.drop(columns=["Country"]), y_tr, X_new)

    value = detector.get_auc_val()
    rf_list.append(value)


# %%
# Plot
plt.figure(figsize=(10, 8))
# XGB
plt.plot(xgb_param, xgb_list, label="XGB")
ci = 1.96 * np.std(xgb_list) / np.sqrt(len(xgb_param))
plt.fill_between(xgb_param, (xgb_list - ci), (xgb_list + ci), alpha=0.1)

# DT
plt.plot(dt_param, dt_list, label="Decision Tree")
ci = 1.96 * np.std(dt_list) / np.sqrt(len(dt_param))
plt.fill_between(dt_param, (dt_list - ci), (dt_list + ci), alpha=0.1)

# RF
plt.plot(rf_param, rf_list, label="Random Forest")
ci = 1.96 * np.std(rf_list) / np.sqrt(len(rf_param))
plt.fill_between(rf_param, (rf_list - ci), (rf_list + ci), alpha=0.1)

plt.xlabel("Max Depth/Hyperparameter")
plt.ylabel("Explanation Shift AUC")
plt.legend()
plt.title("Log. Reg. as Explanation Shift Detector")
plt.savefig("images/NewCategoryHyper.pdf", bbox_inches="tight")
plt.show()
# %%
#######################################
########################################
# XGB
xgb_list = []
xgb_param = np.linspace(1, 10, 10)
for i in xgb_param:
    detector = ExplanationShiftDetector(
        model=XGBRegressor(n_estimators=int(i), max_depth=int(i)),
        gmodel=XGBClassifier(),
    )
    detector.fit(X_tr.drop(columns=["Country"]), y_tr, X_new)

    value = detector.get_auc_val()

    xgb_list.append(value)
# %%
# Decision Tree
dt_list = []
dt_param = np.linspace(1, 10, 9)
for i in dt_param:
    detector = ExplanationShiftDetector(
        model=DecisionTreeRegressor(max_depth=int(i)), gmodel=XGBClassifier()
    )
    detector.fit(X_tr.drop(columns=["Country"]), y_tr, X_new)

    value = detector.get_auc_val()
    dt_list.append(value)
# %%
# Random Forest
rf_list = []
rf_param = np.linspace(1, 10, 10)
for i in tqdm(rf_param):
    detector = ExplanationShiftDetector(
        model=RandomForestRegressor(n_estimators=int(i), max_depth=int(i)),
        gmodel=XGBClassifier(),
    )

    detector.fit(X_tr.drop(columns=["Country"]), y_tr, X_new)

    value = detector.get_auc_val()
    rf_list.append(value)


# %%
# Plot
plt.figure(figsize=(10, 8))
# XGB
plt.plot(xgb_param, xgb_list, label="XGB")
ci = 1.96 * np.std(xgb_list) / np.sqrt(len(xgb_param))
plt.fill_between(xgb_param, (xgb_list - ci), (xgb_list + ci), alpha=0.1)

# DT
plt.plot(dt_param, dt_list, label="Decision Tree")
ci = 1.96 * np.std(dt_list) / np.sqrt(len(dt_param))
plt.fill_between(dt_param, (dt_list - ci), (dt_list + ci), alpha=0.1)

# RF
plt.plot(rf_param, rf_list, label="Random Forest")
ci = 1.96 * np.std(rf_list) / np.sqrt(len(rf_param))
plt.fill_between(rf_param, (rf_list - ci), (rf_list + ci), alpha=0.1)

plt.xlabel("Max Depth/Hyperparameter")
plt.ylabel("Explanation Shift AUC")
plt.legend()
plt.title("XGB as Explanation Shift Detector")
plt.savefig("images/NewCategoryHyperXGBSO.pdf", bbox_inches="tight")
plt.show()
# %%
