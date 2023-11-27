# %%
# Import Folktables
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm

plt.style.use("seaborn-whitegrid")
rcParams["axes.labelsize"] = 14
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["figure.figsize"] = 16, 8
rcParams.update({"font.size": 22})
import seaborn as sns
from sklearn.model_selection import train_test_split
from skshift import ExplanationShiftDetector
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import roc_auc_score
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder

# %%
## Year 2022 https://insights.stackoverflow.com/survey
data = pd.read_csv("data/survey_results_public.csv")
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
        "ResponseId",
        "TrueFalse_1",
        "MainBranch",
        "SOComm",
        "Employment",
        "Currency",
        "Gender",
        "TBranch",
        "SOAccount",
        "Frequency_2",
        "TrueFalse_2",
        "VCHostingPersonal use",
        "VCHostingProfessional use",
        "LanguageHaveWorkedWith",
        "LanguageWantToWorkWith",
        "DatabaseHaveWorkedWith",
        "DatabaseWantToWorkWith",
        "PlatformHaveWorkedWith",
        "WebframeHaveWorkedWith",
        "WebframeWantToWorkWith",
        "MiscTechHaveWorkedWith",
        "MiscTechWantToWorkWith",
        "ToolsTechHaveWorkedWith",
        "ToolsTechWantToWorkWith",
        "NEWCollabToolsHaveWorkedWith",
        "NEWCollabToolsWantToWorkWith",
        "OpSysProfessional use",
        "OpSysPersonal use",
        "OfficeStackAsyncHaveWorkedWith",
        "OfficeStackSyncHaveWorkedWith",
    ]
)
# %%
country = data[["Country"]]
X = data.drop(columns=["ConvertedCompYearly", "Country"], axis=1)
y = data["ConvertedCompYearly"]
# Encode
X = TargetEncoder().fit_transform(X, y)
X["Country"] = country
# %%
# Select the most frequent country
top1 = country.value_counts().head(1).index[0][0]
# Select the 2:6 most frequent countries
top5 = country.value_counts().head(6).index[1:]
# %%
# Experiment
aucs = {}
for country in top5:
    countr = "France"
    # Train test val split
    X_ = X[X["Country"] == top1]
    y_ = y[X["Country"] == top1]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_.drop(columns="Country"), y_, test_size=0.25, random_state=42
    )
    X_te, X_val, y_te, y_val = train_test_split(
        X_val, y_val, test_size=0.5, random_state=42
    )

    ## OOD Data
    X_ood = X[X["Country"] == country[0]]

    # Concatenate the training and validation sets
    params = np.linspace(0.05, 0.99, 30)

    aucs_temp = []
    for i in tqdm(params):
        n_samples = X_ood.shape[0] - int(i * X_ood.shape[0])
        n_samples_1 = n_samples

        X_ = X_ood.loc[~X_ood.index.isin(X_ood.sample(n_samples).index)].drop(
            columns="Country"
        )
        # Split X_ into first two:half and second half
        X_1 = X_[int(X_.shape[0] / 2) :]
        X_2 = X_[: int(X_.shape[0] / 2)]

        X_2["ood"] = 1
        X_te["ood"] = 0
        X_new = X_te.append(X_2)

        # Fit models and ESD
        m = Lasso().fit(X_tr, y_tr)

        detector = ExplanationShiftDetector(
            model=m, gmodel=LogisticRegression(), masker=True, data_masker=X_tr
        )
        detector.fit_detector(X_val, X_1)
        # Calculate AUC
        val_ = roc_auc_score(
            X_new["ood"], detector.predict_proba(X_new.drop(columns="ood"))[:, 1]
        )
        aucs_temp.append(val_)

    aucs[country] = aucs_temp
# %%
# Plot
plt.figure(figsize=(10, 6))
for country in top5:

    plt.plot(params, aucs[country], label=country[0])
    ci = 1.96 * np.std(aucs[country]) / np.sqrt(len(params))

    plt.fill_between(params, (aucs[country] - ci), (aucs[country] + ci), alpha=0.1)
plt.xlabel("Fraction of data from previously unseen group", fontsize=20)
plt.ylabel("AUC of Explanation Shift Detector", fontsize=20)
plt.legend(fontsize=18)
plt.savefig("images/NewCategory_SO_linear.pdf", bbox_inches="tight")

