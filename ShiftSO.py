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
from skshift import ExplanationShiftDetector
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import LogisticRegression
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
country = data[['Country']]
X = data.drop(columns=["ConvertedCompYearly",'Country'], axis=1)
y = data["ConvertedCompYearly"]
# Encode
X = TargetEncoder().fit_transform(X, y)
X['Country'] = country
# %%
# Select the most frequent country
top1 = country.value_counts().head(1).index[0][0]
# Select the 2:6 most frequent countries
top5 = country.value_counts().head(6).index[1:]
# %%
# Train OOD split
# Train test split
X_ = X[X['Country'] == top1]
y_ = y[X['Country'] == top1]
X_tr, X_te, y_tr, y_te = train_test_split(
    X_.drop(columns = 'Country'), y_, test_size=0.5, random_state=42
)
## OOD Data
X_ood = X[X['Country'] != top1]
y_ood = y[X['Country'] != top1]

# Train model
model = XGBRegressor()
model.fit(X_tr, y_tr)

# %%
for country in top5:
    X_ood_ = X_ood[X_ood['Country'] == country[0]]
    y_ood_ = y_ood[X_ood['Country'] == country[0]]

    # OOD Train test split
    X_ood_tr, X_ood_te, y_ood_tr, y_ood_te = train_test_split(
        X_ood_.drop(columns = 'Country'), y_ood_, test_size=0.5, random_state=42
    )
    # Fit Inspector
    detector = ExplanationShiftDetector(model=model, gmodel=LogisticRegression())

    detector.fit_detector(X_te, X_ood_tr)
    print(roc_auc_score(y_ood_te, detector.predict_proba(X_ood_te)[:, 1]))

    break

# %%
detector = ExplanationShiftDetector(model=model, gmodel=LogisticRegression())
# %%
detector.fit_detector(X_te, X_ood_tr)
# %%
model = XGBRegressor().fit(X_tr, y_tr)

detector = ExplanationShiftDetector(model=XGBRegressor(), gmodel=XGBClassifier())

detector.fit_pipeline(X_tr, y_tr, X_ood_tr)
# %%
