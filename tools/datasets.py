import pandas as pd
from sklearn.datasets import make_blobs
from folktables import (
    ACSDataSource,
    ACSTravelTime,
    ACSEmployment,
    ACSIncome,
    ACSMobility,
    ACSPublicCoverage,
)
import numpy as np

d = {
    "AGEP": "Age",
    "SCHL": "Education",
    "MAR": "Marital",
    "ESP": "Employment",
    "ST": "State",
    "POVPIP": "PovertyIncome",
    "MIG": "MobilityStat",
    "CIT": "Citizenship",
    "DIS": "Disability",
    "OCCP": "Occupation",
    "PUMA": "Area",
    "JWTR": "WorkTravel",
    "JWTRNS": "WorkTravel2",
    "RAC1P": "Race",
    "AGEP": "Age",
    "POWPUMA": "WorkPlace",
}


class GetData:
    """
    Example:
    from tools.datasets import GetData
    data = GetData(type="blobs")
    X, y, X_ood, y_ood = data.get_data()
    """

    def __init__(self, type: str = "blobs", N: int = 100000, datasets="ACSTravelTime"):
        self.type = type
        self.N = N
        self.X = None
        self.y = None
        self.X_tr = None
        self.X_te = None
        self.y_tr = None
        self.y_te = None
        self.X_ood = None
        self.y_ood = None

        self.supported_types = ["blobs", "synthetic", "real"]
        assert self.type in self.supported_types

        if datasets == "ACSTravelTime":
            self.dataset = ACSTravelTime
        elif datasets == "ACSEmployment":
            self.dataset = ACSEmployment
        elif datasets == "ACSIncome":
            self.dataset = ACSIncome
        elif datasets == "ACSMobility":
            self.dataset = ACSMobility
        elif datasets == "ACSPublicCoverage":
            self.dataset = ACSPublicCoverage

    def get_data(self):
        if self.type == "blobs":
            self.X, self.y = make_blobs(
                n_samples=2000, centers=2, n_features=5, random_state=0
            )
            self.X = pd.DataFrame(self.X, columns=["x1", "x2", "x3", "x4", "x5"])
            self.X_ood, self.y_ood = make_blobs(
                n_samples=self.N, centers=1, n_features=5, random_state=0
            )
            self.X_ood = pd.DataFrame(
                self.X_ood, columns=["x1", "x2", "x3", "x4", "x5"]
            )
        elif self.type == "synthetic":
            ## Synthetic data
            ### Normal
            sigma = 1
            rho = 0.5
            mean = [0, 0]
            cov = [[sigma, 0], [0, sigma]]
            samples = 5_000
            x1, x2 = np.random.multivariate_normal(mean, cov, samples).T
            x3 = np.random.normal(0, sigma, samples)
            # Different values
            mean = [0, 0]
            cov = [[sigma, rho], [rho, sigma]]
            x11, x22 = np.random.multivariate_normal(mean, cov, samples).T
            x33 = np.random.normal(0, sigma, samples)

            # Create Data
            df = pd.DataFrame(data=[x1, x2, x3]).T
            df.columns = ["Var%d" % (i + 1) for i in range(df.shape[1])]
            df["target"] = (
                df["Var1"] * df["Var2"] + df["Var3"] + np.random.normal(0, 0.1, samples)
            )
            X_ood = pd.DataFrame(data=[x11, x22, x33]).T
            self.X_ood.columns = ["Var%d" % (i + 1) for i in range(X_ood.shape[1])]
            self.y_ood = (
                X_ood["Var1"] * X_ood["Var2"]
                + X_ood["Var3"]
                + np.random.normal(0, 0.1, samples)
            )
            self.X = df.drop(columns="target")
            self.y = df["target"]
        elif self.type == "real":
            ## Real data based on US census data
            data_source = ACSDataSource(
                survey_year="2014", horizon="1-Year", survey="person"
            )

            try:
                acs_data = data_source.get_data(states=["CA"], download=False)
            except:
                acs_data = data_source.get_data(states=["CA"], download=True)
            X, y, group = self.dataset.df_to_numpy(acs_data)
            X = pd.DataFrame(X, columns=self.dataset.features).rename(columns=d)

            # OOD data
            acs_data = data_source.get_data(states=["NY"], download=True)
            X_ood, y_ood, group = self.dataset.df_to_numpy(acs_data)
            X_ood = pd.DataFrame(X_ood, columns=self.dataset.features).rename(columns=d)
            self.X = X.rename(columns=d)
            self.X_ood = X_ood.rename(columns=d)
            # Lets make smaller data for computational reasons
            self.X = X.head(self.N)
            self.y = y[: self.N]

            self.X_ood = X_ood.head(self.N)
            self.y_ood = y_ood[: self.N]

        else:
            raise ValueError("type must be one of {}".format(self.supported_types))

        return self.X, self.y, self.X_ood, self.y_ood

    def get_state(self, year: str = "2014", state: str = "NY", N: int = 100000):
        self.N = N
        # OOD data
        data_source = ACSDataSource(survey_year=year, horizon="1-Year", survey="person")
        try:
            acs_data = data_source.get_data(states=[state], download=False)
        except:
            acs_data = data_source.get_data(states=[state], download=True)
        X_ood, y_ood, _ = self.dataset.df_to_numpy(acs_data)
        X_ood = pd.DataFrame(X_ood, columns=self.dataset.features).rename(columns=d)
        self.X_ood = X_ood.head(self.N)
        self.y_ood = y_ood[: self.N]
        return self.X_ood, self.y_ood
