# %%
# Import candidate models
from doubt import Boot, QuantileRegressor, QuantileRegressionForest
from sklearn.linear_model import (LinearRegression, PoissonRegressor, 
                                  GammaRegressor, HuberRegressor)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Import datasets
from doubt.datasets import (Airfoil, Blog, Concrete, CPU, 
                            FacebookComments, FishBioconcentration,
                            FishToxicity, ForestFire, NewTaipeiHousing,
                            PowerPlant, Protein, Servo,
                            SpaceShuttle, Yacht)

# Import external libraries
import pandas as pd
import numpy as np
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt; plt.style.use('ggplot')

import warnings
from collections import defaultdict

from matplotlib import rcParams
plt.style.use('seaborn-whitegrid')
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['figure.figsize'] = 16,8
plt.rcParams['figure.figsize'] = [10, 5]

# Import internal classes
from distributions import DistributionShift
from tqdm.notebook import tqdm
from xgboost import XGBRegressor
from tabulate import tabulate

import shap

from utils import initialise_plot

# %%
dataset_classes = [
    Airfoil,
    #Concrete,
    #FishToxicity,
    #ForestFire,
    #NewTaipeiHousing,
    #PowerPlant,
    #Protein,
    #Servo,
]
for dataset in dataset_classes:
    print(dataset().shape)
# %%
def monitoring_plot(dataset, base_regressor: type, n_boots: int = 20, **kwargs):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Initialise the scaler
        standard_scaler = StandardScaler()

        # Load the dataset and split it
        X, _, y, _ = dataset().split(test_size=0.001, random_seed=4242)

        # Scale the dataset
        X = standard_scaler.fit_transform(X)

        # Back to dataframe
        X = pd.DataFrame(X, columns=["Var %d" % (i + 1) for i in range(X.shape[1])])
        data = X.copy()
        data["target"] = y

        # Train test splitting points
        fracc = 0.33
        oneThird = int(data.shape[0] * fracc)
        twoThird = data.shape[0] - int(data.shape[0] * fracc)
        print(
            "Number of rows {}, 1split {}, 2nd split {}".format(
                data.shape[0], oneThird, twoThird
            )
        )

        # Initialize plots
        num_rows = X.shape[1] // 3
        if X.shape[1] % 3 > 0:
            num_rows += 1
        fig, axs = initialise_plot(
            num_rows=num_rows,
            num_cols=3,
            base_regressor=base_regressor,
            dataset=dataset,
        )

        shap_res = []

        for idx, col in tqdm(enumerate(X.columns), total=len(X.columns)):
            values = defaultdict(list)

            # Sort data on the column
            data = data.sort_values(col).reset_index(drop=True).copy()

            # Train Test Split
            data_sub = data.iloc[:oneThird]
            data_train = data.iloc[oneThird:twoThird]
            data_up = data.iloc[twoThird:]

            X_tot = data.drop(columns="target")
            X_tr = data_train.drop(columns="target")
            X_sub = data_sub.drop(columns="target")
            X_up = data_up.drop(columns="target")

            y_tot = data[["target"]].target.values
            y_tr = data_train[["target"]].target.values
            y_sub = data_sub[["target"]].target.values
            y_up = data_up[["target"]].target.values

            # Fit the regressor
            regressor = base_regressor()
            regressor.fit(X_tr, y_tr)
            ## Regressor on all data
            regressor_full = base_regressor()
            regressor_full.fit(X_tot, y_tot)

            # Predictions
            preds = regressor.predict(X_tot)

            # explain the model's predictions using SHAP
            explainer = shap.Explainer(regressor)
            shap_values = explainer(X_tot)
            shap_values = pd.DataFrame(data=shap_values.values, columns=X_tot.columns)

            ## Full data
            explainer_full = shap.Explainer(regressor_full)
            shap_values_full = explainer_full(X_tot)
            shap_values_full = pd.DataFrame(
                data=shap_values_full.values, columns=X_tot.columns
            )
            # return shap_values,shap_values_full

            # pred_up, interval_up = regressor.predict(X_up, uncertainty=0.05, n_boots=n_boots)
            # pred_tr, interval_tr = regressor.predict(X_tr, uncertainty=0.05, n_boots=n_boots)
            # print("Sub:", np.round(mean_absolute_error(y_sub, pred_sub), decimals=2))
            # print("train:", np.round(mean_absolute_error(y_tr, pred_tr), decimals=2))
            # print("up:", np.round(mean_absolute_error(y_up, pred_up), decimals=2))
            # print("total:", np.round(mean_absolute_error(y_tot, preds), decimals=2))

            # print("Sub Uncertainty:", np.round(np.mean(interval_sub[:, 1] - interval_sub[:, 0]), decimals=2))
            # print("tr Uncertainty:", np.round(np.mean(interval_tr[:, 1] - interval_tr[:, 0]), decimals=2))
            # print("Up Uncertainty:", np.round(np.mean(interval_up[:, 1] - interval_up[:, 0]), decimals=2))
            # print("Total Uncertainty:", np.round(np.mean(intervals[:, 1] - intervals[:, 0]), decimals=2))

            # Statistics
            df = pd.DataFrame(np.abs(preds - y_tot), columns=["error"])
            df["mean_shap"] = np.mean(np.abs(shap_values), axis=1)
            df["mean_shap_full"] = np.mean(np.abs(shap_values_full), axis=1)
            df["shap_diff"] = np.abs(df["mean_shap"] - df["mean_shap_full"])
            shap_res.append(df.shap_diff.mean())
            error_raw = df['error'].values


            ROLLING_STAT = 100
            ### Rolling window on all
            df = df.rolling(ROLLING_STAT, int(ROLLING_STAT * 0.5)).mean().dropna()

            ## Scaling
            df = df.dropna()
            df = pd.DataFrame(standard_scaler.fit_transform(df), columns=df.columns)
            
           

            # Convert to dic for plotting
            for index, col in enumerate(df[['error','shap_diff']].columns):
                values[col] = df[col]

            #uncertainty_res.append(mean_absolute_error(values["error"], values["uncertainty"]))
            #ks_res.append(mean_absolute_error(values["error"], values["ks"]))
            #psi_res.append(mean_absolute_error(values["error"], values["PSI"]))

            # Plotting
            for name, vals in values.items():
                if idx == 0:
                    axs[idx // 3, idx % 3].plot(vals, label=f"{name} values")
                else:
                    axs[idx // 3, idx % 3].plot(vals,label=f"{name} values")
                    

        plt.figure()
        plt.title("Error raw")
        plt.plot(error_raw)
        plt.show()
                
        plt.figure()
        plt.title("Error rolling window")
        plt.plot(df["error"].values)
        plt.xlabel('Sorted column index ')
        plt.ylabel('Absolute Error')
        plt.show()

        plt.figure()
        plt.title("Mean absolute shap value per row (RW)")
        plt.plot(df["mean_shap"].values, color="blue")
        plt.xlabel('Sorted column index ')
        plt.ylabel('Shap value contribution (absolute value)')
        plt.show()
        plt.figure()
        plt.title("Shap values between  (RW)")
        plt.plot(df["mean_shap"].values, color="blue", label="Partial Train")
        plt.plot(df["mean_shap_full"].values, color="k", label="Full Train")
        plt.plot(df["error"].values, color="red", label="Model Performance")
        plt.legend()
        plt.show()

        plt.figure()
        plt.title("Shap values between  (RW)")
        # plt.plot(df["mean_shap"].values,color='blue',label='Partial Train')
        # plt.plot(df["mean_shap_full"].values,color='k',label='Full Train')
        plt.plot(df["error"].values, color="red", label="Model Performance")
        plt.plot(
                    df["shap_diff"].values,
                    color="orange",
                    label="Shap difference(Explanation Error)",
                )
        plt.legend()
        plt.show()
        
        
        resultados = pd.DataFrame({"shap_res": shap_res})
        media = resultados.mean()
        std = resultados.std()
        resultados.loc["mean"] = media
        resultados.loc["std"] = std
        
        print(resultados.to_string())

        fig.legend()
        # plt.savefig("fig.png")
        plt.show()

# %%
for dataset in dataset_classes:
     monitoring_plot(dataset, XGBRegressor)
# %%
