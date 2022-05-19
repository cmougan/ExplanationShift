from pmlb import fetch_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, roc_auc_score
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
import re
import traceback
from fairtools.xaiUtils import ShapEstimator
import xgboost

warnings.filterwarnings("ignore")


def benchmark_experiment(datasets: list, model, classification: str = "classification"):

    assert classification in [
        "classification",
        "regression",
        "explainableAI",
    ], "Classification type introduced --{}-- does not match: classification,regression,explainableAI".format(
        classification
    )

    if classification == "classification":
        extension = "_clas"
    elif classification == "regression":
        extension = "_reg"
    elif classification == "explainableAI":
        extension = "_explain"
    else:
        raise "Classification type not contained"

    results = defaultdict()
    for i, dataset in enumerate(datasets):
        try:
            # Initialise the scaler
            standard_scaler = StandardScaler()

            # Load the dataset and split it
            X, y = fetch_data(dataset, return_X_y=True, local_cache_dir="data/")

            # Scale the dataset
            X = standard_scaler.fit_transform(X)
            if classification == False:
                y = standard_scaler.fit_transform(y.reshape(-1, 1))

            # Back to dataframe
            X = pd.DataFrame(X, columns=["Var %d" % (i + 1) for i in range(X.shape[1])])
            data = X.copy()
            data["target"] = y

            # Min and max data limits for the experiment
            if X.shape[0] < 100:
                continue
            if X.shape[0] > 100_000:
                continue

            # Train test splitting points
            fracc = 0.33
            oneThird = int(data.shape[0] * fracc)
            twoThird = data.shape[0] - int(data.shape[0] * fracc)

            for idx, col in tqdm(enumerate(X.columns), total=len(X.columns)):

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

                # Error Calculation
                if classification == "classification":
                    ## Test predictions
                    pred_test = cross_val_predict(
                        estimator=model,
                        X=X_tr,
                        y=y_tr,
                        cv=KFold(n_splits=5, shuffle=True, random_state=0),
                        method="predict_proba",
                    )[:, 1]

                    ## Train
                    model.fit(X_tr, y_tr)
                    pred_train = model.predict_proba(X_tr)[:, 1]

                    ## OOD
                    X_ood = X_sub.append(X_up)
                    y_ood = np.concatenate((y_sub, y_up))
                    pred_ood = model.predict_proba(X_ood)[:, 1]

                    train_error = roc_auc_score(y_tr, pred_train)
                    test_error = roc_auc_score(y_tr, pred_test)
                    ood_error = roc_auc_score(y_ood, pred_ood)
                    generalizationError = test_error - train_error
                    ood_performance = ood_error - test_error
                elif classification == "regression":
                    ## Test predictions
                    pred_test = cross_val_predict(
                        estimator=model,
                        X=X_tr,
                        y=y_tr,
                        cv=KFold(n_splits=5, shuffle=True, random_state=0),
                    )

                    ## Train
                    model.fit(X_tr, y_tr)
                    pred_train = model.predict(X_tr)

                    ## OOD
                    X_ood = X_sub.append(X_up)
                    y_ood = np.concatenate((y_sub, y_up))
                    pred_ood = model.predict(X_ood)

                    train_error = mean_squared_error(pred_train, y_tr)
                    test_error = mean_squared_error(pred_test, y_tr)
                    ood_error = mean_squared_error(pred_ood, y_ood)

                    generalizationError = test_error - train_error
                    ood_performance = ood_error - test_error
                elif classification == "explainableAI":
                    # Explainer predictor
                    se = ShapEstimator(model=xgboost.XGBRegressor())
                    shap_pred_tr = cross_val_predict(se, X_tr, y_tr, cv=3)
                    shap_pred_tr = pd.DataFrame(shap_pred_tr, columns=X_tr.columns)
                    shap_pred_tr = shap_pred_tr.add_suffix("_shap")
                    se.fit(X_tr, y_tr)

                    ## Test predictions
                    pred_test = cross_val_predict(
                        estimator=model,
                        X=shap_pred_tr,
                        y=y_tr,
                        cv=KFold(n_splits=5, shuffle=True, random_state=0),
                    )

                    ## Train
                    full_train = pd.concat(
                        [
                            X_tr.reset_index(drop=True),
                            shap_pred_tr.reset_index(drop=True),
                        ],
                        axis=1,
                    )
                    error = y_tr - pred_test
                    model.fit(full_train, error)
                    pred_train = model.predict(full_train)

                    ## Generate OOD Shap data
                    X_ood = X_sub.append(X_up)
                    y_ood = np.concatenate((y_sub, y_up))
                    shap_pred_ood = se.predict(X_ood)
                    shap_pred_ood = pd.DataFrame(shap_pred_ood, columns=X_tr.columns)
                    shap_pred_ood = shap_pred_ood.add_suffix("_shap")

                    ## OOD
                    full_ood = pd.concat(
                        [
                            X_ood.reset_index(drop=True),
                            shap_pred_ood.reset_index(drop=True),
                        ],
                        axis=1,
                    )
                    pred_ood = model.predict(full_ood)
                    train_error = mean_squared_error(pred_train, y_tr)
                    test_error = mean_squared_error(pred_test, y_tr)

                    ood_error = mean_squared_error(pred_ood, y_ood)

                    generalizationError = test_error - train_error
                    ood_performance = ood_error - test_error

                # Append Results
                model_name = str(type(model)).split(".")[-1]
                model_name = re.sub("[^A-Za-z0-9]+", "", model_name)
                name = dataset + "_column_" + col
                results[name] = [
                    train_error,
                    test_error,
                    ood_error,
                    generalizationError,
                    ood_performance,
                    model_name,
                ]

        except Exception:
            print(traceback.format_exc())
            print("Not Working:", dataset)
            pass

    df = pd.DataFrame(data=results).T
    df.columns = [
        "trainError",
        "testError",
        "oodError",
        "generalizationError",
        "oodPerformance",
        "model",
    ]
    df.to_csv("results/" + model_name + extension + ".csv")
