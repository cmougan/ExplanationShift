import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from category_encoders import MEstimateEncoder
import numpy as np
from collections import defaultdict
import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def fit_predict(modelo, enc, data, target, test):
    pipe = Pipeline([("encoder", enc), ("model", modelo)])
    pipe.fit(data, target)
    return pipe.predict(test)


def auc_group(model, data, y_true, dicc, group: str = "", min_samples: int = 50):
    aux = data.copy()
    aux["target"] = y_true
    cats = aux[group].value_counts()
    cats = cats[cats > min_samples].index.tolist()
    cats = cats + ["all"]

    if len(dicc) == 0:
        dicc = defaultdict(list, {k: [] for k in cats})

    for cat in cats:
        if cat != "all":
            aux2 = aux[aux[group] == cat]
            preds = model.predict_proba(aux2.drop(columns="target"))[:, 1]
            truth = aux2["target"]
            dicc[cat].append(roc_auc_score(truth, preds))
        elif cat == "all":
            dicc[cat].append(roc_auc_score(y_true, model.predict_proba(data)[:, 1]))
        else:
            pass

    return dicc


def explain(xgb: bool = True):
    """
    Provide a SHAP explanation by fitting MEstimate and GBDT
    """
    if xgb:
        pipe = Pipeline(
            [("encoder", MEstimateEncoder()), ("model", GradientBoostingClassifier())]
        )
        pipe.fit(X_tr, y_tr)
        explainer = shap.Explainer(pipe[1])
        shap_values = explainer(pipe[:-1].transform(X_tr))
        shap.plots.beeswarm(shap_values)
        return pd.DataFrame(np.abs(shap_values.values), columns=X_tr.columns).sum()
    else:
        pipe = Pipeline(
            [("encoder", MEstimateEncoder()), ("model", LogisticRegression())]
        )
        pipe.fit(X_tr, y_tr)
        coefficients = pd.concat(
            [pd.DataFrame(X_tr.columns), pd.DataFrame(np.transpose(pipe[1].coef_))],
            axis=1,
        )
        coefficients.columns = ["feat", "val"]

        return coefficients.sort_values(by="val", ascending=False)


def calculate_cm(true, preds):
    # Obtain the confusion matrix
    cm = confusion_matrix(preds, true)

    #  https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    return TPR[0]


def metric_calculator(
    modelo, data: pd.DataFrame, truth: pd.DataFrame, col: str, group1: str, group2: str
):
    aux = data.copy()
    aux["target"] = truth

    # Filter the data
    g1 = data[data[col] == group1]
    g2 = data[data[col] == group2]

    # Filter the ground truth
    g1_true = aux[aux[col] == group1].target
    g2_true = aux[aux[col] == group2].target

    # Do predictions
    p1 = modelo.predict(g1)
    p2 = modelo.predict(g2)

    # Extract metrics for each group
    res1 = calculate_cm(p1, g1_true)
    res2 = calculate_cm(p2, g2_true)
    return res1 - res2


def plot_rolling(data, roll_mean: int = 5, roll_std: int = 20):
    aux = data.rolling(roll_mean).mean().dropna()
    stand = data.rolling(roll_std).quantile(0.05, interpolation="lower").dropna()
    plt.figure()
    for col in data.columns:
        plt.plot(aux[col], label=col)
        # plt.fill_between(aux.index,(aux[col] - stand[col]),(aux[col] + stand[col]),# color="b",alpha=0.1,)
    plt.legend()
    plt.show()


def scale_output(data):
    return pd.DataFrame(
        StandardScaler().fit_transform(data), columns=data.columns, index=data.index
    )


import numpy as np


def psi(expected, actual, buckettype="bins", buckets=10, axis=0):
    """Calculate the PSI (population stability index) across all variables
    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal
    Returns:
       psi_values: ndarray of psi values for each variable
    Author:
       Matthew Burke
       github.com/mwburke
       worksofchart.com
    """

    def _psi(expected_array, actual_array, buckets):
        """Calculate the PSI for a single variable
        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into
        Returns:
           psi_value: calculated PSI value
        """

        def scale_range(input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == "bins":
            breakpoints = scale_range(
                breakpoints, np.min(expected_array), np.max(expected_array)
            )
        elif buckettype == "quantiles":
            breakpoints = np.stack(
                [np.percentile(expected_array, b) for b in breakpoints]
            )

        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(
            expected_array
        )
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            """Calculate the actual PSI value from comparing the values.
            Update the actual value to a very small number if equal to zero
            """
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return value

        psi_value = np.sum(
            sub_psi(expected_percents[i], actual_percents[i])
            for i in range(0, len(expected_percents))
        )

        return psi_value

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = _psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = _psi(expected[:, i], actual[:, i], buckets)
        elif axis == 1:
            psi_values[i] = _psi(expected[i, :], actual[i, :], buckets)

    return psi_values


def loop_estimators(
    estimator_set: list,
    normal_data,
    normal_data_ood,
    shap_data,
    shap_data_ood,
    performance_ood,
    target,
    state: str,
    error_type: str,
    target_shift: bool = False,
    output_path: str = "",
):
    """
    Loop through the estimators and calculate the performance for each
    """
    res = []

    for estimator in estimator_set:
        ## ONLY DATA
        X_train, X_test, y_train, y_test = train_test_split(
            normal_data, target, test_size=0.33, random_state=42
        )
        estimator_set[estimator].fit(X_train, y_train)
        error_te = mean_absolute_error(estimator_set[estimator].predict(X_test), y_test)
        error_ood = mean_absolute_error(
            estimator_set[estimator].predict(normal_data_ood),
            np.nan_to_num(list(performance_ood.values())),
        )

        res.append([state, error_type, estimator, "Only Data", error_te, error_ood])
        if target_shift == False:
            #### ONLY SHAP
            X_train, X_test, y_train, y_test = train_test_split(
                shap_data, target, test_size=0.33, random_state=42
            )
            estimator_set[estimator].fit(X_train, y_train)
            error_te = mean_absolute_error(
                estimator_set[estimator].predict(X_test), y_test
            )
            error_ood = mean_absolute_error(
                estimator_set[estimator].predict(shap_data_ood),
                np.nan_to_num(list(performance_ood.values())),
            )

            res.append([state, error_type, estimator, "Only Shap", error_te, error_ood])

            ### SHAP + DATA
            X_train, X_test, y_train, y_test = train_test_split(
                pd.concat([shap_data, normal_data], axis=1),
                target,
                test_size=0.33,
                random_state=42,
            )
            estimator_set[estimator].fit(X_train, y_train)
            error_te = mean_absolute_error(
                estimator_set[estimator].predict(X_test), y_test
            )
            error_ood = mean_absolute_error(
                estimator_set[estimator].predict(
                    pd.concat([shap_data_ood, normal_data_ood], axis=1)
                ),
                np.nan_to_num(list(performance_ood.values())),
            )
            res.append(
                [state, error_type, estimator, "Data + Shap", error_te, error_ood]
            )

    folder = os.path.join("results", state + "_" + error_type + ".csv")
    columnas = ["state", "error_type", "estimator", "data", "error_te", "error_ood"]
    pd.DataFrame(res, columns=columnas).to_csv(folder, index=False)


def loop_estimators_fairness(
    estimator_set: list,
    normal_data,
    normal_data_ood,
    target_shift,
    target_shift_ood,
    shap_data,
    shap_data_ood,
    performance_ood,
    target,
    state: str,
    error_type: str,
    output_path: str = "",
):
    """
    Loop through the estimators and calculate the performance for each
    Particular fairness case
    """
    res = []

    for estimator in estimator_set:
        ## ONLY DATA
        X_train, X_test, y_train, y_test = train_test_split(
            normal_data, target, test_size=0.33, random_state=42
        )
        estimator_set[estimator].fit(X_train, y_train)
        error_te = mean_absolute_error(estimator_set[estimator].predict(X_test), y_test)
        error_ood = mean_absolute_error(
            estimator_set[estimator].predict(normal_data_ood),
            np.nan_to_num(performance_ood),
        )

        res.append([state, error_type, estimator, "Only Data", error_te, error_ood])

        #### ONLY SHAP
        X_train, X_test, y_train, y_test = train_test_split(
            shap_data, target, test_size=0.33, random_state=42
        )
        estimator_set[estimator].fit(X_train, y_train)
        error_te = mean_absolute_error(estimator_set[estimator].predict(X_test), y_test)
        error_ood = mean_absolute_error(
            estimator_set[estimator].predict(shap_data_ood),
            np.nan_to_num(performance_ood),
        )
        res.append([state, error_type, estimator, "Only Shap", error_te, error_ood])
        #### ONLY TARGET
        X_train, X_test, y_train, y_test = train_test_split(
            target_shift, target, test_size=0.33, random_state=42
        )
        estimator_set[estimator].fit(X_train, y_train)
        error_te = mean_absolute_error(estimator_set[estimator].predict(X_test), y_test)
        error_ood = mean_absolute_error(
            estimator_set[estimator].predict(target_shift_ood),
            np.nan_to_num(performance_ood),
        )
        res.append([state, error_type, estimator, "Only Target", error_te, error_ood])

        #### TARGET + DISTRIBUTION
        X_train, X_test, y_train, y_test = train_test_split(
            pd.concat([target_shift, normal_data], axis=1),
            target,
            test_size=0.33,
            random_state=42,
        )
        estimator_set[estimator].fit(X_train, y_train)
        error_te = mean_absolute_error(estimator_set[estimator].predict(X_test), y_test)
        error_ood = mean_absolute_error(
            estimator_set[estimator].predict(
                pd.concat([target_shift_ood, normal_data_ood], axis=1)
            ),
            np.nan_to_num(performance_ood),
        )
        res.append([state, error_type, estimator, "Data+Target", error_te, error_ood])
        ### SHAP + DATA
        X_train, X_test, y_train, y_test = train_test_split(
            pd.concat([shap_data, normal_data, target_shift], axis=1),
            target,
            test_size=0.33,
            random_state=42,
        )
        estimator_set[estimator].fit(X_train, y_train)
        error_te = mean_absolute_error(estimator_set[estimator].predict(X_test), y_test)
        error_ood = mean_absolute_error(
            estimator_set[estimator].predict(
                pd.concat([shap_data_ood, normal_data_ood, target_shift_ood], axis=1)
            ),
            np.nan_to_num(performance_ood),
        )
        res.append(
            [state, error_type, estimator, "Data+Target+Shap", error_te, error_ood]
        )

    folder = os.path.join("results", state + "_" + error_type + ".csv")
    columnas = ["state", "error_type", "estimator", "data", "error_te", "error_ood"]
    pd.DataFrame(res, columns=columnas).to_csv(folder, index=False)
