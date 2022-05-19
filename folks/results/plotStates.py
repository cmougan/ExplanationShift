# %%
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from collections import defaultdict

from pyparsing import col

df = pd.read_csv("folks/results/all_results_income.csv")
# %%
aux = df[df["error_type"] == "performance"]
aux = aux[aux["estimator"] == "Linear"]
aux = aux[aux["data"] == "Only Shap"]
fig = px.choropleth(
    aux.groupby(["state"]).min().reset_index(),
    locations="state",
    locationmode="USA-states",
    color="error_ood",
    color_continuous_scale="Reds",
    scope="usa",
    hover_name="state",
    hover_data=["error_ood"],
)
fig.show()
fig.write_image("images/performanceUS.svg", format="svg")
fig.write_image("images/performanceUS.png")
# %%
aux = df[df["error_type"] == "fairness"]
aux = aux[(aux["estimator"] == "Linear") & (aux["estimator"] == "XGBoost")]
aux = aux[aux["data"] == "Only Data"]
fig = px.choropleth(
    aux.groupby(["state"]).min().reset_index(),
    locations="state",
    locationmode="USA-states",
    color="error_ood",
    color_continuous_scale="Reds",
    scope="usa",
    hover_name="state",
    hover_data=["error_ood"],
)
fig.show()


# %%
aux = df[df["error_type"] == "performance"]
aux = aux[(aux["estimator"] == "Linear") | (aux["estimator"] == "Dummy")]

best = []
for state in aux["state"].unique():
    aux_state = aux[(aux["state"] == state) & (aux["estimator"] == "Linear")]
    # Estimators
    data = aux_state[aux_state["data"] == "Only Data"].error_ood.values
    target = aux_state[aux_state["data"] == "Only Target"].error_ood.values
    dataTarget = aux_state[aux_state["data"] == "Data+Target"].error_ood.values
    shap = aux_state[aux_state["data"] == "Only Shap"].error_ood.values
    all = aux_state[aux_state["data"] == "Data+Target+Shap"].error_ood.values

    # Dummy
    aux_state = aux[(aux["state"] == state) & (aux["estimator"] == "Dummy")]
    dummy = aux_state.error_ood.mean()
    d = {
        "Distribution Shift": data,
        "Prediction Shift": target,
        "Dist+Pred Shift": dataTarget,
        "Explanation Shift": shap,
        "Dist+Targ+Exp Shift": all,
        "dummy": dummy,
    }

    best.append([state, min(d, key=d.get)])

best = pd.DataFrame(best, columns=["state", "data"])
# %%
fig = px.choropleth(
    best,
    locations="state",
    locationmode="USA-states",
    color="data",
    # color_continuous_scale="Reds",
    scope="usa",
    hover_name="state",
    # hover_data=["error_ood"],
)
fig.show()
fig.write_image("images/best_method.png")

# %%
