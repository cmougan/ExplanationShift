# %%
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")
# %%
files = os.listdir()

data = pd.DataFrame()
for file in files:
    if file.endswith("performance.csv"):
        data = data.append(pd.read_csv(file))
data.to_csv("all_results.csv", index=False)
