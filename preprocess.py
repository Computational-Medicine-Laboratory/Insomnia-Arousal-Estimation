import numpy as np
import pandas as pd
from scipy.io import loadmat

# Define the path to load data
load_path = "../bayesian-eda/results/log/best/{subject}_{phase}.mat"

# Get the list of subjects and phases
dataset = pd.read_csv("../bayesian-eda/dataset.csv")
dataset = dataset.groupby(["subject", "phase"])


all_df = []
for (subject, phase), data in dataset:
    group = data["group"].iloc[0]

    data_file = loadmat(load_path.format(subject=subject, phase=phase))

    y = data_file["y_obs"].squeeze()
    r = data_file["u"].squeeze()
    s = data_file["tonic"].squeeze()

    # Exclude under-stimulated subjects
    p = np.sum(r > 0) / len(r)
    if p < 0.001:
        continue

    df = pd.DataFrame({"y": y, "r": r, "s": s})
    df.insert(0, "subject", subject)
    df.insert(1, "phase", phase)
    df.insert(2, "group", group)

    all_df.append(df)

all_df = pd.concat(all_df, ignore_index=True)
all_df.to_csv("dataset.csv", index=False)
