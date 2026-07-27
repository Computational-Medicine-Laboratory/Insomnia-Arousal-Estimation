import os
from collections import defaultdict
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import norm, ranksums, wilcoxon
from statsmodels.stats.multitest import multipletests

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["cm"]
plt.rcParams["font.size"] = 12


def main(exp_name, decoder="mpp_cont"):
    load_path = f"results/log/{exp_name}/{{subject}}_{{phase}}.mat"
    save_path = f"results/paper/stats/{decoder}"

    os.makedirs(save_path, exist_ok=True)

    # Dictionary for display
    units = {
        "x_bar": r"a.u.",
        "hai_bar": r"a.u.",
        "x_sigma": r"a.u.",
        "hai_sigma": r"a.u.",
    }
    symbols = {
        "x_bar": r"$\bar{x}$",
        "hai_bar": r"$\bar{h}$",
        "x_sigma": r"$\hat\sigma_x$",
        "hai_sigma": r"$\hat\sigma_h$",
    }
    phase_names = {
        "cond": "Conditioning",
        "ext": "Extinction",
        "recall": "Recall",
    }
    primary_features = {"x_bar", "hai_bar"}

    # Get the list of subjects and phases
    dataset = pd.read_csv("dataset.csv")
    subjects = dataset["subject"].unique()
    groups = {subject: dataset[dataset["subject"] == subject]["group"].iloc[0] for subject in subjects}
    dataset = dataset.groupby(["subject", "phase"])

    # Get number of subjects for each phase and group
    phase_count = {phase: {"ID": 0, "GS": 0} for phase in phase_names.keys()}
    for (subject, phase), data in dataset:
        group = groups[subject]
        phase_count[phase][group] += 1
    df_n_subjects = pd.DataFrame(
        [{"phase": phase_names[phase], "ID": count["ID"], "GS": count["GS"]} for phase, count in phase_count.items()],
        columns=["phase", "ID", "GS"],
    )
    df_n_subjects.to_csv(os.path.join(save_path, "n_subjects.csv"), index=False)

    # Initialize dictionary {group: {phase: {feature: [value]}}}
    features = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Extract features
    for subject in subjects:
        group = groups[subject]
        for phase in phase_names.keys():
            if (subject, phase) in dataset.groups:
                # Load result
                result = loadmat(load_path.format(subject=subject, phase=phase))

                x = result["x_smth"].squeeze(0)
                v = result["v_smth"].squeeze(0)

                x_bar = np.mean(x)
                x_sigma = np.std(x)

                hai = 1 - norm.cdf(np.percentile(x, 50), x, np.sqrt(v))
                hai_bar = np.mean(hai)
                hai_sigma = np.std(hai)
            else:
                # Use nan for excluded data for pairwise tests
                x_bar, x_sigma, hai_bar, hai_sigma = np.nan, np.nan, np.nan, np.nan

            # Extract features
            features[group][phase]["x_bar"].append(x_bar)
            features[group][phase]["x_sigma"].append(x_sigma)
            features[group][phase]["hai_bar"].append(hai_bar)
            features[group][phase]["hai_sigma"].append(hai_sigma)

    # Convert to arrays
    for group in features.keys():
        for phase in features[group].keys():
            for feature in features[group][phase].keys():
                features[group][phase][feature] = np.array(features[group][phase][feature])

    # Box plots
    for phase in phase_names.keys():
        for feature in features["ID"][phase].keys():
            ID_feature = features["ID"][phase][feature]
            GS_feature = features["GS"][phase][feature]
            ID_feature = ID_feature[~np.isnan(ID_feature)]
            GS_feature = GS_feature[~np.isnan(GS_feature)]
            plt.figure(figsize=(3, 3))
            plt.boxplot([ID_feature, GS_feature], tick_labels=["ID", "GS"])
            plt.ylabel(f"{symbols[feature]} ({units[feature]})")
            plt.title(f"{symbols[feature]} in {phase_names[phase]} Phase")
            plt.subplots_adjust(left=0.25)
            plt.savefig(os.path.join(save_path, f"{feature}_{phase}.pdf"))
            plt.close()

    # Features for each phase
    for phase in phase_names.keys():
        rows = []
        for feature in features["ID"][phase].keys():
            ID_feature = features["ID"][phase][feature]
            GS_feature = features["GS"][phase][feature]
            ID_feature = ID_feature[~np.isnan(ID_feature)]
            GS_feature = GS_feature[~np.isnan(GS_feature)]

            # Effect size r = z / sqrt(n_total)
            stat_res = ranksums(ID_feature, GS_feature)
            p_val = stat_res.pvalue
            z_stat = stat_res.statistic
            n_total = len(ID_feature) + len(GS_feature)
            effect_size_r = z_stat / np.sqrt(n_total)

            rows.append(
                {
                    "primary": feature in primary_features,
                    "feature": f"{symbols[feature]} ({units[feature]})",
                    "IDmedian": np.median(ID_feature),
                    "IDlower": np.percentile(ID_feature, 25),
                    "IDupper": np.percentile(ID_feature, 75),
                    "GSmedian": np.median(GS_feature),
                    "GSlower": np.percentile(GS_feature, 25),
                    "GSupper": np.percentile(GS_feature, 75),
                    "p": p_val,
                    "r": effect_size_r,
                }
            )

        df_features = pd.DataFrame(rows)
        df_features_primary = df_features[df_features["primary"]].copy()

        # p-value correction
        _, df_features_primary["p"], _, _ = multipletests(df_features_primary["p"], alpha=0.05, method="bonferroni")
        _, df_features["p"], _, _ = multipletests(df_features["p"], alpha=0.05, method="bonferroni")

        df_features_primary.to_csv(
            os.path.join(save_path, f"primary_features_{phase}.csv"), index=False, float_format="%.4f"
        )
        df_features.to_csv(os.path.join(save_path, f"features_{phase}.csv"), index=False, float_format="%.4f")

    # Features difference between phases
    phases_combinations = list(combinations(phase_names.keys(), 2))
    for phase1, phase2 in phases_combinations:
        rows = []
        for feature in features["ID"][phase1].keys():
            ID_diff = features["ID"][phase2][feature] - features["ID"][phase1][feature]
            GS_diff = features["GS"][phase2][feature] - features["GS"][phase1][feature]
            ID_diff = ID_diff[~np.isnan(ID_diff)]
            GS_diff = GS_diff[~np.isnan(GS_diff)]

            # Effect size r = z / sqrt(n_total)
            stat_res = ranksums(ID_diff, GS_diff)
            p_val = stat_res.pvalue
            z_stat = stat_res.statistic
            n_total = len(ID_diff) + len(GS_diff)
            effect_size_r = z_stat / np.sqrt(n_total)

            rows.append(
                {
                    "primary": feature in primary_features,
                    "feature": f"{symbols[feature]} ({units[feature]})",
                    "IDmedian": np.median(ID_diff),
                    "IDlower": np.percentile(ID_diff, 25),
                    "IDupper": np.percentile(ID_diff, 75),
                    "IDp": wilcoxon(ID_diff).pvalue,
                    "GSmedian": np.median(GS_diff),
                    "GSlower": np.percentile(GS_diff, 25),
                    "GSupper": np.percentile(GS_diff, 75),
                    "GSp": wilcoxon(GS_diff).pvalue,
                    "p": p_val,
                    "r": effect_size_r,
                }
            )
        df_features = pd.DataFrame(rows)
        df_features_primary = df_features[df_features["primary"]].copy()

        # p-value correction
        _, df_features_primary["IDp"], _, _ = multipletests(df_features_primary["IDp"], alpha=0.05, method="bonferroni")
        _, df_features_primary["GSp"], _, _ = multipletests(df_features_primary["GSp"], alpha=0.05, method="bonferroni")
        _, df_features_primary["p"], _, _ = multipletests(df_features_primary["p"], alpha=0.05, method="bonferroni")
        _, df_features["IDp"], _, _ = multipletests(df_features["IDp"], alpha=0.05, method="bonferroni")
        _, df_features["GSp"], _, _ = multipletests(df_features["GSp"], alpha=0.05, method="bonferroni")
        _, df_features["p"], _, _ = multipletests(df_features["p"], alpha=0.05, method="bonferroni")

        df_features_primary.to_csv(
            os.path.join(save_path, f"primary_features_{phase1}_{phase2}.csv"), index=False, float_format="%.4f"
        )
        df_features.to_csv(os.path.join(save_path, f"features_{phase1}_{phase2}.csv"), index=False, float_format="%.4f")

    n_subjects_diff = {}
    for phase1, phase2 in phases_combinations:
        n_subjects_diff[f"ID{phase1}{phase2}"] = (
            ~np.isnan(features["ID"][phase1][feature]) & ~np.isnan(features["ID"][phase2][feature])
        ).sum()
        n_subjects_diff[f"GS{phase1}{phase2}"] = (
            ~np.isnan(features["GS"][phase1][feature]) & ~np.isnan(features["GS"][phase2][feature])
        ).sum()
    df_n_subjects_diff = pd.DataFrame(n_subjects_diff, index=[0])
    df_n_subjects_diff.to_csv(os.path.join(save_path, "n_subjects_diff.csv"), index=False)


if __name__ == "__main__":
    main("mpp_cont_best", decoder="mpp_cont")
    main("mpp_best", decoder="mpp")
