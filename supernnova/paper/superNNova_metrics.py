import numpy as np
import pandas as pd
from pathlib import Path
from tabulate import tabulate
from ..utils import data_utils as du
from ..utils import logging_utils as lu
from ..utils import performance_utils as pu


"""
Useful functions for metrics quoted in SuperNNova

"""


def select_df(df, sel_criteria, data=None):
    """Select a subsample of a pandas Dataframe
    Valid only for one to two selection criteria

    Args:
        df (Pandas.Dataframe)
        sel_criteria (list): selection criteria
        data (optional): if source_data must be override
    """
    if data:
        df = df[df["source_data"].str.contains(data)]
    else:
        df = df

    n_crit = len(sel_criteria)

    if n_crit == 1:
        df_sel = df[df["model_name_noseed"].str.contains(sel_criteria[0])]
        print(f"{sel_criteria[0]}")
    elif n_crit == 2:
        df_sel = df[
            (df["model_name_noseed"].str.contains(sel_criteria[0]))
            & df["model_name_noseed"].str.contains(sel_criteria[1])
        ]
        print(f"{sel_criteria[0]} {sel_criteria[1]} ")

    return df_sel


def acc_auc_df(df, model_names_list, data=None):
    df_sel = df[df["model_name_noseed"].isin(model_names_list)]
    if data:
        df_sel = df_sel[df_sel["source_data"].str.contains(data)]
    else:
        df_sel = df_sel
    df.round(
        {
            "all_accuracy_mean": 2,
            "all_accuracy_std": 2,
            "all_auc_mean": 4,
            "all_auc_std": 4,
        }
    )
    nice_df_print(
        df_sel,
        keys=[
            "model_name_noseed",
            "all_accuracy_mean",
            "all_accuracy_std",
            "all_auc_mean",
            "all_auc_std",
        ],
    )


def get_metric_ranges(df, sel_criteria, metric="all_accuracy", round_output=2):
    df_sel = select_df(df, sel_criteria)
    mean_metric = np.round(df_sel[f"{metric}_mean"].mean(), round_output)
    std_metric = np.round(df_sel[f"{metric}_mean"].std())
    top_models = df_sel.nlargest(3, f"{metric}_mean").round(round_output)

    print(f"mean of {mean_metric} \pm {std_metric}")
    print(f"top models:")
    nice_df_print(
        top_models, keys=["model_name_noseed", f"{metric}_mean", f"{metric}_std"]
    )

    return df_sel


def nice_df_print(df, keys="keys"):
    if keys != "keys":
        df = df[keys]
    print(tabulate(df, headers=keys, tablefmt="simple", showindex=False))


def get_delta_metrics(df_stats, settings):
    """Difference between models in SuperNNova paper.

    BEWARE: selection hard coded
    
    Args:
        df (pandas.DataFrame): dataframe containing summary stats
        settings (ExperimentSettings): custom class to hold hyperparameters
    Returns:
        df (pandas.DataFrame): dataframe containing delta metrics
    """

    list_metrics = [
        "all_accuracy_mean",
        "mean_all_class0_std_dev_mean",
        "all_entropy_mean",
    ]

    list_df_delta = []

    # Build pairs of models based on available data
    list_config = df_stats[["model_name_noseed", "source_data"]].values.tolist()
    list_pair = []
    # Look for photometry + saltfit
    list_pair += [
        (
            c,
            [c[0].replace("saltfit_DF_1.0", "photometry_DF_0.43"), "photometry"],
            "salt_phot",
        )
        for c in list_config
        if "saltfit_DF_1.0" in c[0]
        and "photometry" in c[1]
        and [c[0].replace("saltfit_DF_1.0", "photometry_DF_0.43"), "photometry"]
        in list_config
    ]

    # Look for saltfit + saltfit
    list_pair += [
        (c, [c[0].replace("saltfit_DF_0.5", "saltfit_DF_1.0"), "saltfit"], "salt_salt")
        for c in list_config
        if "saltfit_DF_0.5" in c[0]
        and "saltfit" in c[1]
        and [c[0].replace("saltfit_DF_0.5", "saltfit_DF_1.0"), "saltfit"] in list_config
    ]

    # Look for photometry + photometry
    list_pair += [
        (
            c,
            [c[0].replace("photometry_DF_0.43", "photometry_DF_1.0"), "photometry"],
            "phot_phot",
        )
        for c in list_config
        if "photometry_DF_0.43" in c[0]
        and "photometry" in c[1]
        and [c[0].replace("photometry_DF_0.43", "photometry_DF_1.0"), "photometry"]
        in list_config
    ]

    for (c0, c1, delta_type) in list_pair:
        df1 = df_stats[
            (df_stats["model_name_noseed"] == c0[0])
            & (df_stats["source_data"] == c0[1])
        ].reset_index(drop=True)
        df2 = df_stats[
            (df_stats["model_name_noseed"] == c1[0])
            & (df_stats["source_data"] == c1[1])
        ].reset_index(drop=True)

        df_delta = (df1[list_metrics] - df2[list_metrics]).reset_index(drop=True)
        df_delta = df_delta.add_suffix("_delta")
        df_delta["model_name_left"] = df1["model_name_noseed"]
        df_delta["model_name_right"] = df2["model_name_noseed"]
        df_delta["delta_type"] = delta_type

        list_df_delta.append(df_delta)

    if len(list_df_delta) != 0:

        orderded_columns = ["model_name_left", "model_name_right", "delta_type"] + list(
            map(lambda x: f"{x}_delta", list_metrics)
        )

        df_delta = pd.concat(list_df_delta).reset_index(drop=True)
        df_delta = df_delta[orderded_columns]

        df_delta.to_csv(
            Path(settings.stats_dir) / "summary_stats_delta.csv", index=False
        )
    else:
        df_delta = pd.DataFrame(
            columns=[
                "model_name_left",
                "model_name_right",
                "delta_type",
                "all_accuracy_mean_delta","mean_all_class0_std_dev_mean_delta",
                "all_entropy_mean_delta",
            ]
        )

    # Also look for difference between OOD / not OOD
    orderded_columns = ["model_name_noseed"]

    for OOD in du.OOD_TYPES:
        df_stats[f"{OOD}_delta_entropy"] = (
            df_stats["all_entropy_mean"] - df_stats[f"all_{OOD}_entropy_mean"]
        )
        df_stats[f"{OOD}_delta_std_dev"] = (
            df_stats["mean_all_class0_std_dev_mean"]
            - df_stats[f"mean_all_{OOD}_class0_std_dev_mean"]
        )

        orderded_columns += [f"{OOD}_delta_entropy", f"{OOD}_delta_std_dev"]

    df_delta_ood = df_stats[orderded_columns]
    df_delta_ood.to_csv(
        Path(settings.stats_dir) / "summary_stats_delta_OOD.csv", index=False
    )
    return df_delta, df_delta_ood


def create_accuracy_latex_tables(df, settings):
    """Latex accuracy tables for paper.

    BEWARE: Selection is hardcoded

    Args:
        df (pandas.DataFrame) : summary statistics df
        settings (ExperimentSettings): custom class to hold hyperparameters
    """

    tables_to_plot = {
        "accuracies_biclass.tex": {
            "list_criteria": [
                [
                    "vanilla",
                    "_CLF_2",
                    "saltfit",
                    "_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean_C",
                ],
                [
                    "vanilla",
                    "_CLF_2",
                    "photometry",
                    "_DF_0.43_N_global_lstm_32x2_0.05_128_True_mean_C",
                ],
                [
                    "vanilla",
                    "_CLF_2",
                    "photometry",
                    "_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean_C",
                ],
            ],
            "list_title": [
                "SALT2 fitted dataset",
                "$43 \\%$ of complete dataset",
                "Complete dataset",
            ],
        },
        "accuracies_multiclass.tex": {
            "list_criteria": [
                [
                    "vanilla",
                    "_CLF_3",
                    "photometry",
                    "_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean_C",
                ],
                [
                    "vanilla",
                    "_CLF_7",
                    "photometry",
                    "_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean_C",
                ],
            ],
            "list_title": ["Ternary classification", "Seven-way classification"],
        },
    }

    for tabl in tables_to_plot.keys():
        to_write = []
        for i in range(len(tables_to_plot[tabl]["list_criteria"])):
            df_sel = df.copy()
            for j in range(len(tables_to_plot[tabl]["list_criteria"][i])):
                df_sel = df_sel[
                    df_sel["model_name_noseed"].str.contains(
                        tables_to_plot[tabl]["list_criteria"][i][j]
                    )
                ]
            df_sel = df_sel[
                df_sel["source_data"].str.contains(
                    tables_to_plot[tabl]["list_criteria"][i][2]
                )
            ]
            to_write.append(
                pu.create_latex_accuracy_singletable(
                    df_sel,
                    "".join(tables_to_plot[tabl]["list_criteria"][i]),
                    tables_to_plot[tabl]["list_title"][i],
                )
            )

        with open(f"{settings.latex_dir}/{tabl}", "w") as tf:
            for i in range(len(tables_to_plot[tabl]["list_criteria"])):
                tf.write(to_write[i])


def print_contamination(df, sel_criteria, settings, data="saltfit"):
    df_sel = select_df(df, sel_criteria, data=data)
    df_sel = df_sel.round(2)
    print(lu.str_to_bluestr(f"Contamination and efficiency {sel_criteria}"))
    for sntype in [k for k in settings.sntypes.keys() if k != 101]:
        key_list = [
            f"all_contamination_{sntype}_mean",
            f"all_contamination_{sntype}_std",
        ]
        df_sel[f"str_all_contamination_{sntype}"] = df_sel[key_list].apply(
            lambda x: " \pm ".join(x.map(str)), axis=1
        )
        key_list = [f"0_contamination_{sntype}_mean", f"0_contamination_{sntype}_std"]
        df_sel[f"str_0_contamination_{sntype}"] = df_sel[key_list].apply(
            lambda x: " \pm ".join(x.map(str)), axis=1
        )

    print("all")
    keys_to_use = ["model_name_noseed"] + [
        k
        for k in df_sel.keys()
        if "str_all_contamination_" in k or "all_efficiency_" in k or "all_purity_" in k
    ]
    nice_df_print(df_sel, keys=keys_to_use)
    print("Peakmjd")
    keys_to_use = ["model_name_noseed"] + [
        k
        for k in df_sel.keys()
        if "str_0_contamination_" in k or "0_efficiency_" in k or "0_purity_" in k
    ]
    nice_df_print(df_sel, keys=keys_to_use)
