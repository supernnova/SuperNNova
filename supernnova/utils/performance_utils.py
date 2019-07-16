import re
import numpy as np
import pandas as pd
from sklearn import metrics


def performance_metrics(df, sample_target=0):
    """Get performance metrics
    AUC: only valid for binomial classification, input proba of highest label class.

    Args:
        df (pandas.DataFrame) (str): with columns [target, predicted_target, class1]
        (optional) sample_target (str): for SNIa sample default is target 0

    Returns:
        accuracy, auc, purity, efficiency,truepositivefraction
    """
    n_targets = len(np.unique(df["target"]))

    # Accuracy & AUC
    accuracy = metrics.balanced_accuracy_score(df["target"].astype(int), df["predicted_target"])
    accuracy = round(accuracy * 100, 2)
    if n_targets == 2:  # valid for biclass only
        auc = round(metrics.roc_auc_score(df["target"], df["class1"]), 4)
    else:
        auc = 0.0

    SNe_Ia = df[df["target"] == sample_target]
    SNe_CC = df[df["target"] != sample_target]
    TP = len(SNe_Ia[SNe_Ia["predicted_target"] == sample_target])
    FP = len(SNe_CC[SNe_CC["predicted_target"] == sample_target])

    P = len(SNe_Ia)
    N = len(SNe_CC)

    truepositivefraction = P / (P + N)
    purity = round(100 * TP / (TP + FP), 2) if (TP + FP) > 0 else 0
    efficiency = round(100.0 * TP / P, 2) if P > 0 else 0

    return accuracy, auc, purity, efficiency, truepositivefraction


def contamination_by_SNTYPE(df, settings, sample_target=0):
    """Get contamination contribution by each SN type in sample percentage

    Args:
        df (pandas.DataFrame) (str): with columns [target, predicted_target, SNTYPE]
        (optional) sample_target (str): for SNIa sample default is target 0

    Returns:
        df (pandas.DataFrame) (str): with columns [SNTYPE,contamination_percentage]
    """

    # Get contamination SNe
    df_cont = df[
        (df["target"] != sample_target) & (df["predicted_target"] == sample_target)
    ]
    sample_size = len(df[df["predicted_target"] == sample_target])
    # Get contamination percentage
    contribution_arr = []
    type_arr = []
    for typ in [int(t) for t in settings.sntypes.keys() if t != 101]:
        df_selection = df_cont[df_cont["SNTYPE"] == typ]
        if sample_size > 1:
            contribution_arr.append(round(100 * len(df_selection) / sample_size, 2))
        else:
            contribution_arr.append(0)
        type_arr.append(typ)

    # Save contamination into df
    df_contamination_by_type = pd.DataFrame()
    df_contamination_by_type["SNTYPE"] = np.array(type_arr)
    df_contamination_by_type["contamination_percentage"] = np.array(contribution_arr)

    return df_contamination_by_type


def get_quantity_vs_variable(
    quantity,
    variable,
    df,
    settings,
    contamination_by=None,
    nbins=10,
    intervals=False,
    mean_bins=False,
):
    """Get contamination/purity vs redshift

    Args:
        quantity (str): quantity to compute
        variable (str): variable for binning quantity
        df (pandas.DataFrame) (str):
            columns [SIM_REDSHIFT_CMB, target, predicted_target, class1/0(fraction of positives),SNTYPE]
        settings (str): superNNova settings
        contamination_by (str): if a contamination by a SNTYPE SNe is required
        nbins (str): number of bins to use for varaible binning
        intervals (str): instead of returning bin center, return right bound
        mean_bins (str): instead of returning bin center, return variable mean in bin

    Returns:
        bin center  (numpy array) (str): or right bound (interval option), variable mean in that bin (mean_bins option)
        quantity (numpy array)
    """

    # Slice in 10 redshift bins
    if "class" in variable:
        bins = np.arange(0, 11) / 10
        df["bin"], sliced_bins = pd.cut(df[variable], bins, retbins=True)
    else:
        df["bin"], sliced_bins = pd.cut(df[variable], nbins, retbins=True)

    sorted_df = df.sort_values(variable)
    bin_list = sorted_df["bin"].unique()

    quantity_list = []
    varbins_list = []
    for i, mybin in enumerate(bin_list):
        selected = sorted_df[sorted_df["bin"] == mybin]
        # get quantity
        if quantity == "contamination":
            cont_df = contamination_by_SNTYPE(selected, settings)
            if contamination_by:
                quant = cont_df[cont_df["SNTYPE"] == contamination_by][
                    "contamination_percentage"
                ].values[0]
            else:
                quant = cont_df["contamination_percentage"].sum()
        else:
            d = {}
            for key, value in zip(
                ["accuracy", "auc", "purity", "efficiency", "truepositivefraction"],
                performance_metrics(selected),
            ):
                d[key] = value
            quant = d[quantity]
        quantity_list.append(quant)

        # now the bin centers/mean/interval list
        if intervals:
            varbins_list.append(mybin.right)
        elif mean_bins:
            varbins_list.append(selected[variable].mean())
        else:
            varbins_list.append(mybin.left + (mybin.right - mybin.left) / 2.0)
    return np.array(varbins_list), np.array(quantity_list)


def reformat_df(df, key, keep=None, group_bayesian=False):
    """Change column names to format used by performance functions

    Args:
        df (pandas.DataFrame) (str): with columns [key_class*]
        keep (list) (str): list of keys to keep apart from ["target","SNTYPE","SNID","SIM_REDSHIFT_CMB"} if available

    Returns:
        df (pandas.DataFrame) (str): with columns [class*, predicted_target]
    """
    # Key will change depending if around max or complete lc
    if key != "all" and key != "probability":
        format_key = f"PEAKMJD{key}"
    else:
        format_key = key

    # new dataframe to save
    tmp_df = pd.DataFrame()
    # get class columns and reformat
    ori_class_columns = [k for k in df.keys() if f"{format_key}_class" in k]
    tmp_df[[k.split("_")[-1] for k in ori_class_columns]] = df[ori_class_columns].copy()
    # get the rest of features
    cols_to_keep = [
        k for k in ["target", "SNTYPE", "SNID", "SIM_REDSHIFT_CMB"] if k in df.keys()
    ]
    if keep:
        cols_to_keep += keep
    tmp_df[cols_to_keep] = df[cols_to_keep].copy()
    # if variational or bayesian we can get the median prediction only
    if group_bayesian:
        tmp_df = tmp_df.groupby("SNID", as_index=False).median()
    # protect against NaNs (e.g. too early classifications)
    tmp_df = tmp_df[~np.isnan(tmp_df["class0"])]
    # set predicted target to max value in columns
    tmp_df["predicted_target"] = (
        tmp_df[[k for k in tmp_df.keys() if "class" in k]]
        .idxmax(axis=1)
        .str.strip("class")
        .astype(int)
    )
    return tmp_df


def create_latex_accuracy_singletable(df, outname, title):
    """Create latex tables from summary stats selection
    Table of metric by redshift given and early classification

    Args:
        df (pandas.DataFrame) : dataframe to convert
        outname (str) : name of latex table
        (optional) metric (str): which metric to output, default is accuracy

    Returns:
        Latex table: to write

    """

    # Get redshifts and accuracies in good format
    df = df.copy()
    df["redshift"] = df["model_name_noseed"].apply(
        lambda x: re.search(r"(?<=R\_)[A-Za-z]+", x).group()
    )
    metric = "accuracy"
    list_keys = ["-2", "0", "+2", "all"]
    for k in list_keys:
        df[k] = (
            "$"
            + df[f"{k}_{metric}_mean"].round(2).map(str)
            + " \pm "
            + df[f"{k}_{metric}_std"].round(2).map(str)
            + "$"
        )

    to_write = df.to_latex(index=False, columns=["redshift"] + list_keys, escape=False)

    # Formatting table for MNRAS
    title_to_use = "\n\\multicolumn{5}{c}{" + title + "} \\\\\n"
    to_write = to_write.replace("\n", title_to_use, 1)
    to_write = (
        to_write.replace("toprule", "hline")
        .replace("midrule", "hline")
        .replace("bottomrule", "hline")
        .replace("{lllll}", "{l  cccc }")
    )

    return to_write
