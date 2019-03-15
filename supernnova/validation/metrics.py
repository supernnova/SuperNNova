import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from ..utils import data_utils as du
from ..utils import performance_utils as pu
from ..utils import logging_utils as lu

plt.switch_backend("agg")


def aggregate_metrics(settings):
    """Aggregate all pre-computed METRICS files into a single dataframe
    for analysis

    Save a csv dataframe aggregating all the metrics

    Args:
        settings (ExperimentSettings): custom class to hold hyperparameters

    """

    list_files = Path(f"{settings.models_dir}").glob("**/*METRICS*.pickle")
    list_files = list(map(str, list_files))
    assert len(list_files) != 0, lu.str_to_redstr(
        "No predictions found. Please train and validate randomforest and vanilla models"
    )

    # read all performance metrics
    list_df = []
    for f in list_files:
        df = pd.read_pickle(f)
        model_name = df["model_name"][0]
        source_data = df["source_data"][0]
        model_name_noseed = re.sub(r"S\_\d+_", "", model_name)
        model_name_noseed = f"{model_name_noseed}"
        df["model_name_noseed"] = model_name_noseed
        df["source_data"] = source_data
        list_df.append(df)

    df_all = pd.concat(list_df, axis=0, sort=True)

    # Groupby model and average over seed
    group_cols = ["model_name_noseed", "source_data"]
    df_mean = df_all.groupby(group_cols).mean().add_suffix("_mean")
    df_std = df_all.groupby(group_cols).std().add_suffix("_std")

    mean_columns = df_mean.columns
    std_columns = df_std.columns

    orderded_columns = [
        item for sublist in zip(mean_columns, std_columns) for item in sublist
    ]
    orderded_columns = ["model_name_noseed", "source_data"] + orderded_columns

    df_stats = pd.concat([df_mean, df_std], axis=1).reset_index()[orderded_columns]
    df_stats.to_csv(Path(settings.stats_dir) / "summary_stats.csv", index=False)


def get_metrics_singlemodel(settings, prediction_file=None, model_type="rnn"):
    """Launch computation of all evaluation metrics for a given model, specified
    by the settings object or by a model file

    Save a pickled dataframe (we pickle  because we're saving numpy arrays, which
    are not easily savable with the ``to_csv`` method).

    Args:
        settings (ExperimentSettings): custom class to hold hyperparameters
        prediction_file (str): Path to saved predictions. Default: ``None``
        model_type (str): Choose ``rnn`` or ``randomforest``

    Returns:
        (pandas.DataFrame) holds the performance metrics for this dataframe
    """

    df_SNinfo = du.load_HDF5_SNinfo(settings)
    host = pd.read_pickle(f"{settings.processed_dir}/hostspe_SNID.pickle")
    host_zspe_list = host["SNID"].tolist()

    if prediction_file is not None:
        # Overwrite dump_dir: use the one corresponding to specified model_file
        # Useful for representativeness studies
        dump_dir = str(Path(prediction_file).parent)
        # Also overwrite model name
        model_name = Path(prediction_file).parent.name
        metrics_file = str(prediction_file).replace("PRED_", "METRICS_")
        source_data = (
            "photometry" if "photometry" in Path(prediction_file).name else "saltfit"
        )
    else:
        model_name = (
            settings.pytorch_model_name
            if model_type == "rnn"
            else settings.randomforest_model_name
        )

        dump_dir = f"{settings.models_dir}/{model_name}"
        prediction_file = (
            f"{dump_dir}/" f"PRED_{model_name}.pickle"
        )
        metrics_file = (
            f"{dump_dir}/" f"METRICS_{model_name}.pickle"
        )
        source_data = settings.source_data

    assert os.path.isfile(prediction_file), lu.str_to_redstr(
        f"{prediction_file} DOES NOT EXIST"
    )

    df = pd.read_pickle(prediction_file)
    df = pd.merge(df, df_SNinfo[["SNID", "SNTYPE"]], on="SNID", how="left")

    list_df_metrics = []

    # Metrics shared between RF and RNN
    list_df_metrics.append(get_calibration_metrics_singlemodel(df))

    if model_type == "rnn":
        # RNN-specific metrics
        list_df_metrics.append(
            get_rnn_performance_metrics_singlemodel(settings, df, host_zspe_list)
        )
        list_df_metrics.append(get_uncertainty_metrics_singlemodel(df))
        list_df_metrics.append(get_entropy_metrics_singlemodel(df, settings.nb_classes))
        list_df_metrics.append(
            get_classification_stats_singlemodel(df, settings.nb_classes)
        )
    else:
        # RF-specific metrics
        list_df_metrics.append(
            get_randomforest_performance_metrics_singlemodel(
                settings, df, host_zspe_list
            )
        )

    df_metrics = pd.concat(list_df_metrics, 1)

    df_metrics["model_name"] = model_name
    df_metrics["source_data"] = source_data
    df_metrics.to_pickle(metrics_file)

    lu.print_green("Finished getting metrics ")


def get_rnn_performance_metrics_singlemodel(settings, df, host_zspe_list):
    """Compute performance metrics (accuracy, AUC, purity etc) for
    an RNN model

    - Compute metrics around peak light (i.e. ``PEAKMJD``) and for the full lightcurve.
    - For bayesian models, compute multiple predictions per lightcurve and then take the median

    Args:
        settings (ExperimentSettings): custom class to hold hyperparameters
        df (pandas.DataFrame): dataframe containing a model's predictions
        host_zspe_list (list): available host galaxy spectroscopic redshifts

    Returns:
        (pandas.DataFrame) holds the performance metrics for this dataframe
    """

    # Compute metrics around peak light, and with full lightcurve
    list_keys = ["-2", "", "+2"] + ["all"]
    perf_dic = {}
    for key in list_keys:
        # Need to select data (remove NAN) as sometimes, MJD happens too early
        # and MJD+(-2) (for instance) cannot be computed, hence NaN
        if key != "all":
            format_key = f"PEAKMJD{key}"
        else:
            format_key = key
        selection = df[~np.isnan(df[f"{format_key}_class1"])]
        if "bayesian" or "variational" in settings.pytorch_model_name:
            group_bayesian = True
        else:
            group_bayesian = False
        # general metrics
        # TODO refactor
        reformatted_selection = pu.reformat_df(
            selection, key, group_bayesian=group_bayesian
        )
        accuracy, auc, purity, efficiency, _ = pu.performance_metrics(
            reformatted_selection
        )
        contamination_df = pu.contamination_by_SNTYPE(reformatted_selection, settings)

        if key == "":
            savekey = "0"
        else:
            savekey = key
        perf_dic[f"{savekey}_accuracy"] = accuracy
        perf_dic[f"{savekey}_auc"] = auc
        perf_dic[f"{savekey}_purity"] = purity
        perf_dic[f"{savekey}_efficiency"] = efficiency
        for sntype, contamination_percentage in contamination_df.values:
            perf_dic[
                f"{savekey}_contamination_{int(sntype)}"
            ] = contamination_percentage

        # Reweighted for SNe with zspe
        zspe_df = selection[selection["SNID"].isin(host_zspe_list)]
        if len(zspe_df) > 0:
            zspe_df = pu.reformat_df(zspe_df, key, group_bayesian=group_bayesian)
            accuracy_zspe, auc_zspe, purity_zspe, efficiency_zspe, _ = pu.performance_metrics(
                zspe_df
            )
        else:
            accuracy_zspe, auc_zspe, purity_zspe, efficiency_zspe = (0.0, 0.0, 0.0, 0.0)

        perf_dic[f"{savekey}_zspe_accuracy"] = accuracy_zspe
        perf_dic[f"{savekey}_zspe_auc"] = auc_zspe
        perf_dic[f"{savekey}_zspe_purity"] = purity_zspe
        perf_dic[f"{savekey}_zspe_efficiency"] = efficiency_zspe

    # Create a dataframe where the columns are the keys of perf_dic
    df_perf = pd.DataFrame.from_dict(perf_dic, orient="index").transpose()

    return df_perf


def get_randomforest_performance_metrics_singlemodel(settings, df, host_zspe_list):
    """Compute performance metrics (accuracy, AUC, purity etc) for
    a randomforest model

    Args:
        settings (ExperimentSettings): custom class to hold hyperparameters
        df (pandas.DataFrame): dataframe containing a model's predictions
        host_zspe_list (list): available host galaxy spectroscopic redshifts

    Returns:
        (pandas.DataFrame) holds the performance metrics for this dataframe
    """

    # Compute metrics
    zspe_df = pu.reformat_df(df, "all")
    accuracy, auc, purity, efficiency, _ = pu.performance_metrics(zspe_df)
    contamination_df = pu.contamination_by_SNTYPE(zspe_df, settings)

    # Reweighted for SNe with zspe
    zspe_df = zspe_df[zspe_df["SNID"].isin(host_zspe_list)]
    if len(zspe_df) > 0:
        accuracy_zspe, auc_zspe, purity_zspe, efficiency_zspe, _ = pu.performance_metrics(
            zspe_df
        )
    else:
        accuracy_zspe, auc_zspe, purity_zspe, efficiency_zspe = (0.0, 0.0, 0.0, 0.0)

    list_columns = [
        "all_accuracy",
        "all_auc",
        "all_purity",
        "all_efficiency",
        "all_zspe_accuracy",
        "all_zspe_auc",
        "all_zspe_purity",
        "all_zspe_efficiency",
    ]

    data = np.array(
        [
            accuracy,
            auc,
            purity,
            efficiency,
            accuracy_zspe,
            auc_zspe,
            purity_zspe,
            efficiency_zspe,
        ]
    ).reshape(1, -1)

    df_perf = pd.DataFrame(data, columns=list_columns)

    for sntype, contamination_percentage in contamination_df.values:
        df_perf[f"all_contamination_{int(sntype)}"] = contamination_percentage

    return df_perf


def get_uncertainty_metrics_singlemodel(df):
    """For any lightcurve, compute the standard deviation of the model's
    predictions (this is only valid for bayesian models which yield
    a distribution of predictions).

    Then, compute the mean and std dev of this distribution across all lightcurves
    A higher mean indicates a model which is less confident in its predictions

    Args:
        df (pandas.DataFrame): dataframe containing a model's predictions

    Returns:
        (pandas.DataFrame) holds the uncertainty metrics for this dataframe
    """

    columns = ["SNID", "all_class0"] + [f"all_{OOD}_class0" for OOD in du.OOD_TYPES]

    g = df[columns].groupby("SNID").std()

    mean_std_dev = g.mean()
    std_std_dev = g.std()

    df_mean = (
        pd.DataFrame(
            data=mean_std_dev.values.reshape(1, -1), columns=mean_std_dev.index.values
        )
        .add_prefix("mean_")
        .add_suffix("_std_dev")
    )

    df_std = (
        pd.DataFrame(
            data=std_std_dev.values.reshape(1, -1), columns=std_std_dev.index.values
        )
        .add_prefix("std_")
        .add_suffix("_std_dev")
    )

    df_uncertainty = pd.concat([df_mean, df_std], axis=1)

    return df_uncertainty


def get_entropy_metrics_singlemodel(df, nb_classes):
    """Compute the entropy of the predictions
    Low entropy indicates a model that is very confident of its predictions

    Args:
        df (pandas.DataFrame): dataframe containing a model's predictions
        nb_classes (int): the number of classes in the classification task

    Returns:
        (pandas.DataFrame) holds the entropy metrics for this dataframe
    """

    list_prefixes = ["all"] + [f"all_{OOD}" for OOD in du.OOD_TYPES]
    list_data = []

    for prefix in list_prefixes:
        list_columns = [f"{prefix}_class{i}" for i in range(nb_classes)]

        arr_proba = df[list_columns].values
        entropy = -(np.log(arr_proba) * arr_proba).sum(axis=-1).mean()

        data = np.array([entropy]).reshape(1, -1)
        list_data.append(data)

    data = np.concatenate(list_data, axis=-1)
    df_entropy = pd.DataFrame(
        data, columns=[f"{prefix}_entropy" for prefix in list_prefixes]
    )

    return df_entropy


def get_calibration_metrics_singlemodel(df):
    """Compute probability calibration dataframe.
    If the calibration curve is close to identity, the model is considered
    well-calibrated.

    Args:
        df (pandas.DataFrame): dataframe containing a model's predictions

    Returns:
        (pandas.DataFrame) holds the calibration metrics for this dataframe
    """

    # TODO: clarify
    bins = np.arange(0, 11) / 10
    df["calibration_TPF"] = df["target"] != (df["all_class0"] < 0.5)
    df["prob_bin"] = pd.cut(df.all_class0.values, bins=bins, labels=range(10))
    df_calib = df[["calibration_TPF", "prob_bin"]].groupby("prob_bin").mean()

    df_calib.loc[df_calib.index >= 5] = 1 - df_calib.loc[df_calib.index >= 5]

    df_calib = df_calib.reset_index()

    # Add mean bins
    df_calib["calibration_mean_bins"] = (
        df[["all_class0", "prob_bin"]].groupby("prob_bin").mean()["all_class0"]
    )

    # Add dispersion
    dispersion = (
        (df_calib["calibration_mean_bins"] - df_calib["calibration_TPF"]) ** 2
    ).mean()

    df_calib_flat = pd.DataFrame([dispersion], columns=["calibration_dispersion"])
    for col in ["calibration_mean_bins", "calibration_TPF"]:
        df_calib_flat[col] = [df_calib[col].values]

    return df_calib_flat


def get_classification_stats_singlemodel(df, nb_classes):
    """Find out how many lightcurves are classified in each class

    Args:
        df (pandas.DataFrame): dataframe containing a model's predictions
        nb_classes (int): the number of classes in the classification task

    Returns:
        (pandas.DataFrame) holds the calibration metrics for this dataframe
    """

    list_prefixes = ["all"] + [f"all_{OOD}" for OOD in du.OOD_TYPES]
    list_df = []

    for prefix in list_prefixes:
        list_columns = [f"{prefix}_class{i}" for i in range(nb_classes)]

        arr_preds = df[list_columns].values
        pred_class = np.argmax(arr_preds, axis=1)
        list_clf_stats = [len(np.where(pred_class == i)[0]) for i in range(nb_classes)]
        # percentage of non-classified lcs
        threshold = {2: 0.6, 3: 0.4, 7: 0.2}  # choosing half of the score
        idx = np.where(np.max(arr_preds, axis=1) < threshold[nb_classes])[0]
        percentage = len(idx) * 100. / len(arr_preds)
        list_clf_stats.append(percentage)

        data = np.array(list_clf_stats).reshape(1, -1)
        columns = [f"{prefix}_num_pred_class{i}" for i in range(nb_classes)]
        columns += [f"{prefix}_percentage_non_pred"]

        list_df.append(pd.DataFrame(data, columns=columns))

    df_stats = pd.concat(list_df, axis=1)

    return df_stats
