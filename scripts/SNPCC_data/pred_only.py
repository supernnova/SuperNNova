import h5py
import json
import yaml
import math
import shutil
import argparse
import importlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from supernnova.utils import training_utils as tu
from supernnova.utils import logging_utils as lu
from supernnova.utils import data_utils as du
from supernnova.validation import metrics
from supernnova.data.dataset import HDF5Dataset

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import plots

from constants import (
    SNTYPES,
    OOD_TYPES,
    LIST_FILTERS,
    OFFSETS,
    OFFSETS_STR,
    FILTER_DICT,
    INVERSE_FILTER_DICT,
    LIST_FILTERS_COMBINATIONS,
)


def find_idx(array, value):

    idx = np.searchsorted(array, value, side="left")

    return min(idx, len(array))


def get_mse_loss(pred, target, mask):

    return ((pred - target).pow(2) * mask).sum() / mask.sum()


def get_cross_entropy_loss(pred, target):

    return torch.nn.functional.cross_entropy(pred, target, reduction="none").mean(0)


def get_accuracy_loss(pred, target):

    return (target == pred.argmax(1)).sum().float() / pred.shape[0]


def forward_pass(model, data, num_batches, return_preds=False):

    X_flux = data["X_flux"]
    X_fluxerr = data["X_fluxerr"]
    X_flt = data["X_flt"]
    X_time = data["X_time"]
    X_mask = data["X_mask"]
    X_meta = data.get("X_meta", None)

    X_target_class = data["X_target_class"]

    outs = model(X_flux, X_fluxerr, X_flt, X_time, X_mask, x_meta=X_meta)

    X_pred_class = outs.get("X_pred_class", None)

    if return_preds:
        return X_pred_class, X_target_class

    d_losses = {}

    # classification loss
    d_losses["clf_loss"] = get_cross_entropy_loss(X_pred_class, X_target_class)
    # Accuracy metric
    d_losses["accuracy"] = get_accuracy_loss(X_pred_class, X_target_class)

    # Optional KL loss
    if hasattr(model, "kl"):
        batch_size = X_target_class.shape[0]
        kl = model.kl / (batch_size * num_batches)
        d_losses["kl"] = kl

    return d_losses


def eval_pass(model, data_iterator, n_batches):

    d_losses = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for data in data_iterator:
            losses = forward_pass(model, data, n_batches)

            for key, val in losses.items():

                d_losses[key].append(val.item())

    return d_losses


def load_model(config, device, weights_file=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = importlib.import_module(f"supernnova.modules.{config['module']}").Model
    model = Model(**config["model"]).to(device)
    if weights_file is not None:
        model.load_state_dict(
            torch.load(weights_file, map_location=lambda storage, loc: storage)
        )

    return model


def load_dataset(dataset_path, config, SNID_train=None, SNID_val=None, SNID_test=None):

    dataset = HDF5Dataset(
        f"{dataset_path}/database.h5",
        config["metadata_features"],
        SNTYPES,
        config["nb_classes"],
        data_fraction=config.get("data_fraction", 1.0),
        SNID_train=SNID_train,
        SNID_val=SNID_val,
        SNID_test=SNID_test,
        load_all=True,
    )

    return dataset


def get_predictions(model_dir, pred_dir, dataset_path):

    config = yaml.load(open(Path(model_dir) / "cf.yml", "r"),
                       Loader=yaml.FullLoader)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(config, device, weights_file=Path(model_dir) / "net.pt")

    dataset = load_dataset(dataset_path, config)

    Path(pred_dir).mkdir(exist_ok=True, parents=True)

    prediction_file = f"{pred_dir}/PRED.pickle"
    nb_classes = config["nb_classes"]
    nb_inference_samples = config["nb_inference_samples"]

    n_test_batches = dataset.get_length("all", config["batch_size"])
    data_iterator = dataset.create_iterator(
        "all", config["batch_size"], device, tqdm_desc=None
    )
    num_elem = len(dataset.splits["all"])

    torch.set_grad_enabled(False)
    model.eval()

    # Prepare output arrays
    d_pred = {
        key: np.zeros((num_elem, nb_inference_samples, nb_classes)).astype(np.float32)
        for key in ["all"] + [f"PEAKMJD{s}" for s in OFFSETS_STR]
    }
    for key in ["target", "SNID"]:
        d_pred[key] = np.zeros(
            (num_elem, nb_inference_samples)).astype(np.int64)

    # Fetch SN info
    df_SNinfo = du.load_HDF5_SNinfo(dataset_path).set_index("SNID")

    start_idx = 0

    # Loop over data and make prediction
    for data in data_iterator:

        SNIDs = data["X_SNID"]
        delta_times = data["X_time"].detach().cpu().numpy()

        peak_MJDs = df_SNinfo.loc[SNIDs]["PEAKMJDNORM"].values
        times = [np.cumsum(t) for t in delta_times]
        batch_size = len(times)

        end_idx = start_idx + len(SNIDs)

        #############################
        # Full lightcurve prediction
        #############################
        for iter_ in range(nb_inference_samples):

            X_pred, X_target = forward_pass(
                model, data, n_test_batches, return_preds=True
            )
            arr_preds, arr_target = X_pred.cpu().numpy(), X_target.cpu().numpy()

            d_pred["all"][start_idx:end_idx, iter_] = arr_preds
            d_pred["target"][start_idx:end_idx, iter_] = arr_target
            d_pred["SNID"][start_idx:end_idx, iter_] = SNIDs

        #############################
        # Predictions around PEAKMJD
        #############################
        for offset in OFFSETS:
            lengths = [
                find_idx(times[k], peak_MJDs[k] + offset) for k in range(batch_size)
            ]
            # Split in 2 arrays:
            # oob_idxs: the slice for early prediction is empty for those indices
            # inb_idxs: the slice is not empty
            oob_idxs = np.where(np.array(lengths) < 1)[0]
            inb_idxs = np.where(np.array(lengths) >= 1)[0]

            if len(inb_idxs) > 0:
                data_tmp = deepcopy(data)
                max_length = max(lengths)

                for key in ["X_flux", "X_fluxerr", "X_time", "X_flt", "X_mask"]:
                    data_tmp[key] = data_tmp[key][:, :max_length]

                for idx in range(batch_size):
                    length = lengths[idx]
                    # To avoid errors when length is 0, we clamp it at 1
                    # This works because later on, we fill such samples with nans
                    length = max(1, length)
                    for key in ["X_flux", "X_fluxerr", "X_time", "X_flt", "X_mask"]:
                        if key == "X_mask":
                            data_tmp[key][idx, length:] = False
                        else:
                            data_tmp[key][idx, length:] = 0

                # for iter_ in range(nb_inference_samples):

                    # import ipdb; ipdb.set_trace()
                    # X_pred, X_target = forward_pass(
                    #     model, data_tmp, n_test_batches, return_preds=True
                    # )
                    # arr_preds, arr_target = X_pred.cpu().numpy(), X_target.cpu().numpy()

                    # suffix = str(offset) if offset != 0 else ""
                    # suffix = f"+{suffix}" if offset > 0 else suffix
                    # col = f"PEAKMJD{suffix}"

                    # d_pred[col][start_idx + inb_idxs, iter_] = arr_preds[inb_idxs]
                    # # For oob_idxs, no prediction can be made, fill with nan
                    # d_pred[col][start_idx + oob_idxs, iter_] = np.nan

        # start_idx = end_idx

    # Flatten all arrays and aggregate in dataframe
    d_series = {}
    for (key, value) in d_pred.items():
        value = value.reshape((num_elem * nb_inference_samples, -1))
        value_dim = value.shape[1]
        if value_dim == 1:
            d_series[key] = np.ravel(value)
        else:
            for i in range(value_dim):
                d_series[f"{key}_class{i}"] = value[:, i]
    df_pred = pd.DataFrame.from_dict(d_series)

    # Saving aggregated preds in case multiple predictions were sampled
    df_median = df_pred.groupby("SNID").median()
    df_median.columns = [str(col) + "_median" for col in df_median.columns]
    df_std = df_pred.groupby("SNID").std()
    df_std.columns = [str(col) + "_std" for col in df_std.columns]
    df_median = df_median.merge(df_std, on="SNID", how="left")

    df_pred = df_pred.merge(df_median, on="SNID", how="left")
    # Save predictions
    df_pred.to_pickle(prediction_file)

    g_pred = df_pred.groupby("SNID").median()
    preds = g_pred[[f"all_class{i}" for i in range(nb_classes)]].values
    preds = np.argmax(preds, 1)
    acc = (preds == g_pred.target.values).sum() / len(g_pred)

    # Display accuracy
    lu.print_green(f"Accuracy ({nb_inference_samples} inference samples)", acc)
    for col in [f"PEAKMJD{s}" for s in OFFSETS_STR]:

        preds_target = g_pred[
            [f"{col}_class{i}" for i in range(nb_classes)] + ["target"]
        ].dropna()
        preds = preds_target[[f"{col}_class{i}" for i in range(nb_classes)]].values
        target = preds_target["target"].values
        preds = np.argmax(preds, 1)
        acc = (preds == target).sum() / len(g_pred)

        lu.print_green(f"{col} Accuracy", acc)

    class_col = [f"all_class{i}" for i in range(nb_classes)]
    tmp = df_pred[["SNID", "target"] + class_col].groupby("SNID").mean()
    preds = np.argmax(tmp[class_col].values, 1)
    acc = (preds == tmp.target.values).sum() / len(tmp)
    lu.print_green(f"Accuracy (mean prediction)", acc)

    lu.print_green("Finished getting predictions ")

    torch.set_grad_enabled(True)


def get_metrics(model_dir, pred_dir, dataset_path):
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

    config = yaml.load(open(Path(model_dir) / "cf.yml", "r"),
                       Loader=yaml.FullLoader)

    nb_classes = config["nb_classes"]
    processed_dir = dataset_path  # config["processed_dir"]
    prediction_file = (Path(pred_dir) / f"PRED.pickle").as_posix()
    metrics_file = (Path(pred_dir) / f"METRICS.pickle").as_posix()

    df_SNinfo = du.load_HDF5_SNinfo(processed_dir)
    host = pd.read_pickle(f"{processed_dir}/hostspe_SNID.pickle")
    host_zspe_list = host["SNID"].tolist()

    df = pd.read_pickle(prediction_file)
    df = pd.merge(df, df_SNinfo[["SNID", "SNTYPE"]], on="SNID", how="left")

    list_df_metrics = []

    list_df_metrics.append(metrics.get_calibration_metrics_singlemodel(df))
    list_df_metrics.append(
        metrics.get_rnn_performance_metrics_singlemodel(
            config, df, SNTYPES, host_zspe_list
        )
    )
    if OOD_TYPES:
        list_df_metrics.append(
            metrics.get_uncertainty_metrics_singlemodel(df, OOD_TYPES)
        )
        list_df_metrics.append(
            metrics.get_entropy_metrics_singlemodel(df, OOD_TYPES, nb_classes)
        )
        list_df_metrics.append(
            metrics.get_classification_stats_singlemodel(
                df, OOD_TYPES, nb_classes)
        )

    df_metrics = pd.concat(list_df_metrics, 1)

    df_metrics["model_name"] = Path(model_dir).name
    # TODO
    df_metrics["source_data"] = "saltfit"
    df_metrics.to_pickle(metrics_file)


def get_plots(model_dir, pred_dir, dataset_path):

    config = yaml.load(open(Path(model_dir) / "cf.yml", "r"),
                       Loader=yaml.FullLoader)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(config, device, weights_file=Path(model_dir) / "net.pt")

    dataset = load_dataset(dataset_path, config)

    data_iterator = dataset.create_iterator("all", 1, device, tqdm_desc=None)

    config["dump_dir"] = pred_dir

    plots.make_early_prediction(dataset_path,model, config, data_iterator, LIST_FILTERS, INVERSE_FILTER_DICT, device, SNTYPES
                                )


def main(config_path, pred_dir, dataset_path):

    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    # setting random seeds
    np.random.seed(config["seed"])

    # Get predictions
    get_predictions(config["dump_dir"], pred_dir, dataset_path)
    lu.print_blue("Finished test set predictions")

    # Compute metrics
    get_metrics(config["dump_dir"], pred_dir, dataset_path)
    lu.print_blue("Finished metrics")

    # Plot some lightcurves
    get_plots(config["dump_dir"], pred_dir, dataset_path)
    lu.print_blue("Finished plotting lightcurves")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dataset building")
    parser.add_argument("config_path", help="Path to yml config file")
    parser.add_argument("pred_dir", help="Path where to save predictions")
    parser.add_argument(
        "dataset_path", help="Path to where data is stored in HDF5")

    args = parser.parse_args()

    main(args.config_path, args.pred_dir, args.dataset_path)
