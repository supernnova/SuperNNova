import h5py
import yaml
import shutil
import argparse
import importlib
import numpy as np
import pandas as pd
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
    X_target_peak = data["X_target_peak"].squeeze(-1)
    X_target_peak_single = data["X_target_peak_single"].squeeze(-1)

    outs = model(X_flux, X_fluxerr, X_flt, X_time, X_mask, x_meta=X_meta)

    X_pred_class = outs.get("X_pred_class", None)
    X_pred_peak = outs.get("X_pred_peak", None)

    if return_preds:
        # TODO change
        # return X_pred_class, X_target_class, X_pred_peak, X_target_peak
        return X_pred_class, X_target_class, X_pred_peak, X_target_peak_single

    d_losses = {}

    # peak prediction loss
    # TODO change
    # d_losses["peak_loss"] = get_mse_loss(X_pred_peak, X_target_peak, X_mask)
    d_losses["peak_loss"] = get_mse_loss(X_pred_peak, X_target_peak_single, X_mask)
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


def load_dataset(config, SNID_train=None, SNID_val=None, SNID_test=None):

    dataset = HDF5Dataset(
        f"{config['processed_dir']}/database.h5",
        config["metadata_features"],
        SNTYPES,
        config["nb_classes"],
        data_fraction=config.get("data_fraction", 1.0),
        SNID_train=SNID_train,
        SNID_val=SNID_val,
        SNID_test=SNID_test,
    )

    return dataset


def train(config):
    """Train RNN models with a decay on plateau policy

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
    """

    config["model"]["num_embeddings"] = len(LIST_FILTERS_COMBINATIONS)
    shutil.rmtree(Path(config["dump_dir"]), ignore_errors=True)
    Path(config["dump_dir"]).mkdir(parents=True)

    # Data
    dataset = load_dataset(config)

    # Model specification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(config, device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=1e-6
    )

    if model.normalize:
        # Load normalizations
        processed_dir = config["processed_dir"]
        file_name = f"{processed_dir}/database.h5"
        with h5py.File(file_name, "r") as hf:
            flux_norm = np.array(hf["data"].attrs["flux_norm"]).astype(np.float32)
            fluxerr_norm = np.array(hf["data"].attrs["fluxerr_norm"]).astype(np.float32)
            delta_time_norm = np.array(hf["data"].attrs["delta_time_norm"]).astype(
                np.float32
            )

            flux_norm = torch.from_numpy(flux_norm).to(device)
            fluxerr_norm = torch.from_numpy(fluxerr_norm).to(device)
            delta_time_norm = torch.from_numpy(delta_time_norm).to(device)

            model.flux_norm.data = flux_norm
            model.fluxerr_norm.data = fluxerr_norm
            model.delta_time_norm.data = delta_time_norm

    print(model)

    loss_str = ""
    d_monitor_train = defaultdict(list)
    d_monitor_val = defaultdict(list)
    log_dir = Path(config["dump_dir"]) / "tensorboard"
    log_dir.mkdir()
    writer = SummaryWriter(log_dir=log_dir.as_posix())

    # Save config
    with open(Path(config["dump_dir"]) / "cf.yml", "w") as f:
        yaml.dump(config, f)

    # Save the dataset splits splits
    df_train = pd.DataFrame(dataset.SNID_train.reshape(-1, 1), columns=["SNID"])
    df_val = pd.DataFrame(dataset.SNID_val.reshape(-1, 1), columns=["SNID"])
    df_test = pd.DataFrame(dataset.SNID_test.reshape(-1, 1), columns=["SNID"])

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    df_splits = pd.concat([df_train, df_val, df_test], 0)
    save_file = (Path(config["dump_dir"]) / f"data_splits.csv").as_posix()
    df_splits.to_csv(save_file, index=False)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        factor=config["lr_factor"],
        min_lr=config["min_lr"],
        patience=config["patience"],
        verbose=True,
    )

    n_train_batches = dataset.get_length("train", config["batch_size"])
    n_val_batches = dataset.get_length("val", config["batch_size"])

    batch = 0
    best_loss = float("inf")

    for epoch in range(config["nb_epoch"]):

        desc = f"Epoch: {epoch} -- {loss_str}"

        d_losses_train = defaultdict(list)

        for data in dataset.create_iterator(
            "train",
            config["batch_size"],
            device,
            tqdm_desc=desc,
            random_length=config.get("random_length", False),
        ):

            model.train()

            # Train step : forward backward pass
            losses_train = forward_pass(model, data, n_train_batches)

            loss = (
                losses_train["clf_loss"] * config.get("clf_weight", 1.0)
                + losses_train.get("kl", 0.0)
                + losses_train["peak_loss"] * config.get("peak_weight", 1.0)
            )

            for key, val in losses_train.items():
                d_losses_train[key].append(val.item())

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            batch += 1

        val_iterator = dataset.create_iterator(
            "val", config["batch_size"], device, tqdm_desc=None
        )
        d_losses_val = eval_pass(model, val_iterator, n_val_batches)

        # Monitor losses in dict + tensorboard
        d_monitor_train["epoch"].append(epoch + 1)
        d_monitor_val["epoch"].append(epoch + 1)
        for key in d_losses_train.keys():

            d_losses_train[key] = np.mean(d_losses_train[key])
            d_losses_val[key] = np.mean(d_losses_val[key])

            d_monitor_train[key].append(d_losses_train[key])
            d_monitor_val[key].append(d_losses_val[key])

            writer.add_scalars(
                f"Metrics/{key.title()}",
                {"training": d_losses_train[key], "valid": d_losses_val[key]},
                batch,
            )

        # Prepare loss_str to update progress bar
        loss_str = tu.get_loss_string(d_losses_train, d_losses_val)

        # Plot losses
        save_prefix = f"{config['dump_dir']}/loss"
        tu.plot_loss(d_monitor_train, d_monitor_val, save_prefix)

        # Make some lightcurve plots to check predictions
        data_iterator = dataset.create_iterator("test", 1, device, tqdm_desc=None)
        figs = plots.make_early_prediction(
            model,
            config,
            data_iterator,
            LIST_FILTERS,
            INVERSE_FILTER_DICT,
            device,
            SNTYPES,
            nb_lcs=9,
            return_fig=True,
        )
        for idx, fig in enumerate(figs):
            writer.add_figure(f"Lightcurves/{idx}", fig, batch)
            plt.close(fig)
        plt.clf()
        plt.close("all")

        # Save on progress
        candidate_loss = d_losses_val["clf_loss"] + d_losses_val["peak_loss"]
        if candidate_loss < best_loss:
            best_loss = candidate_loss
            torch.save(model.state_dict(), f"{config['dump_dir']}/net.pt")

        # LR scheduling
        scheduler.step(candidate_loss)
        lr_value = next(iter(optimizer.param_groups))["lr"]
        if lr_value <= config["min_lr"]:
            print("Minimum LR reached, ending training")
            break


def get_predictions(dump_dir):

    config = yaml.load(open(Path(dump_dir) / "cf.yml", "r"), Loader=yaml.FullLoader)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(config, device, weights_file=Path(dump_dir) / "net.pt")

    # Re-use same splits as training
    df_splits = pd.read_csv(Path(dump_dir) / "data_splits.csv")
    SNID_train = df_splits[df_splits.split == "train"]["SNID"].values
    SNID_val = df_splits[df_splits.split == "val"]["SNID"].values
    SNID_test = df_splits[df_splits.split == "test"]["SNID"].values

    dataset = load_dataset(
        config, SNID_train=SNID_train, SNID_val=SNID_val, SNID_test=SNID_test
    )

    prediction_file = f"{dump_dir}/PRED.pickle"
    nb_classes = config["nb_classes"]
    nb_inference_samples = config["nb_inference_samples"]

    n_test_batches = dataset.get_length("test", config["batch_size_test"])
    data_iterator = dataset.create_iterator(
        "test", config["batch_size_test"], device, tqdm_desc="Test set predictions"
    )
    num_elem = len(dataset.splits["test"])

    torch.set_grad_enabled(False)
    model.eval()

    # Prepare output arrays
    d_pred = {
        key: np.zeros((num_elem, nb_inference_samples, nb_classes)).astype(np.float32)
        for key in ["all"] + [f"PEAKMJD{s}" for s in OFFSETS_STR]
    }
    for key in ["target", "SNID"]:
        d_pred[key] = np.zeros((num_elem, nb_inference_samples)).astype(np.int64)

    for key in ["target_peak", "all_peak"] + [f"PEAKMJD{s}_peak" for s in OFFSETS_STR]:
        d_pred[key] = np.zeros((num_elem, nb_inference_samples)).astype(np.float32)

    # Fetch SN info
    df_SNinfo = du.load_HDF5_SNinfo(config["processed_dir"]).set_index("SNID")

    start_idx = 0

    # Loop over data and make prediction
    for data in data_iterator:

        SNIDs = data["X_SNID"]
        delta_times = data["X_time"].squeeze(-1).detach().cpu().numpy()
        full_lengths = data["X_mask"].sum(1).long().detach().cpu().numpy()

        peak_MJDs = df_SNinfo.loc[SNIDs]["PEAKMJDNORM"].values
        times = [
            np.cumsum(t[:length]) for (t, length) in zip(delta_times, full_lengths)
        ]
        batch_size = len(times)

        end_idx = start_idx + len(SNIDs)

        #############################
        # Full lightcurve prediction
        #############################
        for iter_ in range(nb_inference_samples):
            X_pred_class, X_target_class, X_pred_peak, X_target_peak = forward_pass(
                model, data, n_test_batches, return_preds=True
            )

            arr_class_preds, arr_class_target = (
                X_pred_class.cpu().numpy(),
                X_target_class.cpu().numpy(),
            )
            d_pred["all"][start_idx:end_idx, iter_] = arr_class_preds
            d_pred["target"][start_idx:end_idx, iter_] = arr_class_target
            d_pred["SNID"][start_idx:end_idx, iter_] = SNIDs
            # select the last peak pred
            last_time_length = data["X_mask"].sum(1) - 1
            last_peak_preds = torch.gather(
                X_pred_peak, 1, (last_time_length).view(-1, 1)
            ).squeeze(-1)
            arr_peak_preds = last_peak_preds.cpu().numpy()

            last_peak_target = torch.gather(
                X_target_peak, 1, (last_time_length).view(-1, 1)
            ).squeeze(-1)
            arr_peak_target = last_peak_target.cpu().numpy()

            d_pred["all_peak"][start_idx:end_idx, iter_] = arr_peak_preds
            d_pred["target_peak"][start_idx:end_idx, iter_] = arr_peak_target

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

                for key in [
                    "X_flux",
                    "X_fluxerr",
                    "X_time",
                    "X_flt",
                    "X_mask",
                    "X_target_peak",
                    "X_target_peak_single",
                ]:
                    data_tmp[key] = data_tmp[key][:, :max_length]

                for idx in range(batch_size):
                    length = lengths[idx]
                    # To avoid errors when length is 0, we clamp it at 1
                    # This works because later on, we fill such samples with nans
                    length = max(1, length)
                    for key in [
                        "X_flux",
                        "X_fluxerr",
                        "X_time",
                        "X_flt",
                        "X_mask",
                        "X_target_peak",
                        "X_target_peak_single",
                    ]:
                        if key == "X_mask":
                            data_tmp[key][idx, length:] = False
                        else:
                            data_tmp[key][idx, length:] = 0

                for iter_ in range(nb_inference_samples):
                    try:
                        X_pred_class, X_target_class, X_pred_peak, X_target_peak = forward_pass(
                            model, data_tmp, n_test_batches, return_preds=True
                        )
                    except Exception:
                        import ipdb

                        ipdb.set_trace()

                    arr_class_preds, arr_class_target = (
                        X_pred_class.cpu().numpy(),
                        X_target_class.cpu().numpy(),
                    )
                    suffix = str(offset) if offset != 0 else ""
                    suffix = f"+{suffix}" if offset > 0 else suffix
                    col = f"PEAKMJD{suffix}"

                    d_pred[col][start_idx + inb_idxs, iter_] = arr_class_preds[inb_idxs]
                    # For oob_idxs, no prediction can be made, fill with nan
                    d_pred[col][start_idx + oob_idxs, iter_] = np.nan

                    # select the last peak pred
                    last_time_length = (X_target_peak != 0).sum(1) - 1
                    last_peak_preds = torch.gather(
                        X_pred_peak, 1, (last_time_length).view(-1, 1)
                    ).squeeze(-1)
                    arr_peak_preds = last_peak_preds.cpu().numpy()

                    last_peak_target = torch.gather(
                        X_target_peak, 1, (last_time_length).view(-1, 1)
                    ).squeeze(-1)
                    arr_peak_target = last_peak_target.cpu().numpy()

        start_idx = end_idx

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


def get_metrics(dump_dir):
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

    config = yaml.load(open(Path(dump_dir) / "cf.yml", "r"), Loader=yaml.FullLoader)

    nb_classes = config["nb_classes"]
    processed_dir = config["processed_dir"]
    prediction_file = (Path(dump_dir) / f"PRED.pickle").as_posix()
    metrics_file = (Path(dump_dir) / f"METRICS.pickle").as_posix()

    df_SNinfo = du.load_HDF5_SNinfo(config["processed_dir"])
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
            metrics.get_classification_stats_singlemodel(df, OOD_TYPES, nb_classes)
        )

    df_metrics = pd.concat(list_df_metrics, 1)

    df_metrics["model_name"] = Path(dump_dir).name
    # TODO
    df_metrics["source_data"] = "saltfit"
    df_metrics.to_pickle(metrics_file)


def get_plots(dump_dir):

    config = yaml.load(open(Path(dump_dir) / "cf.yml", "r"), Loader=yaml.FullLoader)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(config, device, weights_file=Path(dump_dir) / "net.pt")

    # Re-use same splits as training
    df_splits = pd.read_csv(Path(dump_dir) / "data_splits.csv")
    SNID_train = df_splits[df_splits.split == "train"]["SNID"].values
    SNID_val = df_splits[df_splits.split == "val"]["SNID"].values
    SNID_test = df_splits[df_splits.split == "test"]["SNID"].values

    dataset = load_dataset(
        config, SNID_train=SNID_train, SNID_val=SNID_val, SNID_test=SNID_test
    )

    data_iterator = dataset.create_iterator("test", 1, device, tqdm_desc=None)

    plots.make_early_prediction(
        model, config, data_iterator, LIST_FILTERS, INVERSE_FILTER_DICT, device, SNTYPES
    )


def main(config_path):

    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    # setting random seeds
    np.random.seed(config["seed"])

    # Train
    train(config)
    lu.print_blue("Finished rnn training")

    # Get predictions
    get_predictions(config["dump_dir"])
    lu.print_blue("Finished test set predictions")

    # Compute metrics
    get_metrics(config["dump_dir"])
    lu.print_blue("Finished metrics")

    # Plot some lightcurves
    get_plots(config["dump_dir"])
    lu.print_blue("Finished plotting lightcurves")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dataset building")
    parser.add_argument("config_path", help="Path to yml config gile")

    args = parser.parse_args()

    main(args.config_path)
