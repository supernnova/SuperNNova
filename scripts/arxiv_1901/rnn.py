import h5py
import json
import yaml
import shutil
import argparse
import importlib
import numpy as np
from tqdm import tqdm
from time import time
from pathlib import Path
from collections import defaultdict

from supernnova.utils import training_utils as tu
from supernnova.utils import logging_utils as lu

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from constants import (
    SNTYPES,
    LIST_FILTERS,
    OFFSETS,
    OFFSETS_STR,
    FILTER_DICT,
    INVERSE_FILTER_DICT,
    LIST_FILTERS_COMBINATIONS,
)


def forward_pass(model, data):

    X_flux = data["X_flux"]  # .detach().cpu().numpy()
    X_fluxerr = data["X_fluxerr"]  # .detach().cpu().numpy()
    X_flt = data["X_flt"]  # .detach().cpu().numpy()
    X_time = data["X_time"]  # .detach().cpu().numpy()
    X_mask = data["X_mask"]  # .detach().cpu().numpy()
    X_meta = data.get("X_meta", None)

    X_target = data["X_target"]  # .detach().cpu().numpy()

    # import matplotlib.pylab as plt
    # import matplotlib.gridspec as gridspec
    # from matplotlib.pyplot import cm

    # for i in range(X_flux.shape[0]):
    #     fig = plt.figure(figsize=(10, 10))
    #     gs = gridspec.GridSpec(1, 1)
    #     ax = plt.subplot(gs[0])
    #     time = X_time[i].cumsum()
    #     length = X_mask[i].astype(int).sum()
    #     for j, c in enumerate(LIST_FILTERS):
    #         flux = [
    #             X_flux[i, t, j]
    #             for t in range(length)
    #             if c in INVERSE_FILTER_DICT[X_flt[i, t]]
    #         ]
    #         fluxerr = [
    #             X_fluxerr[i, t, j]
    #             for t in range(length)
    #             if c in INVERSE_FILTER_DICT[X_flt[i, t]]
    #         ]
    #         tmp = [
    #             time[t] for t in range(length) if c in INVERSE_FILTER_DICT[X_flt[i, t]]
    #         ]
    #         ax.errorbar(tmp, flux, yerr=fluxerr, color=f"C{j}")

    #     plt.title(f"Class {X_target[i]}")
    #     plt.savefig(f"fig_{i}.png")
    #     plt.clf()
    #     plt.close("all")

    # import ipdb

    # ipdb.set_trace()

    X_pred = model(X_flux, X_fluxerr, X_flt, X_time, X_mask, x_meta=X_meta)

    loss = torch.nn.functional.cross_entropy(X_pred, X_target, reduction="none").mean(0)

    return loss, X_pred, X_target


def get_predictions(model, list_data, list_batches, device):

    list_target = []
    list_pred = []

    model.eval()
    with torch.no_grad():
        for batch_idxs in list_batches:
            data = tu.get_data_batch(list_data, batch_idxs, device)
            _, X_pred, X_target = forward_pass(model, data)
            list_target.append(X_target)
            list_pred.append(X_pred)

    X_pred = torch.cat(list_pred, dim=0)
    X_target = torch.cat(list_target, dim=0)

    X_pred = torch.nn.functional.softmax(X_pred, dim=-1)

    return X_target, X_pred



def get_test_predictions(model, config, list_data, device):

    prediction_file = f"{config['dump_dir']}/PRED.pickle"

    model.eval()

    num_elem = len(list_batches)
    num_batches = max(1, num_elem // config["batch_size_test"])
    list_batches = np.array_split(np.arange(num_elem), num_batches)

    # Prepare output arrays
    d_pred = {
        key: np.zeros(
            (num_elem, settings.num_inference_samples, settings.nb_classes)
        ).astype(np.float32)
        for key in [
            "all",
            "PEAKMJD-2",
            "PEAKMJD-1",
            "PEAKMJD",
            "PEAKMJD+1",
            "PEAKMJD+2",
        ]
    }
    for key in ["target", "SNID"]:
        d_pred[key] = np.zeros((num_elem, settings.num_inference_samples)).astype(
            np.int64
        )

    d_pred_MFE = {
        key: np.zeros((num_elem, 1, settings.nb_classes)).astype(np.float32)
        for key in ["all"] + [f"all_{OOD}" for OOD in du.OOD_TYPES]
    }
    for key in ["target", "SNID"]:
        d_pred_MFE[key] = np.zeros((num_elem, 1)).astype(np.int64)

    # Fetch SN info
    df_SNinfo = du.load_HDF5_SNinfo(settings).set_index("SNID")

    # Loop over data and make prediction
    for batch_idxs in tqdm(
        list_batches, desc="Computing predictions on test set", ncols=100
    ):

        start_idx, end_idx = batch_idxs[0], batch_idxs[-1] + 1
        SNIDs = [data[2] for data in list_data_test[start_idx:end_idx]]

        peak_MJDs = df_SNinfo.loc[SNIDs]["PEAKMJDNORM"].values
        delta_times = [
            data[3][:, settings.d_feat_to_idx["delta_time"]]
            for data in list_data_test[start_idx:end_idx]
        ]
        times = [np.cumsum(t) for t in delta_times]

        with torch.no_grad():

            #############################
            # Full lightcurve prediction
            #############################

            packed, _, target_tensor, idxs_rev_sort = tu.get_data_batch(
                list_data_test, batch_idxs, settings
            )

            for iter_ in tqdm(range(settings.num_inference_samples), ncols=100):

                arr_preds, arr_target = get_batch_predictions(
                    rnn, packed, target_tensor
                )

                # Rever sorting that occurs in get_batch_predictions
                arr_preds = arr_preds[idxs_rev_sort]
                arr_target = arr_target[idxs_rev_sort]

                d_pred["all"][start_idx:end_idx, iter_] = arr_preds
                d_pred["target"][start_idx:end_idx, iter_] = arr_target
                d_pred["SNID"][start_idx:end_idx, iter_] = SNIDs

            # MFE
            arr_preds, arr_target = get_batch_predictions_MFE(
                rnn, packed, target_tensor
            )

            # Rever sorting that occurs in get_batch_predictions
            arr_preds = arr_preds[idxs_rev_sort]
            arr_target = arr_target[idxs_rev_sort]

            d_pred_MFE["all"][start_idx:end_idx, 0] = arr_preds
            d_pred_MFE["target"][start_idx:end_idx, 0] = arr_target
            d_pred_MFE["SNID"][start_idx:end_idx, 0] = SNIDs

            #############################
            # Predictions around PEAKMJD
            #############################
            for offset in [-2, -1, 0, 1, 2]:
                slice_idxs = [
                    find_idx(times[k], peak_MJDs[k] + offset) for k in range(len(times))
                ]
                # Split in 2 arrays:
                # oob_idxs: the slice for early prediction is empty for those indices
                # inb_idxs: the slice is not empty
                oob_idxs = np.where(np.array(slice_idxs) < 1)[0]
                inb_idxs = np.where(np.array(slice_idxs) >= 1)[0]

                if len(inb_idxs) > 0:
                    # We only carry out prediction for samples in ``inb_idxs``
                    offset_batch_idxs = [batch_idxs[b] for b in inb_idxs]
                    max_lengths = [slice_idxs[b] for b in inb_idxs]
                    packed, _, target_tensor, idxs_rev_sort = tu.get_data_batch(
                        list_data_test,
                        offset_batch_idxs,
                        settings,
                        max_lengths=max_lengths,
                    )

                    for iter_ in tqdm(range(settings.num_inference_samples), ncols=100):

                        arr_preds, arr_target = get_batch_predictions(
                            rnn, packed, target_tensor
                        )

                        # Rever sorting that occurs in get_batch_predictions
                        arr_preds = arr_preds[idxs_rev_sort]

                        suffix = str(offset) if offset != 0 else ""
                        suffix = f"+{suffix}" if offset > 0 else suffix
                        col = f"PEAKMJD{suffix}"

                        d_pred[col][start_idx + inb_idxs, iter_] = arr_preds
                        # For oob_idxs, no prediction can be made, fill with nan
                        d_pred[col][start_idx + oob_idxs, iter_] = np.nan

            #############################
            # OOD predictions
            #############################

            for OOD in ["random", "shuffle", "reverse", "sin"]:
                packed, _, target_tensor, idxs_rev_sort = tu.get_data_batch(
                    list_data_test, batch_idxs, settings, OOD=OOD
                )

                for iter_ in tqdm(range(settings.num_inference_samples), ncols=100):

                    arr_preds, arr_target = get_batch_predictions(
                        rnn, packed, target_tensor
                    )

                    # Revert sorting that occurs in get_batch_predictions
                    arr_preds = arr_preds[idxs_rev_sort]
                    arr_target = arr_target[idxs_rev_sort]

                    d_pred[f"all_{OOD}"][start_idx:end_idx, iter_] = arr_preds

                arr_preds, arr_target = get_batch_predictions_MFE(
                    rnn, packed, target_tensor
                )

                # Revert sorting that occurs in get_batch_predictions
                arr_preds = arr_preds[idxs_rev_sort]
                arr_target = arr_target[idxs_rev_sort]

                d_pred_MFE[f"all_{OOD}"][start_idx:end_idx, 0] = arr_preds

    # Flatten all arrays and aggregate in dataframe
    d_series = {}
    for (key, value) in d_pred.items():
        value = value.reshape((num_elem * settings.num_inference_samples, -1))
        value_dim = value.shape[1]
        if value_dim == 1:
            d_series[key] = np.ravel(value)
        else:
            for i in range(value_dim):
                d_series[f"{key}_class{i}"] = value[:, i]
    df_pred = pd.DataFrame.from_dict(d_series)

    # Flatten all arrays and aggregate in dataframe
    d_series_MFE = {}
    for (key, value) in d_pred_MFE.items():
        value = value.reshape((num_elem * 1, -1))
        value_dim = value.shape[1]
        if value_dim == 1:
            d_series_MFE[key] = np.ravel(value)
        else:
            for i in range(value_dim):
                d_series_MFE[f"{key}_class{i}"] = value[:, i]
    df_pred_MFE = pd.DataFrame.from_dict(d_series_MFE)

    # Save predictions
    df_pred.to_pickle(prediction_file)

    # Saving aggregated preds for bayesian models
    if settings.model == "variational" or settings.model == "bayesian":
        med_pred = df_pred.groupby("SNID").median()
        med_pred.columns = [str(col) + "_median" for col in med_pred.columns]
        std_pred = df_pred.groupby("SNID").std()
        std_pred.columns = [str(col) + "_std" for col in std_pred.columns]
        df_bayes = pd.merge(med_pred, std_pred, on="SNID")
        df_bayes["SNID"] = df_bayes.index
        df_bayes["target"] = df_bayes["target_median"]
        bay_pred_file = prediction_file.replace(".pickle", "_aggregated.pickle")
        df_bayes.to_pickle(bay_pred_file)

    g_pred = df_pred.groupby("SNID").median()
    preds = g_pred[[f"all_class{i}" for i in range(settings.nb_classes)]].values
    preds = np.argmax(preds, 1)
    acc = (preds == g_pred.target.values).sum() / len(g_pred)

    # Display accuracy
    lu.print_green("Full Accuracy", acc)
    for col in [f"PEAKMJD{s}" for s in du.OFFSETS_STR]:

        preds_target = g_pred[
            [f"{col}_class{i}" for i in range(settings.nb_classes)] + ["target"]
        ].dropna()
        preds = preds_target[
            [f"{col}_class{i}" for i in range(settings.nb_classes)]
        ].values
        target = preds_target["target"].values
        preds = np.argmax(preds, 1)
        acc = (preds == target).sum() / len(g_pred)

        lu.print_green(f"{col} Accuracy", acc)

    print()
    print()

    class_col = [f"all_class{i}" for i in range(settings.nb_classes)]
    tmp = df_pred[["SNID", "target"] + class_col].groupby("SNID").mean()
    preds = np.argmax(tmp[class_col].values, 1)
    acc = (preds == tmp.target.values).sum() / len(tmp)
    lu.print_green(f"Accuracy MC", acc)

    for OOD in ["random", "reverse", "shuffle", "sin"]:
        class_col_ood = [f"all_{OOD}_class{i}" for i in range(settings.nb_classes)]
        entropy_ood = (
            -(df_pred[class_col_ood].values * np.log(df_pred[class_col_ood].values))
            .sum(1)
            .mean()
        )
        entropy = (
            -(df_pred[class_col].values * np.log(df_pred[class_col].values))
            .sum(1)
            .mean()
        )
        lu.print_green(f"Delta Entropy {OOD} MC", entropy_ood - entropy)

    print()
    print()

    tmp = df_pred_MFE[["SNID", "target"] + class_col].groupby("SNID").mean()
    preds = np.argmax(tmp[class_col].values, 1)
    acc = (preds == tmp.target.values).sum() / len(tmp)
    lu.print_green(f"Accuracy MFE", acc)

    for OOD in ["random", "reverse", "shuffle", "sin"]:
        class_col_ood = [f"all_{OOD}_class{i}" for i in range(settings.nb_classes)]
        entropy_ood = (
            -(
                df_pred_MFE[class_col_ood].values
                * np.log(df_pred_MFE[class_col_ood].values)
            )
            .sum(1)
            .mean()
        )
        entropy = (
            -(df_pred_MFE[class_col].values * np.log(df_pred_MFE[class_col].values))
            .sum(1)
            .mean()
        )
        lu.print_green(f"Delta Entropy {OOD} MFE", entropy_ood - entropy)

    lu.print_green("Finished getting predictions ")

    return prediction_file

def train(config):
    """Train RNN models with a decay on plateau policy

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
    """

    shutil.rmtree(Path(config["dump_dir"]), ignore_errors=True)
    Path(config["dump_dir"]).mkdir(parents=True)

    # Data
    list_data_train, list_data_val, list_data_test = tu.load_HDF5(
        config, SNTYPES
    )

    num_elem = len(list_data_train)
    num_batches = max(1, num_elem // config["batch_size"])
    list_batches_train = np.array_split(np.arange(num_elem), num_batches)

    num_elem = len(list_data_val)
    num_batches = max(1, num_elem // config["batch_size"])
    list_batches_val = np.array_split(np.arange(num_elem), num_batches)

    # Model specification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config["model"]["num_embeddings"] = len(LIST_FILTERS_COMBINATIONS)
    Model = importlib.import_module(f"supernnova.modules.{config['module']}").Model
    model = Model(**config["model"]).to(device)

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

    loss_str = ""
    d_monitor_train = defaultdict(list)
    d_monitor_val = defaultdict(list)
    log_dir = Path(config["dump_dir"]) / "tensorboard"
    log_dir.mkdir()
    writer = SummaryWriter(log_dir=log_dir.as_posix())

    # TODO KL

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        factor=config["lr_factor"],
        min_lr=config["min_lr"],
        patience=config["patience"],
        verbose=True,
    )

    batch = 0
    best_loss = float("inf")

    for epoch in range(config["nb_epoch"]):

        desc = f"Epoch: {epoch} -- {loss_str}"

        np.random.shuffle(list_batches_train)

        for batch_idxs in tqdm(
            list_batches_train,
            desc=desc,
            ncols=100,
            bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}",
        ):

            # Sample a batch
            data = tu.get_data_batch(list_data_train, batch_idxs, device)

            model.train()
            optimizer.zero_grad()

            # Train step : forward backward pass
            loss, *_ = forward_pass(model, data)

            loss.backward()
            optimizer.step()

            batch += 1

        # Get metrics (subsample training set to same size as validation set for speed)
        X_target_train, X_pred_train = get_predictions(
            model, list_data_train, list_batches_train[: len(list_batches_val)], device
        )
        X_target_val, X_pred_val = get_predictions(
            model, list_data_val, list_batches_val, device
        )

        d_losses_train = tu.get_evaluation_metrics(
            X_pred_train, X_target_train, nb_classes=config["nb_classes"]
        )
        d_losses_val = tu.get_evaluation_metrics(
            X_pred_val, X_target_val, nb_classes=config["nb_classes"]
        )

        # Add current loss avg to list of losses
        for key in d_losses_train.keys():
            d_monitor_train[key].append(d_losses_train[key])
            d_monitor_val[key].append(d_losses_val[key])

        d_monitor_train["epoch"].append(epoch + 1)
        d_monitor_val["epoch"].append(epoch + 1)

        for metric in d_losses_train:
            writer.add_scalars(
                f"Metrics/{metric.title()}",
                {"training": d_losses_train[metric], "valid": d_losses_val[metric]},
                batch,
            )

        # Prepare loss_str to update progress bar
        loss_str = tu.get_loss_string(d_losses_train, d_losses_val)

        save_prefix = f"{config['dump_dir']}/loss"
        tu.plot_loss(d_monitor_train, d_monitor_val, save_prefix)
        if d_monitor_val["log_loss"][-1] < best_loss:
            best_loss = d_monitor_val["log_loss"][-1]
            torch.save(model.state_dict(), f"{config['dump_dir']}/net.pt")

        # LR scheduling
        scheduler.step(d_losses_val["log_loss"])
        lr_value = next(iter(optimizer.param_groups))["lr"]
        if lr_value <= config["min_lr"]:
            print("Minimum LR reached, ending training")
            break

    lu.print_green("Finished training")

    # Start validating on test set


def main(config_path):

    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    # setting random seeds
    np.random.seed(config["seed"])

    # Train and sav predictions
    train(config)
    # Compute metrics
    get_metrics(config)

    logging_utils.print_blue("Finished rnn training, validating and testing")

    # # Obtain predictions
    # validate_rnn.get_predictions(settings)
    # # Compute metrics
    # metrics.get_metrics_singlemodel(settings, model_type="rnn")
    # # Plot some lightcurves
    # early_prediction.make_early_prediction(settings)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dataset building")
    parser.add_argument("config_path", help="Path to yml config gile")

    args = parser.parse_args()

    main(args.config_path)
