import h5py
import json
import yaml
import argparse
import importlib
import numpy as np
from tqdm import tqdm
from time import time
from pathlib import Path

from supernnova.utils import training_utils as tu
from supernnova.utils import logging_utils as lu

import torch
import torch.nn as nn

from constants import (
    SNTYPES,
    LIST_FILTERS,
    OFFSETS,
    OFFSETS_STR,
    FILTER_DICT,
    INVERSE_FILTER_DICT,
    LIST_FILTERS_COMBINATIONS,
)


# def get_lr(settings):
#     """Select optimal starting learning rate when training with a 1-cycle policy

#     Args:
#         settings (ExperimentSettings): controls experiment hyperparameters
#     """

#     # Data
#     list_data_train, list_data_val = tu.load_HDF5(settings, test=False)

#     num_elem = len(list_data_train)
#     num_batches = num_elem // min(num_elem // 2, settings.batch_size)
#     list_batches = np.array_split(np.arange(num_elem), num_batches)
#     np.random.shuffle(list_batches)

#     lr_init_value = 1e-8
#     lr = float(lr_init_value)
#     lr_final_value = 10.0
#     beta = 0.98
#     avg_loss = 0.0
#     best_loss = 0.0
#     batch_num = 0
#     list_losses = []
#     list_lr = []
#     mult = (lr_final_value / lr_init_value) ** (1 / num_batches)

#     settings.learning_rate = lr_init_value

#     # Model specification
#     rnn = tu.get_model(settings, len(settings.training_features))
#     criterion = nn.CrossEntropyLoss()
#     optimizer = tu.get_optimizer(settings, rnn)

#     # Prepare for GPU if required
#     if settings.use_cuda:
#         rnn.cuda()
#         criterion.cuda()

#     for batch_idxs in tqdm(list_batches, ncols=100):

#         batch_num += 1

#         # Sample a batch in packed sequence form
#         packed, _, target_tensor, idxs_rev_sort = tu.get_data_batch(
#             list_data_train, batch_idxs, settings
#         )
#         # Train step : forward backward pass
#         loss = tu.train_step(
#             settings,
#             rnn,
#             packed,
#             target_tensor,
#             criterion,
#             optimizer,
#             target_tensor.size(0),
#             len(list_batches),
#         )
#         loss = loss.detach().cpu().numpy().item()

#         # Compute the smoothed loss
#         avg_loss = beta * avg_loss + (1 - beta) * loss
#         smoothed_loss = avg_loss / (1 - beta ** batch_num)
#         # Stop if the loss is exploding
#         if batch_num > 1 and smoothed_loss > 4 * best_loss:
#             break
#         # Record the best loss
#         if smoothed_loss < best_loss or batch_num == 1:
#             best_loss = smoothed_loss
#         # Store the values
#         list_losses.append(smoothed_loss)
#         list_lr.append(lr)
#         # Update the lr for the next step
#         lr *= mult

#         # Set learning rate
#         for param_group in optimizer.param_groups:

#             param_group["lr"] = lr

#     idx_min = np.argmin(list_losses)
#     print("Min loss", list_losses[idx_min], "LR", list_lr[idx_min])

#     return list_lr[idx_min]


# def train_cyclic(settings):
#     """Train RNN models with a 1-cycle policy

#     Args:
#         settings (ExperimentSettings): controls experiment hyperparameters
#     """
#     # save training data config
#     save_normalizations(settings)

#     max_learning_rate = get_lr(settings) / 10
#     min_learning_rate = max_learning_rate / 10
#     settings.learning_rate = min_learning_rate
#     print("Setting learning rate to", min_learning_rate)

#     def one_cycle_sched(epoch, minv, maxv, phases):
#         if epoch <= phases[0]:
#             out = minv + (maxv - minv) / (phases[0]) * epoch
#         elif phases[0] < epoch <= phases[1]:
#             increment = (minv - maxv) / (phases[1] - phases[0])
#             out = maxv + increment * (epoch - phases[0])
#         else:
#             increment = (minv / 100 - minv) / (phases[2] - phases[1])
#             out = minv + increment * (epoch - phases[1])

#         return out

#     # Data
#     list_data_train, list_data_val = tu.load_HDF5(settings, test=False)

#     # Model specification
#     rnn = tu.get_model(settings, len(settings.training_features))
#     criterion = nn.CrossEntropyLoss()
#     optimizer = tu.get_optimizer(settings, rnn)

#     # Prepare for GPU if required
#     if settings.use_cuda:
#         rnn.cuda()
#         criterion.cuda()

#     # Keep track of losses for plotting
#     loss_str = ""
#     d_monitor_train = {"loss": [], "AUC": [], "Acc": [], "epoch": []}
#     d_monitor_val = {"loss": [], "AUC": [], "Acc": [], "epoch": []}
#     if "bayesian" in settings.pytorch_model_name:
#         d_monitor_train["KL"] = []
#         d_monitor_val["KL"] = []

#     lu.print_green("Starting training")

#     best_loss = float("inf")

#     settings.cyclic_phases

#     training_start_time = time()

#     for epoch in tqdm(range(settings.cyclic_phases[-1]), desc="Training", ncols=100):

#         desc = f"Epoch: {epoch} -- {loss_str}"

#         num_elem = len(list_data_train)
#         num_batches = num_elem // min(num_elem // 2, settings.batch_size)
#         list_batches = np.array_split(np.arange(num_elem), num_batches)
#         np.random.shuffle(list_batches)
#         for batch_idxs in tqdm(
#             list_batches,
#             desc=desc,
#             ncols=100,
#             bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}",
#         ):

#             # Sample a batch in packed sequence form
#             packed, _, target_tensor, idxs_rev_sort = tu.get_data_batch(
#                 list_data_train, batch_idxs, settings
#             )
#             # Train step : forward backward pass
#             tu.train_step(
#                 settings,
#                 rnn,
#                 packed,
#                 target_tensor,
#                 criterion,
#                 optimizer,
#                 target_tensor.size(0),
#                 len(list_batches),
#             )

#         for param_group in optimizer.param_groups:

#             param_group["lr"] = one_cycle_sched(
#                 epoch, min_learning_rate, max_learning_rate, settings.cyclic_phases
#             )

#         if (epoch + 1) % settings.monitor_interval == 0:

#             # Get metrics (subsample training set to same size as validation set for speed)
#             d_losses_train = tu.get_evaluation_metrics(
#                 settings, list_data_train, rnn, sample_size=len(list_data_val)
#             )
#             d_losses_val = tu.get_evaluation_metrics(
#                 settings, list_data_val, rnn, sample_size=None
#             )

#             # Add current loss avg to list of losses
#             for key in d_losses_train.keys():
#                 d_monitor_train[key].append(d_losses_train[key])
#                 d_monitor_val[key].append(d_losses_val[key])
#             d_monitor_train["epoch"].append(epoch + 1)
#             d_monitor_val["epoch"].append(epoch + 1)

#             # Prepare loss_str to update progress bar
#             loss_str = tu.get_loss_string(d_losses_train, d_losses_val)

#             tu.plot_loss(d_monitor_train, d_monitor_val, epoch, settings)
#             if d_monitor_val["loss"][-1] < best_loss:
#                 best_loss = d_monitor_val["loss"][-1]
#                 torch.save(
#                     rnn.state_dict(),
#                     f"{settings.rnn_dir}/{settings.pytorch_model_name}.pt",
#                 )

#     training_time = time() - training_start_time

#     lu.print_green("Finished training")

#     tu.save_training_results(settings, d_monitor_val, training_time)


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


def train(config):
    """Train RNN models with a decay on plateau policy

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
    """

    # Data
    list_data_train, list_data_val = tu.load_HDF5(config, SNTYPES, test=False)

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
    d_monitor_train = {"loss": [], "AUC": [], "Acc": [], "epoch": []}
    d_monitor_val = {"loss": [], "AUC": [], "Acc": [], "epoch": []}

    # TODO KL

    # TODO scheduling
    # plateau_accuracy = tu.StopOnPlateau(reduce_lr_on_plateau=True)

    best_loss = float("inf")
    training_start_time = time()

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

        if (epoch + 1) % config["monitor_interval"] == 0:

            # Get metrics (subsample training set to same size as validation set for speed)
            X_target_train, X_pred_train = get_predictions(
                model,
                list_data_train,
                list_batches_train[: len(list_batches_val)],
                device,
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

            # Prepare loss_str to update progress bar
            loss_str = tu.get_loss_string(d_losses_train, d_losses_val)

            save_prefix = f"{config['dump_dir']}/loss"
            tu.plot_loss(d_monitor_train, d_monitor_val, save_prefix)
            if d_monitor_val["loss"][-1] < best_loss:
                best_loss = d_monitor_val["loss"][-1]
                torch.save(model.state_dict(), f"{config['dump_dir']}/net.pt")

    lu.print_green("Finished training")

    training_time = time() - training_start_time


def main(config_path):

    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    # setting random seeds
    np.random.seed(config["seed"])

    # Train and sav predictions
    train(config)
    # Compute metrics
    get_metrics(config)

    logging_utils.print_blue("Finished rf training, validating and testing")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dataset building")
    parser.add_argument("config_path", help="Path to yml config gile")

    args = parser.parse_args()

    main(args.config_path)
