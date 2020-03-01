import torch
import json
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from time import time
from pathlib import Path
from ..utils import training_utils as tu
from ..utils import logging_utils as lu


def get_lr(settings):
    """Select optimal starting learning rate when training with a 1-cycle policy

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
    """

    # Data
    list_data_train, list_data_val = tu.load_HDF5(settings, test=False)

    num_elem = len(list_data_train)
    num_batches = num_elem // min(num_elem // 2, settings.batch_size)
    list_batches = np.array_split(np.arange(num_elem), num_batches)
    np.random.shuffle(list_batches)

    lr_init_value = 1e-8
    lr = float(lr_init_value)
    lr_final_value = 10.0
    beta = 0.98
    avg_loss = 0.0
    best_loss = 0.0
    batch_num = 0
    list_losses = []
    list_lr = []
    mult = (lr_final_value / lr_init_value) ** (1 / num_batches)

    settings.learning_rate = lr_init_value

    # Model specification
    rnn = tu.get_model(settings, len(settings.training_features))
    criterion = nn.CrossEntropyLoss()
    optimizer = tu.get_optimizer(settings, rnn)

    # Prepare for GPU if required
    if settings.use_cuda:
        rnn.cuda()
        criterion.cuda()

    for batch_idxs in tqdm(list_batches, ncols=100):

        batch_num += 1

        # Sample a batch in packed sequence form
        packed, _, target_tensor, idxs_rev_sort = tu.get_data_batch(
            list_data_train, batch_idxs, settings
        )
        # Train step : forward backward pass
        loss = tu.train_step(
            settings,
            rnn,
            packed,
            target_tensor,
            criterion,
            optimizer,
            target_tensor.size(0),
            len(list_batches),
        )
        loss = loss.detach().cpu().numpy().item()

        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss
        smoothed_loss = avg_loss / (1 - beta ** batch_num)
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            break
        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
        # Store the values
        list_losses.append(smoothed_loss)
        list_lr.append(lr)
        # Update the lr for the next step
        lr *= mult

        # Set learning rate
        for param_group in optimizer.param_groups:

            param_group["lr"] = lr

    idx_min = np.argmin(list_losses)
    print("Min loss", list_losses[idx_min], "LR", list_lr[idx_min])

    return list_lr[idx_min]


def train_cyclic(settings):
    """Train RNN models with a 1-cycle policy

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
    """
    # save training data config
    save_normalizations(settings)

    max_learning_rate = get_lr(settings) / 10
    min_learning_rate = max_learning_rate / 10
    settings.learning_rate = min_learning_rate
    print("Setting learning rate to", min_learning_rate)

    def one_cycle_sched(epoch, minv, maxv, phases):
        if epoch <= phases[0]:
            out = minv + (maxv - minv) / (phases[0]) * epoch
        elif phases[0] < epoch <= phases[1]:
            increment = (minv - maxv) / (phases[1] - phases[0])
            out = maxv + increment * (epoch - phases[0])
        else:
            increment = (minv / 100 - minv) / (phases[2] - phases[1])
            out = minv + increment * (epoch - phases[1])

        return out

    # Data
    list_data_train, list_data_val = tu.load_HDF5(settings, test=False)

    # Model specification
    rnn = tu.get_model(settings, len(settings.training_features))
    criterion = nn.CrossEntropyLoss()
    optimizer = tu.get_optimizer(settings, rnn)

    # Prepare for GPU if required
    if settings.use_cuda:
        rnn.cuda()
        criterion.cuda()

    # Keep track of losses for plotting
    loss_str = ""
    d_monitor_train = {"loss": [], "AUC": [], "Acc": [], "epoch": []}
    d_monitor_val = {"loss": [], "AUC": [], "Acc": [], "epoch": []}
    if "bayesian" in settings.pytorch_model_name:
        d_monitor_train["KL"] = []
        d_monitor_val["KL"] = []

    lu.print_green("Starting training")

    best_loss = float("inf")

    settings.cyclic_phases

    training_start_time = time()

    for epoch in tqdm(range(settings.cyclic_phases[-1]), desc="Training", ncols=100):

        desc = f"Epoch: {epoch} -- {loss_str}"

        num_elem = len(list_data_train)
        num_batches = num_elem // min(num_elem // 2, settings.batch_size)
        list_batches = np.array_split(np.arange(num_elem), num_batches)
        np.random.shuffle(list_batches)
        for batch_idxs in tqdm(
            list_batches,
            desc=desc,
            ncols=100,
            bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}",
        ):

            # Sample a batch in packed sequence form
            packed, _, target_tensor, idxs_rev_sort = tu.get_data_batch(
                list_data_train, batch_idxs, settings
            )
            # Train step : forward backward pass
            tu.train_step(
                settings,
                rnn,
                packed,
                target_tensor,
                criterion,
                optimizer,
                target_tensor.size(0),
                len(list_batches),
            )

        for param_group in optimizer.param_groups:

            param_group["lr"] = one_cycle_sched(
                epoch, min_learning_rate, max_learning_rate, settings.cyclic_phases
            )

        if (epoch + 1) % settings.monitor_interval == 0:

            # Get metrics (subsample training set to same size as validation set for speed)
            d_losses_train = tu.get_evaluation_metrics(
                settings, list_data_train, rnn, sample_size=len(list_data_val)
            )
            d_losses_val = tu.get_evaluation_metrics(
                settings, list_data_val, rnn, sample_size=None
            )

            # Add current loss avg to list of losses
            for key in d_losses_train.keys():
                d_monitor_train[key].append(d_losses_train[key])
                d_monitor_val[key].append(d_losses_val[key])
            d_monitor_train["epoch"].append(epoch + 1)
            d_monitor_val["epoch"].append(epoch + 1)

            # Prepare loss_str to update progress bar
            loss_str = tu.get_loss_string(d_losses_train, d_losses_val)

            tu.plot_loss(d_monitor_train, d_monitor_val, epoch, settings)
            if d_monitor_val["loss"][-1] < best_loss:
                best_loss = d_monitor_val["loss"][-1]
                torch.save(
                    rnn.state_dict(),
                    f"{settings.rnn_dir}/{settings.pytorch_model_name}.pt",
                )

    training_time = time() - training_start_time

    lu.print_green("Finished training")

    tu.save_training_results(settings, d_monitor_val, training_time)


def save_normalizations(settings):
    """Save normalization used for training

    Saves a json file with the normalization used for each feature

    Arguments:
        settings (ExperimentSettings): controls experiment hyperparameters
    """

    dic_norm = {}
    for i, f in enumerate(settings.training_features_to_normalize):
        dic_norm[f] = {}
        for j, w in enumerate(["min", "mean", "std"]):
            dic_norm[f][w] = float(settings.arr_norm[i, j])

    fname = f"{Path(settings.rnn_dir)}/data_norm.json"
    with open(fname, "w") as f:
        json.dump(dic_norm, f, indent=4, sort_keys=True)


def train(settings):
    """Train RNN models with a decay on plateau policy

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
    """

    # save training data config
    save_normalizations(settings)

    # Data
    list_data_train, list_data_val = tu.load_HDF5(settings, test=False)
    # Model specification
    rnn = tu.get_model(settings, len(settings.training_features))
    if settings.__class__.__name__ == "PlasticcSettings":
        criterion = nn.CrossEntropyLoss(
            weight=torch.from_numpy(
                np.array([1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1]).astype(np.float32)
            )
        )
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = tu.get_optimizer(settings, rnn)

    # Prepare for GPU if required
    if settings.use_cuda:
        rnn.cuda()
        criterion.cuda()

    # Keep track of losses for plotting
    loss_str = ""
    d_monitor_train = {"loss": [], "AUC": [], "Acc": [], "epoch": []}
    d_monitor_val = {"loss": [], "AUC": [], "Acc": [], "epoch": []}
    if "bayesian" in settings.pytorch_model_name:
        d_monitor_train["KL"] = []
        d_monitor_val["KL"] = []

    lu.print_green("Starting training")

    plateau_accuracy = tu.StopOnPlateau(reduce_lr_on_plateau=True)

    best_loss = float("inf")

    training_start_time = time()

    for epoch in tqdm(range(settings.nb_epoch), desc="Training", ncols=100):

        desc = f"Epoch: {epoch} -- {loss_str}"

        num_elem = len(list_data_train)
        num_batches = num_elem // min(num_elem // 2, settings.batch_size)
        list_batches = np.array_split(np.arange(num_elem), num_batches)
        np.random.shuffle(list_batches)
        for batch_idxs in tqdm(
            list_batches,
            desc=desc,
            ncols=100,
            bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}",
        ):

            # Sample a batch in packed sequence form
            packed, _, target_tensor, idxs_rev_sort = tu.get_data_batch(
                list_data_train, batch_idxs, settings
            )
            # Train step : forward backward pass
            tu.train_step(
                settings,
                rnn,
                packed,
                target_tensor,
                criterion,
                optimizer,
                target_tensor.size(0),
                len(list_batches),
            )

        if (epoch + 1) % settings.monitor_interval == 0:

            # Get metrics (subsample training set to same size as validation set for speed)
            d_losses_train = tu.get_evaluation_metrics(
                settings, list_data_train, rnn, sample_size=len(list_data_val)
            )
            d_losses_val = tu.get_evaluation_metrics(
                settings, list_data_val, rnn, sample_size=None
            )

            end_condition = plateau_accuracy.step(d_losses_val["Acc"], optimizer)
            if end_condition is True:
                break

            # Add current loss avg to list of losses
            for key in d_losses_train.keys():
                d_monitor_train[key].append(d_losses_train[key])
                d_monitor_val[key].append(d_losses_val[key])
            d_monitor_train["epoch"].append(epoch + 1)
            d_monitor_val["epoch"].append(epoch + 1)

            # Prepare loss_str to update progress bar
            loss_str = tu.get_loss_string(d_losses_train, d_losses_val)

            tu.plot_loss(d_monitor_train, d_monitor_val, epoch, settings)
            if d_monitor_val["loss"][-1] < best_loss:
                best_loss = d_monitor_val["loss"][-1]
                torch.save(
                    rnn.state_dict(),
                    f"{settings.rnn_dir}/{settings.pytorch_model_name}.pt",
                )

    lu.print_green("Finished training")

    training_time = time() - training_start_time

    tu.save_training_results(settings, d_monitor_val, training_time)
