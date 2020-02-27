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
from matplotlib.colors import LogNorm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from supernnova.utils import training_utils as tu
from supernnova.utils import logging_utils as lu
from supernnova.utils.optim import AdaMod
from supernnova.utils import data_utils as du
from supernnova.validation import metrics
from supernnova.data.dataset import HDF5Dataset

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(in_act, n_channels):
    n_channels_int = n_channels[0]
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class WN(torch.nn.Module):
    """
    This is the WaveNet like layer for the affine coupling.  The primary difference
    from WaveNet is the convolutions need not be causal.  There is also no dilation
    size reset.  The dilation only doubles on each layer
    """

    def __init__(self, input_size, num_layers, num_channels, kernel_size):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        assert num_channels % 2 == 0
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(input_size, num_channels, 1)
        start = torch.nn.utils.weight_norm(start, name="weight")
        self.start = start

        end = torch.nn.Conv1d(num_channels, num_channels, 1)
        self.end = end

        for i in range(num_layers):
            dilation = 2 ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(
                num_channels,
                2 * num_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = torch.nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < num_layers - 1:
                res_skip_channels = 2 * num_channels
            else:
                res_skip_channels = num_channels
            res_skip_layer = torch.nn.Conv1d(num_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, audio):
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.num_channels])

        for i in range(self.num_layers):
            acts = fused_add_tanh_sigmoid_multiply(
                self.in_layers[i](audio), n_channels_tensor
            )

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.num_layers - 1:
                audio = audio + res_skip_acts[:, : self.num_channels, :]
                output = output + res_skip_acts[:, self.num_channels :, :]
            else:
                output = output + res_skip_acts

        return self.end(output)


def find_idx(array, value):

    idx = np.searchsorted(array, value, side="left")

    return min(idx, len(array))


def get_mse_loss(pred, target, mask):

    return ((pred - target).pow(2) * mask).sum() / mask.sum()


def get_cross_entropy_loss(pred, target):

    return torch.nn.functional.cross_entropy(pred, target, reduction="none").mean(0)


def get_accuracy_loss(pred, target):

    return (target == pred.argmax(1)).sum().float() / pred.shape[0]


class Model(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        dropout,
        num_embeddings,
        embedding_dim,
        normalize=False,
    ):
        super().__init__()

        self.normalize = normalize

        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)

        hidden_size = 256
        num_layers = 3
        # self.wn = WN(input_size + embedding_dim, 3, 64, 3)

        # encoder_layer = torch.nn.TransformerEncoderLayer(d_model=64, nhead=8)
        # self.tf = torch.nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Define layers
        self.rnn = torch.nn.LSTM(
            input_size + embedding_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
            bias=True,
        )
        self.output_class_layer = torch.nn.Linear(hidden_size * 2, output_size)
        self.output_peak_layer = torch.nn.Linear(hidden_size * 2, 1)

    def forward(self, x_flux, x_fluxerr, x_flt, x_time, x_mask, x_meta=None):

        x_flux = x_flux.clamp(-100)
        x_fluxerr = x_fluxerr.clamp(-100)

        x_flt = self.embedding(x_flt)

        x = torch.cat([x_flux, x_fluxerr, x_time, x_flt], dim=-1)
        # x is (B, L, D)
        # mask is (B, L)
        B, L, _ = x.shape
        if x_meta is not None:
            x_meta = x_meta.unsqueeze(1).expand(B, L, -1)
            x = torch.cat([x, x_meta], dim=-1)

        # x = self.wn(x.transpose(2, 1).contiguous()).transpose(2, 1).contiguous()
        # B, L, D
        # x = self.tf(x.transpose(1, 0).contiguous()).transpose(1, 0).contiguous()

        lengths = x_mask.sum(dim=-1).long()
        x_packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        # Pass it to the RNN
        hidden_packed, (x, _) = self.rnn(x_packed)
        x = x.transpose(1, 0).contiguous().view(B, -1)
        # undo PackedSequence
        hidden, _ = pad_packed_sequence(hidden_packed, batch_first=True)
        # hidden is (B, L, D)

        output_class = self.output_class_layer(hidden)
        output_peak = self.output_peak_layer(hidden)

        return {"X_pred_class": output_class, "X_pred_peak": output_peak}


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

    # last_time_length = data["X_mask"].sum(1) - 1
    # last_peak_preds = torch.gather(
    #     X_pred_peak.squeeze(-1), 1, (last_time_length).view(-1, 1)
    # ).squeeze(-1)

    if return_preds:
        return X_pred_class, X_target_class, X_pred_peak, X_target_peak_single

    d_losses = {}

    loss = torch.nn.SmoothL1Loss(reduction="none")(
        X_pred_peak.squeeze(-1), X_target_peak_single
    )
    loss = (loss * data["X_mask"]).sum() / data["X_mask"].sum()
    d_losses["peak_loss"] = loss
    # nn.L1Loss()(X_target_peak_single[:, 0], X_pred_peak[:, 0])
    # classification loss
    d_losses["clf_loss"] = torch.zeros(1).to(
        X_target_peak.device
    )  # get_cross_entropy_loss(X_pred_class, X_target_class)
    # Accuracy metric
    d_losses["accuracy"] = torch.zeros(1).to(
        X_target_peak.device
    )  # get_accuracy_loss(X_pred_class, X_target_class)

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


def scatter_peak(model, dataset, split, device, writer, batch):

    # Show scatter plot of predictions vs true
    data_iterator = dataset.create_iterator(split, 512, device, tqdm_desc=None)

    list_truth = []
    list_preds = []

    with torch.no_grad():

        for data in data_iterator:

            X_flux = data["X_flux"]
            X_fluxerr = data["X_fluxerr"]
            X_flt = data["X_flt"]
            X_time = data["X_time"]
            X_mask = data["X_mask"]
            X_meta = data.get("X_meta", None)
            X_target_peak_single = data["X_target_peak_single"]

            outs = model(X_flux, X_fluxerr, X_flt, X_time, X_mask, x_meta=X_meta)
            X_pred_peak = outs.get("X_pred_peak", None)

            last_time_length = data["X_mask"].sum(1) - 1
            last_peak_preds = torch.gather(
                X_pred_peak.squeeze(-1), 1, (last_time_length).view(-1, 1)
            ).squeeze(-1)

            list_truth += (
                X_target_peak_single[:, 0].view(-1).detach().cpu().numpy().tolist()
            )
            list_preds += last_peak_preds.view(-1).detach().cpu().numpy().tolist()

    vals = list_truth + list_preds

    fig = plt.figure(figsize=(9, 9))
    plt.plot(
        [min(vals), max(vals)],
        [min(vals), max(vals)],
        linestyle="--",
        color="k",
        label="Perfect pred",
    )
    plt.scatter(list_truth, list_preds)
    plt.xlabel("True Peak MJD")
    plt.ylabel("Pred Peak MJD")
    plt.legend()

    writer.add_figure(f"A/Results/scatter_{split}", fig, batch)
    plt.close(fig)
    plt.clf()
    plt.close("all")

    fig = plt.figure(figsize=(9, 9))
    plt.hist2d(list_truth, list_preds, bins=200, norm=LogNorm())
    plt.plot(
        [min(vals), max(vals)],
        [min(vals), max(vals)],
        linestyle="--",
        color="k",
        label="Perfect pred",
    )
    plt.xlabel("True Peak MJD")
    plt.ylabel("Pred Peak MJD")
    plt.legend()
    plt.colorbar()

    writer.add_figure(f"B/Results/density_{split}", fig, batch)
    plt.close(fig)
    plt.clf()
    plt.close("all")


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
    model = Model(**config["model"]).to(device)

    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

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
        print(desc)
        d_losses_train = defaultdict(list)

        for _ in range(1):

            for data in dataset.create_iterator(
                "train",
                config["batch_size"],
                device,
                # tqdm_desc=desc,
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

        model.eval()

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

        scatter_peak(model, dataset, "train", device, writer, batch)
        scatter_peak(model, dataset, "val", device, writer, batch)
        scatter_peak(model, dataset, "test", device, writer, batch)

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


def main(config_path):

    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    # setting random seeds
    np.random.seed(config["seed"])

    # Train
    train(config)
    lu.print_blue("Finished rnn training")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dataset building")
    parser.add_argument("config_path", help="Path to yml config gile")

    args = parser.parse_args()

    main(args.config_path)
