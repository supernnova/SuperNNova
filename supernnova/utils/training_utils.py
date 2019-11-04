import os
import h5py
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn import metrics
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from . import logging_utils as lu
from supernnova.training import bayesian_rnn
from supernnova.training import variational_rnn
from supernnova.training import vanilla_rnn


plt.switch_backend("agg")


def log_norm(x, min_clip, mean, std, F=torch):
    """
    """

    x = (F.log(x - min_clip + 1e-5) - mean) / std
    return x


def inverse_log_norm(x, min_clip, mean, std):

    x = F.exp(x * std + mean) + min_clip - 1e-5

    return x


def fill_data_list(
    idxs, arr_data, arr_meta, arr_target, arr_SNID, list_features, desc, test=False
):
    """Utility to create a list of data tuples used as inputs to RNN model

    The ``settings`` object specifies which feature are selected

    Args:
        idxs (np.array or list): idx of data point to select
        arr_data (np.array): features
        arr_target (np.array): target
        arr_SNID (np.array): lightcurve unique ID
        settings (ExperimentSettings): controls experiment hyperparameters
        n_features (int): total number of features in arr_data
        desc (str): message to display while loading
        test (bool): If True: add more data to the list, as it is required at test time.
            Default: ``False``

    Returns:
        (list) the list of data tuples
    """

    list_data = []

    iterator = tqdm(idxs, desc=desc, ncols=100) if desc != "" else idxs
    n_features = len(list_features)

    flux_features_idxs = [
        i for i in range(n_features) if "FLUXCAL_" in list_features[i]
    ]
    fluxerr_features_idxs = [
        i for i in range(n_features) if "FLUXCALERR_" in list_features[i]
    ]
    time_idxs = list_features.index("delta_time")
    flt_idxs = list_features.index("FLT")

    for i in iterator:

        X = arr_data[i].reshape(-1, n_features)

        X_flux = X[:, flux_features_idxs]
        X_fluxerr = X[:, fluxerr_features_idxs]
        X_time = X[:, time_idxs]
        X_flt = X[:, flt_idxs]

        target = int(arr_target[i])
        SNID = int(arr_SNID[i])

        d = {
            "X_flux": X_flux,
            "X_fluxerr": X_fluxerr,
            "X_time": X_time,
            "X_flt": X_flt,
            "X_target": target,
            "SNID": SNID,
        }

        if arr_meta is not None:
            d["X_meta"] = arr_meta[i]

        list_data.append(d)

    return list_data


def load_HDF5(config, sntypes, test=False):
    """Load data from HDF5

    Args:
        config (ExperimentSettings): controls experiment hyperparameters
        test (bool): If True: load data for test. Default: ``False``

    Returns:
        list_data_test (list) test data tuples if test is True

        or

        Tuple containing
            - list_data_train (list): training data tuples
            - list_data_val (list): validation data tuples
    """
    processed_dir = config["processed_dir"]
    file_name = f"{processed_dir}/database.h5"
    lu.print_green(f"Loading {file_name}")

    with h5py.File(file_name, "r") as hf:

        list_data_train = []
        list_data_val = []

        # Load data
        arr_data = hf["data"][:]
        arr_SNID = hf["SNID"][:]
        arr_SNTYPE = hf["SNTYPE"][:]

        list_features = hf["data"].attrs["columns"].tolist()

        # Load metadata
        metadata_features = config.get("metadata_features", [])
        arr_meta = hf["metadata"][:]
        columns = hf["metadata"].attrs["columns"]
        df_meta = pd.DataFrame(arr_meta, columns=columns)
        df_meta["SNID"] = arr_SNID
        df_meta["SNTYPE"] = arr_SNTYPE

        # Prepare metadata features array
        arr_meta = df_meta[metadata_features].values if metadata_features else None

        # Create target
        class_map = {}
        nb_classes = config["nb_classes"]
        if nb_classes == 2:
            for key, value in sntypes.items():
                class_map[int(key)] = 0 if value == "Ia" else 1
        else:
            for i, key in enumerate(sntypes):
                class_map[int(key)] = i
        df_meta["target"] = df_meta["SNTYPE"].map(class_map)

        arr_target = df_meta["target"].values

        # Subsample with data fraction
        n_samples = int(config.get("data_fraction", 1) * len(df_meta))
        idxs = np.random.choice(len(df_meta), n_samples, replace=False)
        df_meta = df_meta.iloc[idxs].reset_index(drop=True)

        # Pandas magic to downample each class down to lowest cardinality class
        df_meta = df_meta.groupby("target")
        df_meta = (
            df_meta.apply(lambda x: x.sample(df_meta.size().min()))
            .reset_index(drop=True)
            .sample(frac=1)
        ).reset_index(drop=True)

        n_samples = len(df_meta)

        for t in range(nb_classes):
            n = len(df_meta[df_meta.target == t])
            print(
                f"{n} ({100 * n / n_samples:.2f} %) class {t} samples after balancing"
            )

        # 80/10/10 Train/val/test split
        n_train = int(0.8 * n)
        n_val = int(0.9 * n)
        SNID_train = df_meta["SNID"].values[:n_train]
        SNID_val = df_meta["SNID"].values[n_train:n_val]
        SNID_test = df_meta["SNID"].values[n_val:]

        idxs_train = np.where(np.in1d(arr_SNID, SNID_train))[0]
        idxs_val = np.where(np.in1d(arr_SNID, SNID_val))[0]
        idxs_test = np.where(np.in1d(arr_SNID, SNID_test))[0]

        # Shuffle for good measure
        np.random.shuffle(idxs_train)
        np.random.shuffle(idxs_val)
        np.random.shuffle(idxs_test)

        if test is True:
            return fill_data_list(
                idxs_test,
                arr_data,
                arr_meta,
                arr_target,
                arr_SNID,
                list_features,
                "Loading Test Set",
                test,
            )
        else:

            list_data_train = fill_data_list(
                idxs_train,
                arr_data,
                arr_meta,
                arr_target,
                arr_SNID,
                list_features,
                "Loading Training Set",
            )
            list_data_val = fill_data_list(
                idxs_val,
                arr_data,
                arr_meta,
                arr_target,
                arr_SNID,
                list_features,
                "Loading Validation Set",
            )

        return list_data_train, list_data_val


def get_data_batch(list_data, idxs, device):
    """Create a batch in a deterministic way

    Args:
        list_data: (list) tuples of (X, target, lightcurve_ID)
        idxs: (array / list) indices of batch element in list_data
        settings (ExperimentSettings): controls experiment hyperparameters
        max_length (int): Maximum light curve length to be used Default: ``None``.
        OOD (str): Whether to modify data to create out of distribution data to be used Default: ``None``.

    Returns:
        Tuple containing
            - packed_tensor (torch PackedSequence): the packed features
            - X_tensor (torch Tensor): the features
            - target_tensor (torch Tensor): the target
    """

    list_lengths = [list_data[i]["X_flux"].shape[0] for i in idxs]
    B = len(idxs)
    L = max(list_lengths)
    Dflux = list_data[0]["X_flux"].shape[1]
    Dfluxerr = list_data[0]["X_fluxerr"].shape[1]

    if "X_meta" in list_data[0]:
        has_meta = True
        Dmeta = list_data[0]["X_meta"].shape[0]

    X_flux = np.zeros((B, L, Dflux), dtype=np.float32)
    X_fluxerr = np.zeros((B, L, Dfluxerr), dtype=np.float32)
    X_time = np.zeros((B, L, 1), dtype=np.float32)
    X_meta = np.zeros((B, Dmeta), dtype=np.float32) if has_meta else None
    X_flt = np.zeros((B, L), dtype=np.int64)
    X_target = np.zeros((B,), dtype=np.int64)

    arr_lengths = np.array(list_lengths).astype(np.int64)

    for pos, idx in enumerate(idxs):

        data = list_data[idx]
        length = data["X_flux"].shape[0]

        X_flux[pos, :length, :] = data["X_flux"]
        X_fluxerr[pos, :length, :] = data["X_fluxerr"]
        X_time[pos, :length, 0] = data["X_time"]
        X_flt[pos, :length] = data["X_flt"]
        X_target[pos] = data["X_target"]

        if has_meta:
            X_meta[pos] = data["X_meta"]

    X_mask = (arr_lengths.reshape(-1, 1) > np.arange(L).reshape(1, -1)).astype(np.bool)

    out = {
        "X_flux": torch.from_numpy(X_flux).to(device),
        "X_fluxerr": torch.from_numpy(X_fluxerr).to(device),
        "X_time": torch.from_numpy(X_time).to(device),
        "X_flt": torch.from_numpy(X_flt).to(device),
        "X_target": torch.from_numpy(X_target).to(device),
        "X_mask": torch.from_numpy(X_mask).to(device),
    }

    if has_meta:
        out["X_meta"] = torch.from_numpy(X_meta).to(device)

    return out


def train_step(
    settings,
    rnn,
    packed_tensor,
    target_tensor,
    criterion,
    optimizer,
    batch_size,
    num_batches,
):
    """Full training step : Forward and Backward pass

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
        rnn (torch.nn Model): pytorch model to train
        packed_tensor (torch PackedSequence): input tensor in packed form
        target_tensor (torch Tensor): target tensor
        criterion (torch loss function): loss function to optimize
        optimizer (torch optim): the gradient descent optimizer
        batch_size (int): batch size
        num_batches (int): number of minibatches to scale KL cost in Bayesian
    """

    # Set NN to train mode (deals with dropout and batchnorm)
    rnn.train()

    # Zero out the gradients
    optimizer.zero_grad()

    # Forward pass
    output = rnn(packed_tensor)
    loss = criterion(output.squeeze(), target_tensor)
    # Special case for BayesianRNN, need to use KL loss
    if isinstance(rnn, bayesian_rnn.BayesianRNN):
        loss = loss + rnn.kl / (num_batches * batch_size)
    else:
        loss = criterion(output.squeeze(), target_tensor)

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss


def eval_step(rnn, packed_tensor, batch_size):
    """Eval step: Forward pass only

    Args:
        rnn (torch.nn Model): pytorch model to train
        packed_tensor (torch PackedSequence): input tensor in packed form
        batch_size (int): batch size

    Returns:
        output (torch Tensor): output of rnn
    """

    # Set NN to eval mode (deals with dropout and batchnorm)
    rnn.eval()

    # Forward pass
    output = rnn(packed_tensor)

    return output


def plot_loss(d_train, d_val, epoch, settings):
    """Plot loss curves

    Plot training and validation logloss

    Args:
        d_train (dict of arrays): training log losses
        d_val (dict of arrays): validation log losses
        epoch (int): current epoch
        settings (ExperimentSettings): custom class to hold hyperparameters
    """

    for key in d_train.keys():

        plt.figure()
        plt.plot(d_train["epoch"], d_train[key], label="Train %s" % key.title())
        plt.plot(d_val["epoch"], d_val[key], label="Val %s" % key.title())
        plt.legend(loc="best", fontsize=18)
        plt.xlabel("Step", fontsize=22)
        plt.tight_layout()
        plt.savefig(
            Path(settings.models_dir)
            / f"{settings.pytorch_model_name}"
            / f"train_and_val_{key}_{settings.pytorch_model_name}.png"
        )
        plt.close()
        plt.clf()


def get_evaluation_metrics(settings, list_data, model, sample_size=None):
    """Compute evaluation metrics on a list of data points

    Args:
        settings (ExperimentSettings): custom class to hold hyperparameters
        list_data (list): contains data to evaluate
        model (torch.nn Model): pytorch model
        sample_size (int): subset of the data to use for validation. Default: ``None``

    Returns:
        d_losses (dict) maps metrics to their computed value
    """

    # Validate
    list_pred = []
    list_target = []
    list_kl = []
    num_elem = len(list_data)
    num_batches = num_elem // min(num_elem // 2, settings.batch_size)
    list_batches = np.array_split(np.arange(num_elem), num_batches)

    # If required, pick a subset of list batches at random
    if sample_size:
        batch_idxs = np.random.permutation(len(list_batches))
        num_batches = sample_size // min(sample_size // 2, settings.batch_size)
        batch_idxs = batch_idxs[:num_batches]
        list_batches = [list_batches[batch_idx] for batch_idx in batch_idxs]

    for batch_idxs in list_batches:
        random_length = settings.random_length
        settings.random_length = False
        packed_tensor, X_tensor, target_tensor, idxs_rev_sort = get_data_batch(
            list_data, batch_idxs, settings
        )
        settings.random_length = random_length
        output = eval_step(model, packed_tensor, X_tensor.size(1))

        if "bayesian" in settings.pytorch_model_name:
            list_kl.append(model.kl.detach().cpu().item())

        # Apply softmax
        pred_proba = nn.functional.softmax(output, dim=1)

        # Convert to numpy array
        pred_proba = pred_proba.data.cpu().numpy()
        target_numpy = target_tensor.data.cpu().numpy()

        # Revert sort
        pred_proba = pred_proba[idxs_rev_sort]
        target_numpy = target_numpy[idxs_rev_sort]

        list_pred.append(pred_proba)
        list_target.append(target_numpy)
    targets = np.concatenate(list_target, axis=0)
    preds = np.concatenate(list_pred, axis=0)

    # Check outputs size
    assert len(targets.shape) == 1
    assert len(preds.shape) == 2

    if settings.nb_classes == 2:
        auc = metrics.roc_auc_score(targets, preds[:, 1])
    else:
        # Can't compute AUC for more than 2 classes
        auc = None
    acc = metrics.accuracy_score(targets, np.argmax(preds, 1))
    targets_2D = np.zeros((targets.shape[0], settings.nb_classes))
    for i in range(targets.shape[0]):
        targets_2D[i, targets[i]] = 1
    log_loss = metrics.log_loss(targets_2D, preds)

    d_losses = {"AUC": auc, "Acc": acc, "loss": log_loss}

    if len(list_kl) != 0:
        d_losses["KL"] = np.mean(list_kl)

    return d_losses


def get_loss_string(d_losses_train, d_losses_val):
    """Obtain a loss string to display training progress

    Args:
        d_losses_train (dict): maps {metric:value} for the training data
        d_losses_val (dict): maps {metric:value} for the validation data

    Returns:
        loss_str (str): the loss string to display
    """

    loss_str = "/".join(d_losses_train.keys())

    loss_str += " [T]: " + "/".join(
        [
            f"{value:.3g}" if (value is not None and key != "epoch") else "NA"
            for (key, value) in d_losses_train.items()
        ]
    )
    loss_str += " [V]: " + "/".join(
        [
            f"{value:.3g}" if (value is not None and key != "epoch") else "NA"
            for (key, value) in d_losses_val.items()
        ]
    )

    return loss_str


def save_training_results(settings, d_monitor, training_time):
    """Obtain a loss string to display training progress

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
        d_monitor (dict): maps {metric:value}
        training_time (float): amount of time training took

    Returns:
        loss_str (str): the loss string to display
    """

    d_results = {"training_time": training_time}
    for key in ["AUC", "Acc"]:
        if key == "AUC" and settings.nb_classes > 2:
            d_results[key] = -1
        else:
            d_results[key] = max(d_monitor[key])
    d_results["loss"] = min(d_monitor["loss"])

    try:
        with open(Path(settings.rnn_dir) / "training_log.json", "r") as f:
            d_out = json.load(f)
    except Exception:
        d_out = {}

    with open(Path(settings.rnn_dir) / "training_log.json", "w") as f:
        d_out.update({settings.pytorch_model_name: d_results})
        json.dump(d_out, f)


#######################
# RandomForest Utils
#######################


def save_randomforest_model(save_file, clf):
    """Save RandomForest model

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
        clf (RandomForestClassifier): RandomForest model
    """

    with open(save_file, "wb") as f:
        pickle.dump(clf, f)
    lu.print_green("Saved model")


def load_randomforest_model(settings, model_file=None):
    """Load RandomForest model

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
        model_file (str): path to saved randomforest model. Default: ``None``

    Returns:
        (RandomForestClassifier) RandomForest model
    """

    if model_file is None:
        model_file = f"{settings.rf_dir}/{settings.randomforest_model_name}.pickle"
    assert os.path.isfile(model_file)
    with open(model_file, "rb") as f:
        clf = pickle.load(f)
    lu.print_green("Loaded model")

    return clf


def train_and_evaluate_randomforest_model(clf, X_train, y_train, X_val, y_val):
    """Train a RandomForestClassifier and evaluate AUC, precision, accuracy
    on a validation set

    Args:
        clf (RandomForestClassifier): RandomForest model to fit and evaluate
        X_train (np.array): the training features
        y_train (np.array): the training target
        X_val (np.array): the validation features
        y_val (np.array): the validation target
    """
    lu.print_green("Fitting RandomForest...")
    clf = clf.fit(X_train, y_train)
    lu.print_green("Fitting complete")

    # Evaluate our classifier
    probas_ = clf.predict_proba(X_val)
    # Compute AUC and precision
    fpr, tpr, thresholds = metrics.roc_curve(y_val, probas_[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    pscore = metrics.precision_score(y_val, clf.predict(X_val), average="binary")
    lu.print_green("Validation AUC", roc_auc)
    lu.print_green("Validation precision score", pscore)

    lu.print_green(
        "Train data accuracy",
        100 * (sum(clf.predict(X_train) == y_train)) / X_train.shape[0],
    )
    lu.print_green(
        "Val data accuracy", 100 * (sum(clf.predict(X_val) == y_val)) / X_val.shape[0]
    )

    return clf


class StopOnPlateau(object):
    """
    Detect plateau on accuracy (or any metric)
    If chosen, will reduce learning rate of optimizer once in the Plateau

    .. code: python

          plateau_accuracy = tu.StopOnPlateau()
          for epoch in range(10):
              ... get metric ...
              plateau = plateau_accuracy.step(metric_value)
              if plateau is True:
                   break

    Args:
        patience (int): number of epochs to wait, after which we decrease the LR
            if the validation loss is plateauing
        reduce_lr-on_plateau (bool): If True, reduce LR after loss has not improved
            in the last patience epochs
        max_learning_rate_reduction (float): max factor by which to reduce the learning rate
    """

    def __init__(
        self, patience=10, reduce_lr_on_plateau=False, max_learning_rate_reduction=3
    ):

        self.patience = patience
        self.best = 0.0
        self.num_bad_epochs = 0
        self.is_better = None
        self.last_epoch = -1
        self.list_metric = []
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.max_learning_rate_reduction = max_learning_rate_reduction
        self.learning_rate_reduction = 0

    def step(self, metric_value, optimizer=None, epoch=None):
        current = metric_value
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # Are we under .05 std in accuracy on the last 10 epochs
        self.list_metric.append(current)
        if len(self.list_metric) > 10:
            self.list_metric = self.list_metric[-10:]
            # are we in a plateau?
            # accuracy is not in percentage, so two decimal numbers is actually 4 in this notation
            if np.array(self.list_metric).std() < 0.0005:
                print("Has reached a learning plateau with", current, "\n")
                if optimizer is not None and self.reduce_lr_on_plateau is True:
                    print(
                        "Reducing learning rate by factor of ten",
                        self.learning_rate_reduction,
                        "\n",
                    )
                    for param in optimizer.param_groups:
                        param["lr"] = param["lr"] / 10.0
                    self.learning_rate_reduction += 1
                    if self.learning_rate_reduction == self.max_learning_rate_reduction:
                        return True
                else:
                    return True
            else:
                return False
