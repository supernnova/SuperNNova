import os
import pickle
import numpy as np
from pathlib import Path
from sklearn import metrics
import matplotlib.pyplot as plt

import torch

from . import logging_utils as lu


plt.switch_backend("agg")


def get_data_batch(list_data, idxs, device, max_lengths=None):
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

    list_lengths = (
        [list_data[i]["X_flux"].shape[0] for i in idxs]
        if max_lengths is None
        else max_lengths
    )
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
        length = list_lengths[pos]

        X_flux[pos, :length, :] = data["X_flux"][:length]
        X_fluxerr[pos, :length, :] = data["X_fluxerr"][:length]
        X_time[pos, :length, 0] = data["X_time"][:length]
        X_flt[pos, :length] = data["X_flt"][:length]
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


def plot_loss(d_train, d_val, save_prefix):
    """Plot loss curves

    Plot training and validation logloss

    Args:
        d_train (dict of arrays): training log losses
        d_val (dict of arrays): validation log losses
        settings (ExperimentSettings): custom class to hold hyperparameters
    """

    Path(save_prefix).parent.mkdir(exist_ok=True, parents=True)

    for key in d_train.keys():

        plt.figure()
        plt.plot(d_train["epoch"], d_train[key], label=f"Train {key.title()}")
        plt.plot(d_val["epoch"], d_val[key], label=f"Val {key.title()}")
        plt.legend(loc="best", fontsize=18)
        plt.xlabel("Step", fontsize=22)
        plt.tight_layout()
        plt.savefig(save_prefix + f"_{key}.png")
        plt.close()
        plt.clf()


def get_evaluation_metrics(preds, targets, nb_classes=2):

    if nb_classes == 2:
        auc = metrics.roc_auc_score(targets, preds[:, 1])
    else:
        # Can't compute AUC for more than 2 classes
        auc = None
    acc = metrics.accuracy_score(targets, np.argmax(preds, 1))
    targets_2D = np.zeros((targets.shape[0], nb_classes))
    for i in range(targets.shape[0]):
        targets_2D[i, targets[i]] = 1
    log_loss = metrics.log_loss(targets_2D, preds)

    d_losses = {"AUC": auc, "Acc": acc, "log_loss": log_loss}

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

