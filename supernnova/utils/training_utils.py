import torch.nn as nn
import torch
from supernnova.utils import logging_utils as lu
from supernnova.training import bayesian_rnn
from supernnova.training import bayesian_rnn_2
from supernnova.training import variational_rnn
from supernnova.training import vanilla_rnn
import os
import h5py
import json
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path

from sklearn import metrics
import matplotlib.pyplot as plt

plt.switch_backend("agg")


def normalize_arr(arr, settings):
    """Normalize array before input to RNN

    - Log transform
    - Mean and std dev normalization

    Args:
        arr (np.array) array to normalize
        settings (ExperimentSettings): controls experiment hyperparameters

    Returns:
        (np.array) the normalized array
    """

    if settings.norm == "none":
        return arr

    else:
        arr_min = settings.arr_norm[:, 0]
        arr_mean = settings.arr_norm[:, 1]
        arr_std = settings.arr_norm[:, 2]

        arr_to_norm = arr[:, settings.idx_features_to_normalize]
        # clipping
        arr_to_norm = np.clip(arr_to_norm, arr_min, np.inf)

        if settings.norm != "cosmo":
            # normalize using global norm
            arr_normed = np.log(arr_to_norm - arr_min + 1e-5)
            arr_normed = (arr_normed - arr_mean) / arr_std

        else:
            # normalize all lcs to 1 (fluxes), maintain color info
            # time is normalized as global norm
            arr_normed_cosmo = arr_to_norm
            arr_normed_cosmo[:, :-1] = (
                arr_normed_cosmo[:, :-1] / arr_normed_cosmo[:, :-1].max()
            )
            # time normalization
            tmp_cosmo = np.log(arr_to_norm[:, -1] - arr_min[-1] + 1e-5)
            arr_normed_cosmo[:, -1] = (tmp_cosmo - arr_mean[-1]) / arr_std[-1]
            arr_normed = arr_normed_cosmo

        arr[:, settings.idx_features_to_normalize] = arr_normed

    return arr


def unnormalize_arr(arr, settings):
    """UnNormalize array

    Args:
        arr (np.array) array to normalize
        settings (ExperimentSettings): controls experiment hyperparameters

    Returns:
        (np.array) the normalized array
    """

    if settings.norm == "none":
        return arr

    arr_min = settings.arr_norm[:, 0]
    arr_mean = settings.arr_norm[:, 1]
    arr_std = settings.arr_norm[:, 2]

    if settings.norm == "cosmo":
        # onyl unnormalize time
        arr_to_unnorm = arr[:, settings.idx_features_to_normalize[-1]]

        arr_to_unnorm = arr_to_unnorm * arr_std[-1] + arr_mean[-1]
        arr_unnormed = np.exp(arr_to_unnorm) + arr_min[-1] - 1e-5

        arr[:, settings.idx_features_to_normalize[-1]] = arr_unnormed

    else:
        arr_to_unnorm = arr[:, settings.idx_features_to_normalize]

        arr_to_unnorm = arr_to_unnorm * arr_std + arr_mean
        arr_unnormed = np.exp(arr_to_unnorm) + arr_min - 1e-5

        arr[:, settings.idx_features_to_normalize] = arr_unnormed

    return arr


def fill_data_list(
    idxs, arr_data, arr_target, arr_SNID, settings, n_features, desc, test=False
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

    if desc == "":
        iterator = idxs
    else:
        iterator = tqdm(idxs, desc=desc, ncols=100)

    for i in iterator:

        X_all = arr_data[i].reshape(-1, n_features)
        target = int(arr_target[i])
        lc = str(arr_SNID[i])

        # Keep an unnormalized copy of the data (for test and display)
        X_ori = X_all.copy()[:, settings.idx_features]

        # check if normalization converges
        # using clipping in case of min<model_min
        X_clip = X_all.copy()
        X_clip = np.clip(
            X_clip[:, settings.idx_features_to_normalize],
            settings.arr_norm[:, 0],
            np.inf,
        )
        X_all[:, settings.idx_features_to_normalize] = X_clip

        if settings.norm != "cosmo":
            X_tmp = unnormalize_arr(normalize_arr(X_all.copy(), settings), settings)
            assert np.all(
                np.all(np.isclose(np.ravel(X_all), np.ravel(X_tmp), atol=1e-1))
            )

        # Normalize features that need to be normalized
        X_normed = X_all.copy()
        X_normed_tmp = normalize_arr(X_normed, settings)
        # Select features as specified by the settings
        X_normed = X_normed_tmp[:, settings.idx_features]

        if test is True:
            list_data.append((X_normed, target, lc, X_all, X_ori))
        else:
            list_data.append((X_normed, target, lc))

    return list_data


def load_HDF5(settings, test=False):
    """Load data from HDF5

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
        test (bool): If True: load data for test. Default: ``False``

    Returns:
        list_data_test (list) test data tuples if test is True

        or

        Tuple containing
            - list_data_train (list): training data tuples
            - list_data_val (list): validation data tuples
    """
    file_name = f"{settings.processed_dir}/database.h5"
    lu.print_green(f"Loading {file_name}")

    with h5py.File(file_name, "r") as hf:

        list_data_train = []
        list_data_val = []

        config_name = f"{settings.source_data}_{settings.nb_classes}classes"

        dataset_split_key = f"dataset_{config_name}"
        target_key = f"target_{settings.nb_classes}classes"

        if any([settings.train_plasticc, settings.predict_plasticc]):
            target_key = "target"
            dataset_split_key = "dataset"

        if test:
            # ridiculous failsafe in case we have different classes in dataset/model
            # we will always have 2 classes
            try:
                idxs_test = np.where(hf[dataset_split_key][:] == 2)[0]
            except Exception:
                idxs_test = np.where(hf["dataset_photometry_2classes"][:] != 100)[0]
        else:
            idxs_train = np.where(hf[dataset_split_key][:] == 0)[0]
            idxs_val = np.where(hf[dataset_split_key][:] == 1)[0]
            idxs_test = np.where(hf[dataset_split_key][:] == 2)[0]

            # Shuffle for good measure
            np.random.shuffle(idxs_train)
            np.random.shuffle(idxs_val)
            np.random.shuffle(idxs_test)

            idxs_train = idxs_train[: int(settings.data_fraction * len(idxs_train))]

        n_features = hf["data"].attrs["n_features"]

        training_features = " ".join(hf["features"][:][settings.idx_features])
        lu.print_green("Features used", training_features)

        arr_data = hf["data"][:]
        if test:
            # ridiculous failsafe in case we have different classes in dataset/model
            # we will always have 2 classes
            try:
                arr_target = hf[target_key][:]
            except Exception:
                arr_target = hf["target_2classes"][:]
        else:
            arr_target = hf[target_key][:]
        arr_SNID = hf["SNID"][:]

        if test is True:
            return fill_data_list(
                idxs_test,
                arr_data,
                arr_target,
                arr_SNID,
                settings,
                n_features,
                "Loading Test Set",
                test,
            )
        else:

            list_data_train = fill_data_list(
                idxs_train,
                arr_data,
                arr_target,
                arr_SNID,
                settings,
                n_features,
                "Loading Training Set",
            )
            list_data_val = fill_data_list(
                idxs_val,
                arr_data,
                arr_target,
                arr_SNID,
                settings,
                n_features,
                "Loading Validation Set",
            )

        return list_data_train, list_data_val


def get_model(settings, input_size):
    """Create RNN model

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
        input_size (int): dimension of the input data

    Returns:
        (torch.nn Model) pytorch model
    """

    if settings.model == "vanilla":
        rnn = vanilla_rnn.VanillaRNN
    elif settings.model == "variational":
        rnn = variational_rnn.VariationalRNN
    elif settings.model == "bayesian":
        rnn = bayesian_rnn.BayesianRNN
    elif settings.model == "bayesian_2":
        rnn = bayesian_rnn_2.BayesianRNN

    rnn = rnn(input_size, settings)

    if not settings.no_dump:
        print(rnn)

    return rnn


def get_optimizer(settings, model):
    """Create gradient descent optimizer

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
        model (torch.nn Model): the pytorch model

    Returns:
        (torch.optim) the gradient descent optimizer
    """

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )

    return optimizer


def get_data_batch(list_data, idxs, settings, max_lengths=None, OOD=None):
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

    list_len = []
    list_batch = []

    for pos, i in enumerate(idxs):
        X, target, *_ = list_data[i]
        # X is (L, D)
        if OOD is not None:
            # Make a copy to be sure we do not alter the original data
            X = X.copy()
        if OOD == "reverse":
            # For OOD test, reverse the sequence
            X = np.ascontiguousarray(X[::-1])
        elif OOD == "shuffle":
            # For OOD test, shuffle X
            p = np.random.permutation(X.shape[0])
            X = X[p]
        elif OOD == "sin":
            # For OOD test, set sine values to fluxes
            arr_flux = X[:, settings.idx_flux]
            arr_fluxerr = X[:, settings.idx_fluxerr]

            X_unnorm = unnormalize_arr(X.copy(), settings)
            arr_delta_time = X_unnorm[:, settings.idx_delta_time]
            arr_MJD = np.cumsum(arr_delta_time, axis=0)

            # Sine oscillations with 30 day period
            X[:, settings.idx_flux] = np.sin(arr_MJD * 2 * np.pi / 30) * np.max(
                arr_flux, axis=0, keepdims=True
            )
            X[:, settings.idx_fluxerr] = np.random.uniform(
                arr_fluxerr.min(), arr_fluxerr.max(), size=arr_fluxerr.shape
            )
        elif OOD == "random":
            # For OOD test, set random fluxes and errors
            arr_flux = X[:, settings.idx_flux]
            arr_fluxerr = X[:, settings.idx_fluxerr]

            X[:, settings.idx_flux] = np.random.uniform(
                arr_flux.min(), arr_flux.max(), size=arr_flux.shape
            )
            X[:, settings.idx_fluxerr] = np.random.uniform(
                arr_fluxerr.min(), arr_fluxerr.max(), size=arr_fluxerr.shape
            )

        if max_lengths is not None:
            assert settings.random_length is False
            assert settings.random_redshift is False
            X = X[: max_lengths[pos]]
        if settings.random_length:
            random_length = np.random.randint(1, X.shape[0] + 1)
            X = X[:random_length]
        if settings.redshift == "zspe" and settings.random_redshift:
            if np.random.binomial(1, 0.5) == 0:
                X[:, settings.idx_specz] = -1
        input_dim = X.shape[1]
        list_len.append(X.shape[0])
        list_batch.append((X, target))

    # Get indices to sort the batch by sequence size (needed to use packed sequences in pytorch)
    # Sequences should be arranged in decreasing length
    idx_sort = np.argsort(list_len)[::-1]
    idxs_rev_sort = np.argsort(idx_sort)  # these indices revert the sort
    max_len = list_len[idx_sort[0]]
    X_tensor = torch.zeros((max_len, len(idxs), input_dim))
    list_target = []
    lengths = []
    # Assign values for the tensor
    for i, idx in enumerate(idx_sort):
        X, target = list_batch[idx]
        try:
            X_tensor[: X.shape[0], i, :] = torch.FloatTensor(X)
        except Exception:
            X_tensor[: X.shape[0], i, :] = torch.FloatTensor(
                torch.from_numpy(np.flip(X, axis=0).copy())
            )
        list_target.append(target)
        lengths.append(list_len[idx])

    # Move data to GPU if required
    if settings.use_cuda:
        X_tensor = X_tensor.cuda()
        target_tensor = torch.LongTensor(list_target).cuda()

    else:
        X_tensor = X_tensor
        target_tensor = torch.LongTensor(list_target)

    # Create a packed sequence
    packed_tensor = nn.utils.rnn.pack_padded_sequence(X_tensor, lengths)

    return packed_tensor, X_tensor, target_tensor, idxs_rev_sort


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


def save_randomforest_model(settings, clf):
    """Save RandomForest model

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
        clf (RandomForestClassifier): RandomForest model
    """

    filename = f"{settings.rf_dir}/{settings.randomforest_model_name}.pickle"
    with open(filename, "wb") as f:
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
