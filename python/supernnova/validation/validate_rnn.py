import os
import warnings
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from pathlib import Path
from sklearn import metrics
import torch
import torch.nn as nn

from ..utils import data_utils as du
from ..utils import training_utils as tu
from ..utils import logging_utils as lu
from supernnova.utils.swag_utils import SwagModel


def find_idx(array, value):
    """Utility to find the index of the element of ``array`` that most closely
    matches ``value``

    Args:
        array (np.array): The array in which to search
        value (float): The value for which we are looking for a match

    Returns:
        (int) the index of of the element of ``array`` that most closely
        matches ``value``

    """

    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):
        return idx - 1
    else:
        return idx


def get_batch_predictions(rnn, X, target):
    """Utility to obtain predictions for a given batch

    Args:
        rnn (torch.nn): The RNN model
        X (torch.Tensor): The batch on which to carry out predictions
        target (torch.longTensor): The true class of each element in the batch

    Returns:
        Tuple containing

        - arr_preds (np.array): predictions
        - arr_target (np.array): actual targets

    """

    out = rnn.forward(X)
    arr_preds = nn.functional.softmax(out, dim=-1).data.cpu().numpy()
    arr_target = target.detach().cpu().numpy()

    return arr_preds, arr_target


def get_batch_predictions_MFE(rnn, X, target):
    """Utility to obtain predictions for a given batch

    Args:
        rnn (torch.nn): The RNN model
        X (torch.Tensor): The batch on which to carry out predictions
        target (torch.longTensor): The true class of each element in the batch

    Returns:
        Tuple containing

        - arr_preds (np.array): predictions
        - arr_target (np.array): actual targets

    """

    out = rnn.forward(X, mean_field_inference=True)
    arr_preds = nn.functional.softmax(out, dim=-1).data.cpu().numpy()
    arr_target = target.detach().cpu().numpy()

    return arr_preds, arr_target


def get_batch_predictions_SWAG(model: SwagModel, X, target, scale: float, cov=True):
    """Utility to obtain predictions for a given batch

    Args:
        model: The SwagModel model
        X (torch.Tensor): The batch on which to carry out predictions
        target (torch.longTensor): The true class of each element in the batch
        scale (float): The scale parameter for covariance
        cov (bool): If True, enable calculating low-rank covariance

    Returns:
        Tuple containing

        - arr_preds (np.array): predictions
        - arr_target (np.array): actual targets

    """

    sample_model = model.sample(scale, cov)
    out = sample_model.forward(X)
    arr_preds = nn.functional.softmax(out, dim=-1).data.cpu().numpy()
    arr_target = target.detach().cpu().numpy()

    return arr_preds, arr_target


def dispatch_batch_predictions(model, X, target, mode=None, **kwargs):
    """
    Dispatch different batch prediction functions based on given mode.
    """
    if mode is None:
        arr_preds, arr_target = get_batch_predictions(model, X, target)
    elif mode == "MFE":
        arr_preds, arr_target = get_batch_predictions_MFE(model, X, target)
    elif mode == "SWAG":
        scale = kwargs.get("scale", 0.5)
        cov = kwargs.get("cov", True)
        arr_preds, arr_target = get_batch_predictions_SWAG(model, X, target, scale, cov)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return arr_preds, arr_target


def construct_predict_dict(n_elements, n_samples, nb_classes, MFE=False):
    """Construct a dict to hold prediction results."""
    d_pred = {
        key: np.zeros((n_elements, n_samples, nb_classes)).astype(np.float32)
        for key in ["all"]
        + [f"PEAKMJD{offset}" for offset in du.OFFSETS_STR if not MFE]
        + [f"all_{OOD}" for OOD in du.OOD_TYPES]
    }

    d_pred["target"] = np.zeros((n_elements, n_samples)).astype(np.int64)
    d_pred["SNID"] = np.zeros((n_elements, n_samples)).astype(str)

    return d_pred


def flatten_to_dataframe(pred_dict):
    """Flatten the prediction dict and convert it to pandas.DataFrame."""
    d_series = {}
    (n_elements, n_samples, nb_classes) = pred_dict["all"].shape
    for (key, value) in pred_dict.items():
        value = value.reshape((n_elements * n_samples, -1))
        value_dim = value.shape[1]
        if value_dim == 1:
            d_series[key] = np.ravel(value)
        else:
            for i in range(value_dim):
                d_series[f"{key}_class{i}"] = value[:, i]

    return pd.DataFrame.from_dict(d_series)


def get_aggregated_dataframe(pred_df):
    med_pred = pred_df.groupby("SNID").median()
    med_pred.columns = [str(col) + "_median" for col in med_pred.columns]
    std_pred = pred_df.groupby("SNID").std()
    std_pred.columns = [str(col) + "_std" for col in std_pred.columns]
    df_agg = pd.merge(med_pred, std_pred, on="SNID")
    df_agg["SNID"] = df_agg.index
    df_agg["target"] = df_agg["target_median"]

    return df_agg


def get_full_accuracy(pred_df, settings, model):
    g_pred = pred_df.groupby("SNID").median()
    preds = g_pred[[f"all_class{i}" for i in range(settings.nb_classes)]].values
    preds = np.argmax(preds, 1)
    acc = (preds == g_pred.target.values).sum() / len(g_pred)

    lu.print_green(f"Full Accuracy {model}", acc)
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


def get_other_accuracy(pred_df, settings, mode):
    class_col = [f"all_class{i}" for i in range(settings.nb_classes)]
    tmp = pred_df[["SNID", "target"] + class_col].groupby("SNID").mean()
    preds = np.argmax(tmp[class_col].values, 1)
    acc = (preds == tmp.target.values).sum() / len(tmp)
    lu.print_green(f"Accuracy {mode}", acc)
    lu.print_green(
        "Balanced Accuracy",
        metrics.balanced_accuracy_score(tmp.target.values, preds),
    )

    for OOD in ["random", "reverse", "shuffle", "sin"]:
        class_col_ood = [f"all_{OOD}_class{i}" for i in range(settings.nb_classes)]
        entropy_ood = (
            -(pred_df[class_col_ood].values * np.log(pred_df[class_col_ood].values))
            .sum(1)
            .mean()
        )
        entropy = (
            -(pred_df[class_col].values * np.log(pred_df[class_col].values))
            .sum(1)
            .mean()
        )
        lu.print_green(f"Delta Entropy {OOD} {mode}", entropy_ood - entropy)

    print()
    print()


def get_predictions(settings, model_file=None):
    """Obtain predictions for a given RNN model specified by the
    ``settings`` argument or alternatively, by a model_file

    - Models are benchmarked on the test data set
    - Batch size can be controled to speed up predictions
    - For Bayesian models, multiple predictions are carried to
        obtain a distribution of predictions
    - Predictions are computed for full lightcurves, and around the peak light
    - Predictions are saved to a pickle file (for faster loading)

    Args:
        settings (ExperimentSettings): custom class to hold hyperparameters
        model_file (str): Path to saved model weights. Default: ``None``
    """

    settings.random_length = False
    settings.random_redshift = False

    if "vanilla" in settings.pytorch_model_name:
        settings.num_inference_samples = 1

    # set the default behaviour: get prediction use RNN model only
    no_rnn = 0
    no_swag = 1

    # set output file
    out_files = []

    # Load model files
    if model_file is None:
        dump_dir = f"{settings.models_dir}/{settings.pytorch_model_name}"
        model_file = f"{dump_dir}/{settings.pytorch_model_name}.pt"

        # Load SWAG model
        if settings.swa:
            no_swag = 0
            swa_model_file = f"{dump_dir}/{settings.pytorch_model_name}_swag.pt"
            if not os.path.exists(swa_model_file):
                warnings.warn("SWAG model does not exist.")
                no_swag = 1
    else:
        dump_dir = f"{settings.dump_dir}/models/{settings.pytorch_model_name}"
        os.makedirs(dump_dir, exist_ok=True)
        if model_file.endswith("_swag.pt"):
            swa_model_file = model_file
            no_rnn = 1
            no_swag = 0

    # Load the data
    list_data_test = tu.load_HDF5(settings, test=True)

    # Batching stuff together
    num_elem = len(list_data_test)
    num_batches = num_elem / min(num_elem, settings.batch_size)
    list_batches = np.array_split(np.arange(num_elem), num_batches)

    # Fetch SN info
    df_SNinfo = du.load_HDF5_SNinfo(settings).set_index("SNID")

    def _populate_predict_dict(_model, _pred_dict, _n_samples, _mode=None, **kwargs):
        """Get prediction results and store them in the given dict."""
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

                for iter_ in tqdm(range(_n_samples), ncols=100):

                    arr_preds, arr_target = dispatch_batch_predictions(
                        _model, packed, target_tensor, mode=_mode, **kwargs
                    )

                    # Rever sorting that occurs in get_batch_predictions
                    arr_preds = arr_preds[idxs_rev_sort]
                    arr_target = arr_target[idxs_rev_sort]

                    _pred_dict["all"][start_idx:end_idx, iter_] = arr_preds
                    _pred_dict["target"][start_idx:end_idx, iter_] = arr_target
                    _pred_dict["SNID"][start_idx:end_idx, iter_] = SNIDs

                if _mode != "MFE":
                    #############################
                    # Predictions around PEAKMJD
                    #############################
                    for offset in du.OFFSETS_VAL:
                        slice_idxs = [
                            find_idx(times[k], peak_MJDs[k] + offset)
                            for k in range(len(times))
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

                            for iter_ in tqdm(range(_n_samples), ncols=100):

                                arr_preds, arr_target = dispatch_batch_predictions(
                                    _model, packed, target_tensor, mode=_mode, **kwargs
                                )

                                # Rever sorting that occurs in get_batch_predictions
                                arr_preds = arr_preds[idxs_rev_sort]

                                suffix = str(offset) if offset != 0 else ""
                                suffix = f"+{suffix}" if offset > 0 else suffix
                                col = f"PEAKMJD{suffix}"

                                _pred_dict[col][start_idx + inb_idxs, iter_] = arr_preds
                                # For oob_idxs, no prediction can be made, fill with nan
                                _pred_dict[col][start_idx + oob_idxs, iter_] = np.nan

                #############################
                # OOD predictions
                #############################

                for OOD in ["random", "shuffle", "reverse", "sin"]:
                    packed, _, target_tensor, idxs_rev_sort = tu.get_data_batch(
                        list_data_test, batch_idxs, settings, OOD=OOD
                    )

                    for iter_ in tqdm(range(_n_samples), ncols=100):

                        arr_preds, arr_target = dispatch_batch_predictions(
                            _model, packed, target_tensor, mode=_mode, **kwargs
                        )
                        # Revert sorting that occurs in get_batch_predictions
                        arr_preds = arr_preds[idxs_rev_sort]
                        arr_target = arr_target[idxs_rev_sort]

                        _pred_dict[f"all_{OOD}"][start_idx:end_idx, iter_] = arr_preds

    if no_rnn == 0:
        lu.print_green("Processing prediction using RNN model")
        # Load RNN model
        rnn = tu.get_model(settings, len(settings.training_features))
        rnn_state = torch.load(model_file, map_location=lambda storage, loc: storage)
        rnn.load_state_dict(rnn_state)
        rnn.to(settings.device)
        rnn.eval()

        # Specify output file name
        prediction_file = f"{dump_dir}/PRED_{settings.pytorch_model_name}.pickle"

        out_files.append(prediction_file)

        # Prepare output dicts
        d_pred = construct_predict_dict(
            num_elem, settings.num_inference_samples, settings.nb_classes
        )
        d_pred_MFE = construct_predict_dict(num_elem, 1, settings.nb_classes, MFE=True)

        # Fill in the dict
        _populate_predict_dict(rnn, d_pred, settings.num_inference_samples)
        _populate_predict_dict(rnn, d_pred_MFE, 1, _mode="MFE")

        # Flatten all arrays and aggregate in dataframe
        df_pred = flatten_to_dataframe(d_pred)
        df_pred_MFE = flatten_to_dataframe(d_pred_MFE)

        # Save predictions
        df_pred.to_pickle(prediction_file, protocol=4)

        # Saving aggregated preds for bayesian models
        if settings.model == "variational" or settings.model == "bayesian":
            df_bayes = get_aggregated_dataframe(df_pred)
            bay_pred_file = prediction_file.replace(".pickle", "_aggregated.pickle")
            df_bayes.to_pickle(bay_pred_file, protocol=4)

        # Display accuracy
        get_full_accuracy(df_pred, settings, "RNN")
        get_other_accuracy(df_pred, settings, "MC")
        get_other_accuracy(df_pred_MFE, settings, "MFE")

        lu.print_green("Finished getting RNN predictions ")

    if no_swag == 0:
        lu.print_green("Processing prediction using SWA/SWAG model")

        # Check setting for cov
        if settings.swag_no_cov:
            swag_cov = False
            cov_str = "no_cov"
        else:
            swag_cov = True
            cov_str = "cov"

        # Load SWAG model
        swag_rnn = torch.load(swa_model_file)
        swag_rnn.to(settings.device)

        # Specify output file names
        prediction_file_swa = (
            f"{dump_dir}/PRED_{settings.pytorch_model_name}_swa.pickle"
        )
        out_files.append(prediction_file_swa)

        prediction_file_swag = f"{dump_dir}/PRED_{settings.pytorch_model_name}_scale_{settings.swag_scale}_{cov_str}_swag.pickle"
        out_files.append(prediction_file_swag)

        # Prepare output dicts
        d_pred_SWA = construct_predict_dict(num_elem, 1, settings.nb_classes)
        d_pred_SWAG = construct_predict_dict(
            num_elem, settings.swag_samples, settings.nb_classes
        )

        # Fill in the dict
        _populate_predict_dict(
            swag_rnn, d_pred_SWA, 1, _mode="SWAG", scale=0, cov=False
        )
        _populate_predict_dict(
            swag_rnn,
            d_pred_SWAG,
            settings.swag_samples,
            _mode="SWAG",
            scale=settings.swag_scale,
            cov=swag_cov,
        )

        # Flatten all arrays and aggregate in dataframe
        df_pred_SWA = flatten_to_dataframe(d_pred_SWA)
        df_pred_SWAG = flatten_to_dataframe(d_pred_SWAG)

        # Save predictions
        df_pred_SWA.to_pickle(prediction_file_swa, protocol=4)
        df_pred_SWAG.to_pickle(prediction_file_swag, protocol=4)

        # Saving aggregated preds for swag model
        df_bayes = get_aggregated_dataframe(df_pred_SWAG)
        bay_pred_file = prediction_file_swag.replace(".pickle", "_aggregated.pickle")
        df_bayes.to_pickle(bay_pred_file, protocol=4)

        # Display accuracy for SWA model
        get_full_accuracy(df_pred_SWA, settings, "SWA")
        get_other_accuracy(df_pred_SWA, settings, "SWA MC")

        # Display accuracy for SWAG model
        get_full_accuracy(df_pred_SWAG, settings, "SWAG")
        get_other_accuracy(df_pred_SWAG, settings, "SWAG MC")

        lu.print_green("Finished getting SWA/SWAG predictions ")

    return out_files


def get_predictions_for_speed_benchmark(settings):
    """Test RNN models inference speed

    - Models are benchmarked on the test data set
    - Batch size can be controled to speed up predictions
    - For Bayesian models, multiple predictions are carried to
        obtain a distribution of predictions
    - Results are saved to a .csv for future use

    Args:
        settings (ExperimentSettings): custom class to hold hyperparameters
    """

    settings.random_length = False

    if "vanilla" in settings.pytorch_model_name:
        settings.num_inference_samples = 1

    # Load the data
    list_data_test = tu.load_HDF5(settings, test=True)

    # Load RNN model
    rnn = tu.get_model(settings, len(settings.training_features))
    rnn.to(settings.device)
    rnn.eval()

    # Batching lightcurves
    num_elem = len(list_data_test)
    num_batches = num_elem / min(num_elem, settings.batch_size)
    list_batches = np.array_split(np.arange(num_elem), num_batches)

    start_time = time()

    # Loop over data and predict
    for batch_idxs in tqdm(list_batches, ncols=100):

        with torch.no_grad():

            #############################
            # Full lightcurve prediction
            #############################

            packed, _, target_tensor, _ = tu.get_data_batch(
                list_data_test, batch_idxs, settings
            )

            for iter_ in tqdm(range(settings.num_inference_samples), ncols=100):

                arr_preds, arr_target = get_batch_predictions(
                    rnn, packed, target_tensor
                )

    total_time = time() - start_time
    supernova_per_s = num_elem / total_time
    model_id = f"{settings.model}_{settings.batch_size}_{settings.device}"

    df = pd.DataFrame(
        data=np.array([supernova_per_s]).astype(np.float16), columns=["Supernova_per_s"]
    )
    df["model"] = settings.model
    df["batch_size"] = settings.batch_size
    df["device"] = settings.device
    df["id"] = model_id

    # Save results to csv
    results_file = Path(settings.stats_dir) / "rnn_speed.csv"
    try:
        df_all = pd.read_csv(results_file, index_col="id")
        df_all.loc[model_id] = df.set_index("id").loc[model_id]
        df_all.reset_index().to_csv(results_file, index=False)
    except Exception:
        df.to_csv(results_file, index=False)
