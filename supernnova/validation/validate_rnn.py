import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from pathlib import Path

import torch
import torch.nn as nn

from ..utils import data_utils as du
from ..utils import training_utils as tu
from ..utils import logging_utils as lu


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


def get_batch_predictions(rnn, X, target_tuple):
    """Utility to obtain predictions for a given batch

    Args:
        rnn (torch.nn): The RNN model
        X (torch.Tensor): The batch on which to carry out predictions
        target_tuple (torch.longTensor): The true class and peak of each element in the batch

    Returns:
        Tuple containing
        - arr_class_preds (np.array): class predictions
        - arr_class_target (np.array): actual class targets
        - arr_peak_preds (np.array): peak predictions
        - arr_peak_target (np.array): actual peak targets

    """

    outclass, outpeak, maskpeak = rnn.forward(X)
    arr_class_preds = nn.functional.softmax(
        outclass, dim=-1).data.cpu().numpy()
    arr_class_target = target_tuple[0].detach().cpu().numpy()

    maskpeak = maskpeak.to(outpeak.device)
    arr_peak_preds = outpeak*maskpeak

    target_peak = target_tuple[1].squeeze(-1).to(maskpeak.device)
    arr_peak_target = (target_peak*maskpeak).detach().cpu().numpy()

    return (arr_class_preds, arr_peak_preds), (arr_class_target, arr_peak_target)


def get_batch_predictions_MFE(rnn, X, target_tuple):
    """Utility to obtain predictions for a given batch

    Args:
        rnn (torch.nn): The RNN model
        X (torch.Tensor): The batch on which to carry out predictions
        target (torch.longTensor): The true class of each element in the batch

    Returns:
        Tuple containing

        - arr_class_preds (np.array): class predictions
        - arr_class_target (np.array): actual class targets
        - arr_peak_preds (np.array): peak predictions
        - arr_peak_target (np.array): actual peak targets

    """

    outclass, outpeak, maskpeak = rnn.forward(X, mean_field_inference=True)
    arr_class_preds = nn.functional.softmax(
        outclass, dim=-1).data.cpu().numpy()
    arr_class_target = target_tuple[0].detach().cpu().numpy()

    maskpeak = maskpeak.to(outpeak.device)
    arr_peak_preds = outpeak*maskpeak

    target_peak = target_tuple[1].squeeze(-1).to(maskpeak.device)
    arr_peak_target = (target_peak*maskpeak).detach().cpu().numpy()

    return (arr_class_preds, arr_peak_preds), (arr_class_target, arr_peak_target)


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

    # Load RNN model
    rnn = tu.get_model(settings, len(settings.training_features))
    if model_file is None:
        dump_dir = f"{settings.models_dir}/{settings.pytorch_model_name}"
        model_file = f"{dump_dir}/{settings.pytorch_model_name}.pt"
    else:
        dump_dir = f"{settings.dump_dir}/models/{settings.pytorch_model_name}"
        os.makedirs(dump_dir, exist_ok=True)

    if settings.override_source_data is not None:
        settings.source_data = settings.override_source_data
        settings.set_pytorch_model_name()

    prediction_file = (
        f"{dump_dir}/PRED_{settings.pytorch_model_name}.pickle"
    )

    rnn_state = torch.load(
        model_file, map_location=lambda storage, loc: storage)
    rnn.load_state_dict(rnn_state)
    rnn.to(settings.device)
    rnn.eval()

    # Load the data
    list_data_test = tu.load_HDF5(settings, test=True)

    # Batching stuff together
    num_elem = len(list_data_test)
    factor = (num_elem // 4) if settings.source_data == "photometry" else num_elem
    num_batches = num_elem / min(factor, 100000)
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
        + [f"all_{OOD}" for OOD in du.OOD_TYPES]
    }
    for key in ["target", "SNID"]:
        d_pred[key] = np.zeros((num_elem, settings.num_inference_samples)).astype(
            np.int64
        )
    for key in ["all_peak", "target_peak", "PEAKMJD-2_peak",
                "PEAKMJD-1_peak",
                "PEAKMJD_peak",
                "PEAKMJD+1_peak",
                "PEAKMJD+2_peak"]:
        d_pred[key] = np.zeros((num_elem, settings.num_inference_samples)).astype(
            np.float16
        )

    d_pred_MFE = {
        key: np.zeros((num_elem, 1, settings.nb_classes)).astype(np.float32)
        for key in ["all"] + [f"all_{OOD}" for OOD in du.OOD_TYPES]
    }
    for key in ["target", "SNID", "all_peak", "target_peak"]:
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
        max_lengths = [len(times[i]) for i in range(len(times))]

        with torch.no_grad():

            #############################
            # Full lightcurve prediction
            #############################

            packed, _, target_tensor_tuple, idxs_rev_sort = tu.get_data_batch(
                list_data_test, batch_idxs, settings
            )

            for iter_ in tqdm(range(settings.num_inference_samples), ncols=100):

                arr_preds_tuple, arr_target_tuple = get_batch_predictions(
                    rnn, packed, target_tensor_tuple
                )

                # split tuples
                arr_class_preds, arr_peak_preds = arr_preds_tuple
                arr_class_target, arr_peak_target = arr_target_tuple

                # Rever sorting that occurs in get_batch_predictions
                arr_class_preds = arr_class_preds[idxs_rev_sort]
                arr_class_target = arr_class_target[idxs_rev_sort]
                arr_peak_preds = arr_peak_preds[:, idxs_rev_sort]
                arr_peak_target = arr_peak_target[:, idxs_rev_sort]

                d_pred["all"][start_idx:end_idx, iter_] = arr_class_preds
                d_pred["target"][start_idx:end_idx, iter_] = arr_class_target
                d_pred["SNID"][start_idx:end_idx, iter_] = SNIDs

                # taking last peak prediction
                arr_last_time_step = np.array(
                    [times[i][-1] for i in range(len(times))])
                # to be improved
                for idx in range(len(max_lengths)):
                    last_time_idx = max_lengths[idx]
                    d_pred["all_peak"][start_idx + idx, iter_] = arr_peak_preds[last_time_idx -
                                                                                1, idx].data.cpu().numpy() + arr_last_time_step[idx]
                    d_pred["target_peak"][start_idx + idx,
                                          iter_] = arr_peak_target[last_time_idx-1, idx] + arr_last_time_step[idx]

            # MFE
            arr_preds_tuple, arr_target_tuple = get_batch_predictions_MFE(
                rnn, packed, target_tensor_tuple
            )

            # split tuples
            arr_class_preds, arr_peak_preds = arr_preds_tuple
            arr_class_target, arr_peak_target = arr_target_tuple

            # Rever sorting that occurs in get_batch_predictions
            arr_class_preds = arr_class_preds[idxs_rev_sort]
            arr_class_target = arr_class_target[idxs_rev_sort]
            arr_peak_preds = arr_peak_preds[:, idxs_rev_sort]
            arr_peak_target = arr_peak_target[:, idxs_rev_sort]

            d_pred_MFE["all"][start_idx:end_idx, 0] = arr_class_preds
            d_pred_MFE["target"][start_idx:end_idx, 0] = arr_class_target
            d_pred_MFE["SNID"][start_idx:end_idx, 0] = SNIDs

            # taking last peak prediction
            arr_last_time_step = np.array(
                [times[i][-1] for i in range(len(times))])
            # to be improved
            for idx in range(len(max_lengths)):
                last_time_idx = max_lengths[idx]
                d_pred_MFE["all_peak"][start_idx + idx, iter_] = arr_peak_preds[last_time_idx -
                                                                                1, idx].data.cpu().numpy() + arr_last_time_step[idx]
                d_pred_MFE["target_peak"][start_idx + idx,
                                          iter_] = arr_peak_target[last_time_idx-1, idx]

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
                    packed, _, target_tensor_tuple, idxs_rev_sort = tu.get_data_batch(
                        list_data_test, offset_batch_idxs, settings, max_lengths=max_lengths
                    )

                    for iter_ in tqdm(range(settings.num_inference_samples), ncols=100):

                        arr_preds_tuple, arr_target_tuple = get_batch_predictions(
                            rnn, packed, target_tensor_tuple
                        )

                        # split tuples
                        arr_class_preds, arr_peak_preds = arr_preds_tuple
                        arr_class_target, arr_peak_target = arr_target_tuple

                        # Reverse sorting that occurs in get_batch_predictions
                        arr_class_preds = arr_class_preds[idxs_rev_sort]
                        arr_peak_preds = arr_peak_preds[:, idxs_rev_sort]

                        suffix = str(offset) if offset != 0 else ""
                        suffix = f"+{suffix}" if offset > 0 else suffix
                        col = f"PEAKMJD{suffix}"

                        d_pred[col][start_idx + inb_idxs,
                                    iter_] = arr_class_preds
                        # For oob_idxs, no prediction can be made, fill with nan
                        d_pred[col][start_idx + oob_idxs, iter_] = np.nan

                        # taking last peak prediction
                        # easier here since the lengths have been cut already
                        # to be improved
                        for i, idx in enumerate(inb_idxs):
                            time_idx = max_lengths[i]-1
                            last_time = times[idx][time_idx]
                            pred = arr_peak_preds[time_idx, i].data.cpu().numpy() + last_time
                            d_pred[f"{col}_peak"][start_idx + idx, iter_] = pred
                        d_pred[f"{col}_peak"][start_idx + oob_idxs, iter_] = np.nan

            #############################
            # OOD predictions
            #############################

            for OOD in ["random", "shuffle", "reverse", "sin"]:
                packed, _, target_tensor_tuple, idxs_rev_sort = tu.get_data_batch(
                    list_data_test, batch_idxs, settings, OOD=OOD
                )

                for iter_ in tqdm(range(settings.num_inference_samples), ncols=100):

                    arr_preds_tuple, arr_target_tuple = get_batch_predictions(
                        rnn, packed, target_tensor_tuple
                    )

                    # split tuples
                    arr_class_preds, arr_peak_preds = arr_preds_tuple
                    arr_class_target, arr_peak_target = arr_target_tuple

                    # Revert sorting that occurs in get_batch_predictions
                    arr_class_preds = arr_class_preds[idxs_rev_sort]
                    arr_class_target = arr_class_target[idxs_rev_sort]

                    d_pred[f"all_{OOD}"][start_idx:end_idx, iter_] = arr_class_preds

                arr_preds_tuple, arr_target_tuple = get_batch_predictions_MFE(
                    rnn, packed, target_tensor_tuple
                )

                # split tuples
                arr_class_preds, arr_peak_preds = arr_preds_tuple
                arr_class_target, arr_peak_target = arr_target_tuple

                # Revert sorting that occurs in get_batch_predictions
                arr_class_preds = arr_class_preds[idxs_rev_sort]

                d_pred_MFE[f"all_{OOD}"][start_idx:end_idx, 0] = arr_class_preds

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
    if settings.model == 'variational' or settings.model == 'bayesian':
        med_pred = df_pred.groupby("SNID").median()
        med_pred.columns = [str(col) + '_median' for col in med_pred.columns]
        std_pred = df_pred.groupby("SNID").std()
        std_pred.columns = [str(col) + '_std' for col in std_pred.columns]
        df_bayes = pd.merge(med_pred, std_pred, on="SNID")
        df_bayes["SNID"] = df_bayes.index
        df_bayes["target"] = df_bayes["target_median"]
        bay_pred_file = prediction_file.replace(
            ".pickle", "_aggregated.pickle")
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
            -(df_pred[class_col_ood].values *
              np.log(df_pred[class_col_ood].values))
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
            -(df_pred_MFE[class_col].values *
              np.log(df_pred_MFE[class_col].values))
            .sum(1)
            .mean()
        )
        lu.print_green(f"Delta Entropy {OOD} MFE", entropy_ood - entropy)

    lu.print_green("Finished getting predictions ")

    return prediction_file


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

            packed, _, target_tensor_tuple, _ = tu.get_data_batch(
                list_data_test, batch_idxs, settings
            )

            for iter_ in tqdm(range(settings.num_inference_samples), ncols=100):

                arr_preds_tuple, arr_target_tuple = get_batch_predictions(
                    rnn, packed, target_tensor_tuple
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
