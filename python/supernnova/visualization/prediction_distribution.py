import re
import os
import h5py
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ..utils import data_utils as du
from ..utils import logging_utils as lu
from ..utils import training_utils as tu
from ..utils.visualization_utils import FILTER_COLORS, ALL_COLORS, LINE_STYLE

plt.switch_backend("agg")


def get_predictions(settings, dict_rnn, X, target, OOD=None):

    list_data = [(X.copy(), target)]

    _, X_tensor, *_ = tu.get_data_batch(list_data, [0], settings, OOD=OOD)

    if settings.use_cuda:
        X_tensor.cuda()

    d_pred = {key: {"prob": []} for key in dict_rnn}

    # Apply rnn to obtain prediction
    for model_type, rnn in dict_rnn.items():

        n = settings.num_inference_samples if "variational" in model_type else 1
        new_size = (X_tensor.size(0), n, X_tensor.size(2))

        if "bayesian" in model_type:

            # Loop over num samples to obtain predictions
            list_out = [
                rnn(X_tensor.expand(new_size))
                for i in range(settings.num_inference_samples)
            ]
            out = torch.cat(list_out, dim=0)
            # Apply softmax to obtain a proba
            pred_proba = nn.functional.softmax(out, dim=-1).data.cpu().numpy()
        else:
            out = rnn(X_tensor.expand(new_size))
            # Apply softmax to obtain a proba
            pred_proba = nn.functional.softmax(out, dim=-1).data.cpu().numpy()

        # Add to buffer list
        d_pred[model_type]["prob"].append(pred_proba)

    # Stack
    for key in dict_rnn.keys():
        arr_proba = np.stack(d_pred[key]["prob"], axis=0)
        d_pred[key]["prob"] = arr_proba  # arr_prob is (T, num_samples, 2)

    return d_pred, X_tensor.squeeze().detach().cpu().numpy()


def plot_distributions(settings, list_d_plot):

    plt.figure(figsize=(20, 30))
    gs = gridspec.GridSpec(8, 2, hspace=0.3, wspace=0.1)

    for i in range(len(list_d_plot)):

        d_plot = list_d_plot[i]
        SNID = d_plot["SNID"]
        OOD = d_plot["OOD"]
        target = d_plot["target"]
        redshift = d_plot["redshift"]
        peak_MJD = d_plot["peak_MJD"]
        d_pred = d_plot["d_pred"]

        # Plot the lightcurve
        ax = plt.subplot(gs[2 * i])
        for flt in settings.list_filters:
            flt_time = d_plot[flt]["MJD"]
            # Only plot a time series if it's non empty
            if len(flt_time) > 0:
                flux = d_plot[flt]["FLUXCAL"]
                fluxerr = d_plot[flt]["FLUXCALERR"]
                ax.errorbar(
                    flt_time,
                    flux,
                    yerr=fluxerr,
                    fmt="o",
                    label=f"Filter {flt}",
                    color=FILTER_COLORS[flt],
                )

        ax.set_ylabel("FLUXCAL", fontsize=24)
        if i == 7:
            ax.set_xlabel("days", fontsize=24)
        ylim = ax.get_ylim()
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)

        SNtype = du.sntype_decoded(target, settings)
        if OOD is not None:
            ax.set_title(f"OOD {OOD} ID: {SNID}", fontsize=24)
        else:
            ax.set_title(
                SNtype + f" (ID: {SNID}, redshift: {redshift:.3g})", fontsize=24
            )
            # Add PEAKMJD
            ax.plot([peak_MJD, peak_MJD], ylim, "k--", label="Peak MJD")

        # Plot the classifications
        ax = plt.subplot(gs[2 * i + 1])
        ax.set_xlim(-0.1, 1.1)

        for idx, key in enumerate(d_pred.keys()):

            arr_prob = np.squeeze(d_pred[key]["prob"])
            all_probs = np.ravel(arr_prob)
            _, bin_edges = np.histogram(all_probs, bins=25)

            for class_prob in range(settings.nb_classes):
                color = ALL_COLORS[class_prob + idx * settings.nb_classes]
                label = du.sntype_decoded(class_prob, settings)
                linestyle = LINE_STYLE[class_prob]

                if len(d_pred) > 1:
                    label += f" {key}"

                ax.hist(
                    arr_prob[:, class_prob],
                    color=color,
                    histtype="step",
                    linestyle=linestyle,
                    linewidth=2,
                    label=label,
                    bins=bin_edges,
                )
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)

        ax.set_yscale("log")

        if i == len(list_d_plot) - 1:
            ax.legend(
                bbox_to_anchor=(-2.8, 5.8),
                loc=2,
                borderaxespad=0.0,
                fontsize=16,
                ncol=(settings.nb_classes // 2) + 1,
                framealpha=0,
            )
        if 2 * i + 1 == 15:
            ax.set_xlabel("classification probability", fontsize=24)

    plt.subplots_adjust(
        left=0.08, right=0.99, bottom=0.03, top=0.98, wspace=0.0, hspace=0.02
    )

    if len([settings.model_files]) == 1:
        parent_dir = Path(settings.model_files[0]).parent.name
        fig_path = f"{settings.lightcurves_dir}/{parent_dir}/prediction_distribution"
        fig_name = f"{parent_dir}.png"
    else:
        fig_path = (
            f"{settings.lightcurves_dir}/{settings.pytorch_model_name}/prediction_distribution"
        )
        fig_name = f"{settings.pytorch_model_name}.png"
    Path(fig_path).mkdir(parents=True, exist_ok=True)
    # plt.tight_layout()
    plt.savefig(Path(fig_path) / fig_name)
    plt.clf()
    plt.close()


def plot_prediction_distribution(settings):
    """Load model corresponding to settings
    or (if specified) load a list of models.

    Args:
        settings: (ExperimentSettings) custom class to hold hyperparameters
        int (nb_lcs): number of light-curves to plot, default is 1
    """

    # No plot for ternary tasks
    if settings.nb_classes == 3:
        return

    settings.random_length = False
    settings.random_redshift = False

    # Load the test data
    list_data_test = tu.load_HDF5(settings, test=True)

    # Load features list
    file_name = f"{settings.processed_dir}/database.h5"
    with h5py.File(file_name, "r") as hf:
        features = hf["features"][settings.idx_features]

    # Load RNN model
    dict_rnn = {}
    if settings.model_files is None:
        settings.model_files = [f"{settings.rnn_dir}/{settings.pytorch_model_name}.pt"]
    else:
        assert (
            len(settings.model_files) == 1
        ), "Only one model file allowed at a time for these plots"

    # Check that the settings match the model file
    base_files = [Path(f).name for f in settings.model_files]

    classes = [int(re.search(r"(?<=CLF\_)\d+(?=\_)", f).group()) for f in base_files]
    redshifts = [re.search(r"(?<=R\_)[A-Za-z]+(?=\_)", f).group() for f in base_files]

    nb_classes, redshift = classes[0], redshifts[0]
    assert settings.nb_classes == nb_classes, lu.str_to_redstr(
        "Incompatible nb_classes between CLI and model files"
    )
    assert str(settings.redshift) == redshift, lu.str_to_redstr(
        "Incompatible redshift between CLI and model files"
    )

    for model_file in settings.model_files:
        if "variational" in model_file:
            settings.model = "variational"
        if "vanilla" in model_file:
            settings.model = "vanilla"
        if "bayesian" in model_file:
            settings.model = "bayesian"
        rnn = tu.get_model(settings, len(settings.training_features))
        rnn_state = torch.load(model_file, map_location=lambda storage, loc: storage)
        rnn.load_state_dict(rnn_state)
        rnn.to(settings.device)
        rnn.eval()
        name = (
            f"{settings.model} photometry"
            if "photometry" in model_file
            else f"{settings.model} salt"
        )
        dict_rnn[name] = rnn

    # lOad SN info
    SNinfo_df = du.load_HDF5_SNinfo(settings)

    targets = np.array([o[1] for o in list_data_test])

    if settings.nb_classes == 2:
        # 2 lightcurves of each type
        idxs_keep = (
            np.where(targets == 0)[0][:2].tolist()
            + np.where(targets == 1)[0][:2].tolist()
        )
    elif settings.nb_classes == 7:
        # Ia, Ib, Ic, IIp
        idxs_keep = [
            np.where(targets == list(settings.sntypes.values()).index(i))[0][0]
            for i in ["Ia", "Ib", "Ic", "IIP"]
        ]

    # Carry out 8 plots: 4 real light curves + 4 OOD
    list_OOD_types = ["random", "sin", "reverse", "shuffle"]
    list_data_test = [(list_data_test[i], None) for i in idxs_keep]
    list_data_test += [(o[0], ood) for (o, ood) in zip(list_data_test, list_OOD_types)]

    list_d_plot = []

    # Loop over data to plot prediction
    for ((X, target, SNID, _, X_ori), OOD) in tqdm(list_data_test, ncols=100):

        redshift = SNinfo_df[SNinfo_df["SNID"] == SNID]["SIM_REDSHIFT_CMB"].values[0]
        peak_MJD = SNinfo_df[SNinfo_df["SNID"] == SNID]["PEAKMJDNORM"].values[0]

        # Prepare plotting data in a dict
        d_plot = {
            flt: {"FLUXCAL": [], "FLUXCALERR": [], "MJD": []}
            for flt in settings.list_filters
        }

        with torch.no_grad():
            d_pred, X_normed = get_predictions(settings, dict_rnn, X, target, OOD=OOD)
        # X here has been normalized. We unnormalize X
        X_unnormed = tu.unnormalize_arr(X_normed, settings)
        # Check we do recover X_ori when OOD is None
        if OOD is None:
            assert np.all(np.isclose(np.ravel(X_ori), np.ravel(X_unnormed), atol=1e-2))

        # TODO: IMPROVE
        df_temp = pd.DataFrame(data=X_unnormed, columns=features)
        arr_time = np.cumsum(df_temp.delta_time.values)
        for flt in settings.list_filters:
            non_zero = np.where(
                ~np.isclose(df_temp[f"FLUXCAL_{flt}"].values, 0, atol=1E-2)
            )[0]
            d_plot[flt]["FLUXCAL"] = df_temp[f"FLUXCAL_{flt}"].values[non_zero]
            d_plot[flt]["FLUXCALERR"] = df_temp[f"FLUXCALERR_{flt}"].values[non_zero]
            d_plot[flt]["MJD"] = arr_time[non_zero]

        d_plot["redshift"] = redshift
        d_plot["peak_MJD"] = peak_MJD
        d_plot["SNID"] = SNID
        d_plot["OOD"] = OOD
        d_plot["target"] = target
        d_plot["d_pred"] = d_pred

        list_d_plot.append(d_plot)

    plot_distributions(settings, list_d_plot)
    lu.print_green("Finished plotting lightcurves and predictions ")
