import re
import os
import h5py
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import matplotlib
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

    seq_len = X_tensor.shape[0]
    d_pred = {key: {"prob": []} for key in dict_rnn}

    # Loop over light curve time steps to obtain prediction for each time step
    for i in range(1, seq_len + 1):
        # Slice along the time step dimension
        X_slice = X_tensor[:i]

        # Apply rnn to obtain prediction
        for model_type, rnn in dict_rnn.items():

            n = settings.num_inference_samples if "variational" in model_type else 1
            new_size = (X_slice.size(0), n, X_slice.size(2))

            if "bayesian" in model_type:

                # Loop over num samples to obtain predictions
                list_out = [
                    rnn(X_slice.expand(new_size))
                    for i in range(settings.num_inference_samples)
                ]
                out = torch.cat(list_out, dim=0)
                # Apply softmax to obtain a proba
                pred_proba = nn.functional.softmax(out, dim=-1).data.cpu().numpy()
            else:
                out = rnn(X_slice.expand(new_size))
                # Apply softmax to obtain a proba
                pred_proba = nn.functional.softmax(out, dim=-1).data.cpu().numpy()

            # Add to buffer list
            d_pred[model_type]["prob"].append(pred_proba)

    # Stack
    for key in dict_rnn.keys():
        arr_proba = np.stack(d_pred[key]["prob"], axis=0)
        d_pred[key]["prob"] = arr_proba  # arr_prob is (T, num_samples, 2)
        d_pred[key]["median"] = np.median(arr_proba, axis=1)
        d_pred[key]["perc_16"] = np.percentile(arr_proba, 16, axis=1)
        d_pred[key]["perc_84"] = np.percentile(arr_proba, 84, axis=1)
        d_pred[key]["perc_2"] = np.percentile(arr_proba, 2, axis=1)
        d_pred[key]["perc_98"] = np.percentile(arr_proba, 98, axis=1)

    return d_pred, X_tensor.squeeze().detach().cpu().numpy()


def plot_predictions(
    settings, d_plot, SNID, redshift, peak_MJD, target, arr_time, d_pred, OOD
):

    plt.figure()
    gs = gridspec.GridSpec(2, 1)
    # Plot the lightcurve
    ax = plt.subplot(gs[0])
    for flt in d_plot.keys():
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
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    ax.set_ylabel("FLUXCAL")
    ylim = ax.get_ylim()

    SNtype = du.sntype_decoded(target, settings)
    if settings.data_testing:
        ax.set_title(f"ID: {SNID}")
    elif OOD is not None:
        ax.set_title(f"OOD {OOD} ID: {SNID}")
    else:
        ax.set_title(SNtype + f" (ID: {SNID}, redshift: {redshift:.3g})")
        # Add PEAKMJD
        if OOD is None and not settings.data_testing and arr_time.min()<peak_MJD and peak_MJD>arr_time.max():
            ax.plot([peak_MJD, peak_MJD], ylim, "k--", label="Peak MJD")

    # Plot the classifications
    ax = plt.subplot(gs[1])
    ax.set_ylim(0, 1)

    for idx, key in enumerate(d_pred.keys()):

        for class_prob in range(settings.nb_classes):
            color = ALL_COLORS[class_prob + idx * settings.nb_classes]
            linestyle = LINE_STYLE[class_prob]
            label = du.sntype_decoded(class_prob, settings)

            if len(d_pred) > 1:
                label += f" {key}"

            ax.plot(
                arr_time,
                d_pred[key]["median"][:, class_prob],
                color=color,
                linestyle=linestyle,
                label=label,
            )
            ax.fill_between(
                arr_time,
                d_pred[key]["perc_16"][:, class_prob],
                d_pred[key]["perc_84"][:, class_prob],
                color=color,
                alpha=0.4,
            )
            ax.fill_between(
                arr_time,
                d_pred[key]["perc_2"][:, class_prob],
                d_pred[key]["perc_98"][:, class_prob],
                color=color,
                alpha=0.2,
            )

    ax.set_xlabel("Time (MJD)")
    ax.set_ylabel("classification probability")
    # Add PEAKMJD
    if OOD is None and not settings.data_testing and arr_time.min()<peak_MJD and peak_MJD>arr_time.max():
        ax.plot([peak_MJD, peak_MJD], [0, 1], "k--", label="Peak MJD")
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    prefix = f"OOD_{OOD}_" if OOD is not None else ""

    if len(settings.model_files) > 1:
        fig_path = f"{settings.figures_dir}/{prefix}multi_model_early_prediction"
        fig_name = f"{prefix}multi_model_{SNID}.png"
    elif len([settings.model_files]) == 1:
        parent_dir = Path(settings.model_files[0]).parent.name
        fig_path = f"{settings.lightcurves_dir}/{parent_dir}/{prefix}early_prediction"
        fig_name = f"{parent_dir}_{prefix}class_pred_with_lc_{SNID}.png"
    else:
        fig_path = (
            f"{settings.lightcurves_dir}/{settings.pytorch_model_name}/{prefix}early_prediction"
        )
        fig_name = (
            f"{settings.pytorch_model_name}_{prefix}class_pred_with_lc_{SNID}.png"
        )
    Path(fig_path).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(Path(fig_path) / fig_name)
    plt.clf()
    plt.close()


def make_early_prediction(settings, nb_lcs=1, do_gifs=False):
    """Load model corresponding to settings
    or (if specified) load a list of models.

    - Show evolution of classification for one time-step, then 2, up to all of the lightcurve
    - For Bayesian models, show uncertainty in the prediction
    - Figures are save in the figures repository

    Args:
        settings: (ExperimentSettings) custom class to hold hyperparameters
        int (nb_lcs): number of light-curves to plot, default is 1
    """
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
        # check if the model files are there
        tmp_not_found = [m for m in settings.model_files if not os.path.exists(m)]
        if len(tmp_not_found) > 0:
            print(lu.str_to_redstr(f"Files not found {tmp_not_found}"))
            tmp_model_files = [m for m in settings.model_files if os.path.exists(m)]
            settings.model_files = tmp_model_files

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

    # load SN info
    SNinfo_df = du.load_HDF5_SNinfo(settings)

    # Loop over data to plot prediction
    # randomly select lcs to plot
    list_entries = np.random.randint(0,high=len(list_data_test),size=nb_lcs)
    subset_to_plot = [list_data_test[i] for i in list_entries]
    for X, target, SNID, _, X_ori in tqdm(subset_to_plot, ncols=100):
        
        try:
            redshift = SNinfo_df[SNinfo_df["SNID"] == SNID]["SIM_REDSHIFT_CMB"].values[0]
            peak_MJD = SNinfo_df[SNinfo_df["SNID"] == SNID]["PEAKMJDNORM"].values[0]
        except Exception:
            redshift = 0.0
            peak_MJD = 0.0

        # Prepare plotting data in a dict
        d_plot = {
            flt: {"FLUXCAL": [], "FLUXCALERR": [], "MJD": []}
            for flt in settings.list_filters
        }
        for OOD in [None] + du.OOD_TYPES:
            with torch.no_grad():
                d_pred, X_normed = get_predictions(
                    settings, dict_rnn, X, target, OOD=OOD
                )
            # X here has been normalized. We unnormalize X
            X_unnormed = tu.unnormalize_arr(X_normed, settings)
            # Check we do recover X_ori when OOD is None
            if OOD is None and settings.norm != 'cosmo':
                #check if normalization converges
                # using clipping in case of min<model_min
                X_clip = X_ori.copy()
                X_clip = np.clip(X_clip[:,settings.idx_features_to_normalize], settings.arr_norm[:, 0], np.inf)
                X_ori[:,settings.idx_features_to_normalize] = X_clip
                assert np.all(np.all(np.isclose(np.ravel(X_ori), np.ravel(X_unnormed), atol=1e-1)))

            # TODO: IMPROVE
            df_temp = pd.DataFrame(data=X_unnormed, columns=features)
            arr_time = np.cumsum(df_temp.delta_time.values)
            df_temp['time'] = arr_time
            for flt in settings.list_filters:
                non_zero = np.where(
                    ~np.isclose(df_temp[f"FLUXCAL_{flt}"].values, 0, atol=1E-2)
                )[0]
                d_plot[flt]["FLUXCAL"] = df_temp[f"FLUXCAL_{flt}"].values[non_zero]
                d_plot[flt]["FLUXCALERR"] = df_temp[f"FLUXCALERR_{flt}"].values[
                    non_zero
                ]
                d_plot[flt]["MJD"] = arr_time[non_zero]
            plot_predictions(
                settings,
                d_plot,
                SNID,
                redshift,
                peak_MJD,
                target,
                arr_time,
                d_pred,
                OOD,
            )

            # use to create GIFs
            if not OOD:
                if do_gifs:
                    plot_gif(settings,
                             df_temp,
                             SNID,
                             redshift,
                             peak_MJD,
                             target,
                             arr_time,
                             d_pred)
    lu.print_green("Finished plotting lightcurves and predictions ")


def plot_gif(settings, df_plot, SNID, redshift, peak_MJD, target, arr_time, d_pred
             ):
    """ Create GIFs for classification
    """
    import imageio

    def plot_image_for_gif(fig,gs, df_plot,d_pred,time,SNtype):

        # Plot the lightcurve
        ax = plt.subplot(gs[0])
        # Used to keep the limits constant
        flux_max = max(df_plot[[k for k in df_plot.keys() if 'FLUXCAL_' in k]].max())
        flux_min = min(df_plot[[k for k in df_plot.keys() if 'FLUXCAL_' in k]].min())
        ax.set_ylim(flux_min - 5, flux_max + 5)
        ax.set_xlim(-.5, max(df_plot['time']) + 2)

        # slice for gif
        df_sel = df_plot[df_plot['time'] <= time]
        for flt in settings.list_filters:
            ax.errorbar(df_sel['time'], df_sel[f"FLUXCAL_{flt}"],
                        yerr=df_sel[f"FLUXCALERR_{flt}"],
                        fmt="o",
                        label=f"Filter {flt}",
                        color=FILTER_COLORS[flt])

        ax.set(xlabel='', ylabel='flux',
               title=f"{SNtype} (ID: {SNID}, redshift: {redshift:.3g})")

        # Plot the classifications
        ax = plt.subplot(gs[1])
        ax.clear()
        ax.set_ylim(0, 1)
        ax.set_xlim(-.5, max(df_plot['time']) + 2)
        # select classification of same length
        for idx, key in enumerate(d_pred.keys()):

            for class_prob in range(settings.nb_classes):
                color = ALL_COLORS[class_prob + idx * settings.nb_classes]
                linestyle = LINE_STYLE[class_prob]
                label = du.sntype_decoded(class_prob, settings)

                if len(d_pred) > 1:
                    label += f" {key}"

                ax.plot(
                    arr_time[:len(df_sel)],
                    d_pred[key]["median"][:, class_prob][:len(df_sel)],
                    color=color,
                    linestyle=linestyle,
                    label=label,
                )
                ax.fill_between(
                    arr_time[:len(df_sel)],
                    d_pred[key]["perc_16"][:, class_prob][:len(df_sel)],
                    d_pred[key]["perc_84"][:, class_prob][:len(df_sel)],
                    color=color,
                    alpha=0.4,
                )
                ax.fill_between(
                    arr_time[:len(df_sel)],
                    d_pred[key]["perc_2"][:, class_prob][:len(df_sel)],
                    d_pred[key]["perc_98"][:, class_prob][:len(df_sel)],
                    color=color,
                    alpha=0.2,
                )
        ax.set_ylabel("classification probability")
        ax.set_xlabel("time")

        # Used to return the plot as an image rray
        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        from PIL import Image
        import PIL.ImageOps as pops
        im = Image.fromarray(image)
        # make background transparent
        # im = im.convert('RGB')
        # im =pops.invert(im)
        image = im

        return image

    kwargs_write = {'fps':1.0, 'quantizer':'nq'}
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1)
    SNtype = du.sntype_decoded(target, settings)

    fig_path = (
        f"{settings.lightcurves_dir}/{settings.pytorch_model_name}/gif"
    )
    fig_name = (
        f"{settings.pytorch_model_name}_class_pred_with_lc_{SNID}.gif"
    )
    Path(fig_path).mkdir(parents=True, exist_ok=True)
    arr_images = [plot_image_for_gif(fig, gs,df_plot,d_pred,time,SNtype) for time in arr_time]
    arr_images[0].save(str(Path(fig_path) / fig_name), save_all=True, append_images=arr_images, loop=5, duration=200)
