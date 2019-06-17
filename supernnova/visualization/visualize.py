import os
from pathlib import Path
import h5py
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.switch_backend("agg")
import matplotlib.gridspec as gridspec


def plot_lightcurves(df, SNIDs, settings):
    """Utility for gridspec of lightcruves

    Args:
        df (pandas.DataFrame): dataframe holding the data
        SNIDs (np.array or list): holds lightcurve ids
        settings (ExperimentSettings): controls experiment hyperparameters
    """

    plt.figure(figsize=(20, 10))
    plt.suptitle("Sample of SN Ia light curves")
    gs = gridspec.GridSpec(4, 4, hspace=0.4)

    for i in range(16):
        ax = plt.subplot(gs[i])

        SNID = SNIDs[i]
        df_temp = df.loc[SNID]

        # Prepare plotting data in a dict
        d = {flt: {"FLUXCAL": [], "FLUXCALERR": [], "MJD": []} for flt in settings.list_filters}

        current_time = 0
        for idx in range(len(df_temp)):
            flt = df_temp.FLT.values[idx]
            d[flt]["FLUXCAL"].append(df_temp.FLUXCAL.values[idx])
            d[flt]["FLUXCALERR"].append(df_temp.FLUXCALERR.values[idx])
            current_time += df_temp.delta_time.values[idx]
            d[flt]["MJD"].append(current_time)

        for flt in d.keys():
            time = d[flt]["MJD"]
            # Only plot a time series if it's non empty
            if len(time) > 0:
                flux = d[flt]["FLUXCAL"]
                fluxerr = d[flt]["FLUXCALERR"]
                ax.errorbar(time, flux, yerr=fluxerr, label=f"Filter {flt}")

        ax.set_title(SNID, fontsize=18)
        ax.legend(loc="best")
        ax.set_aspect("auto")

    plt.savefig(Path(settings.explore_dir) / "sample_lightcurves.png")


def plot_random_preprocessed_lightcurves(settings, SNIDs):
    """Plot lightcurves specified by SNID_idxs from the
    preprocessed, pickled database

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
        SNIDs (list): list of SN lightcurve IDs to plot
    """

    list_files = [
        f for f in glob.glob(os.path.join(settings.preprocessed_dir, "*_PHOT.pickle"))
    ]

    df = pd.concat(list(map(pd.read_pickle, list_files))).set_index("SNID")

    # Plot and save
    plot_lightcurves(df, SNIDs, settings)


def plot_lightcurves_from_hdf5(settings, SNID_idxs):
    """Plot lightcurves specified by SNID_idxs from the
    HDF5 database

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
        SNID_idxs (list): list of SN lightcurve index to plot
    """

    with h5py.File(settings.hdf5_file_name, "r") as hf:

        features = hf["features"][:].astype(str)
        n_features = len(features)

        plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(4, 4, hspace=0.4)

        for idx, SNID_idx in enumerate(SNID_idxs):

            ax = plt.subplot(gs[idx])

            SNID = hf["SNID"][SNID_idx]
            PEAKMJD = str(hf["PEAKMJD"][SNID_idx])
            PEAKMJDNORM = hf["PEAKMJDNORM"][SNID_idx]
            data = hf["data"][SNID_idx].reshape(-1, n_features)

            df = pd.DataFrame(data, columns=features)

            non_filter_columns = [
                "FLUXCAL_g",
                "FLUXCAL_i",
                "FLUXCAL_r",
                "FLUXCAL_z",
                "FLUXCALERR_g",
                "FLUXCALERR_i",
                "FLUXCALERR_r",
                "FLUXCALERR_z",
                "delta_time",
                "HOSTGAL_PHOTOZ",
                "HOSTGAL_PHOTOZ_ERR",
                "HOSTGAL_SPECZ",
                "HOSTGAL_SPECZ_ERR",
            ]

            filter_columns = [
                c for c in df.columns.values if c not in non_filter_columns
            ]

            present_filters = df[filter_columns].transpose().idxmax().values
            list_present_filters = [set(f) for f in present_filters]

            max_y = -float("Inf")
            min_y = float("Inf")

            for FLT in settings.list_filters:
                idxs = np.array(
                    [i for i in range(len(df)) if FLT in list_present_filters[i]]
                )
                if len(idxs) == 0:
                    continue
                arr_flux = df[f"FLUXCAL_{FLT}"].values[idxs]
                arr_fluxerr = df[f"FLUXCALERR_{FLT}"].values[idxs]
                arr_time = df["delta_time"].cumsum().values[idxs]
                ax.errorbar(arr_time, arr_flux, yerr=arr_fluxerr, label=f"Filter {FLT}")

                if np.max(arr_flux) > max_y:
                    max_y = np.max(arr_flux)

                if np.min(arr_flux) < min_y:
                    min_y = np.min(arr_flux)

            ax.plot(
                [PEAKMJDNORM, PEAKMJDNORM], [min_y, max_y], color="k", linestyle="--"
            )
            ax.set_title(f"{SNID} -- {PEAKMJD}", fontsize=18)
            ax.legend(loc="best")
            ax.set_aspect("auto")

        plt.savefig(Path(settings.explore_dir) / "sample_lightcurves_from_hdf5.png")


def visualize(settings):
    """Plot a random subset of lightcurves

    2 plots: one with preprocessed data and one with processed data
    The two plots should show the same data

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
    """

    # Check the data has been created
    settings.check_data_exists()

    # Set a random seed
    np.random.seed()

    with h5py.File(settings.hdf5_file_name, "r") as hf:
        SNID_idxs = np.random.permutation(hf["SNID"].shape[0])[:16]
        SNIDs = hf["SNID"][:][SNID_idxs]

    plot_random_preprocessed_lightcurves(settings, SNIDs)
    plot_lightcurves_from_hdf5(settings, SNID_idxs)
