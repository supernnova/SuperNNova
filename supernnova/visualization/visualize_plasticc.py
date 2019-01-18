import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# plt.switch_backend("agg")
import matplotlib.gridspec as gridspec
from ..utils.data_utils import DICT_PLASTICC_FILTERS, PLASTICC_FILTERS
from ..data import make_dataset_plasticc


def plot_histogram(df, settings):

    list_fields = ["delta_time"]
    list_fields += ["FLUXCAL_%s" % k for k in ["r", "g", "z", "i"]]
    list_fields += ["FLUXCALERR_%s" % k for k in ["r", "g", "z", "i"]]

    plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(3, 3, hspace=0.4)
    for i in range(9):
        ax = plt.subplot(gs[i])
        arr = df[list_fields[i]].values
        ax.hist(arr, bins=500)
        # ax.set_xlim([minx, maxx])
        ax.set_title(list_fields[i], fontsize=22)
        ax.set_yscale("log")
    plt.savefig(os.path.join(settings.fig_dir, "histogram_feature_distribution.png"))


def plot_lightcurves(df, SNIDs):
    """Utility for gridspec of lightcruves

    Args:
        df (pandas.DataFrame): dataframe holding the data
        SNIDs (np.array or list): holds lightcurve ids
    """

    plt.figure(figsize=(20, 10))
    plt.suptitle("Sample of SN Ia light curves")
    gs = gridspec.GridSpec(4, 4, hspace=0.4)

    for i in range(16):
        ax = plt.subplot(gs[i])

        SNID = SNIDs[i]
        df_temp = df.loc[SNID]

        # Prepare plotting data in a dict
        d = {flt: {"FLUXCAL": [], "FLUXCALERR": [], "MJD": []} for flt in FILTERS}

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

    plt.savefig("pickle.svg")


def plot_lightcurves_from_hdf5(settings, SNID_idxs):

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

            for FLT in FILTERS:
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

        plt.savefig("figure.svg")


def visualize_plasticc(settings):

    # Check the data has been created
    settings.check_data_exists()

    d_rev = {value: key for (key, value) in DICT_PLASTICC_FILTERS.items()}

    # Set a random seed
    np.random.seed()

    with h5py.File(settings.hdf5_file_name, "r") as hf:
        SNID_idxs = np.random.permutation(hf["SNID"].shape[0])[:16]
        SNIDs = hf["SNID"][:][SNID_idxs]

    df = pd.read_csv(os.path.join(settings.raw_dir, "training_set.csv"))
    df_meta = pd.read_csv(os.path.join(settings.raw_dir, "training_set_metadata.csv"))
    df = df.merge(df_meta, on="object_id", how="left")

    df_preproc = make_dataset_plasticc.preprocess_plasticc(settings, df.copy())
    df_pivot = make_dataset_plasticc.pivot_dataframe(df_preproc.copy()).reset_index()

    for fig_idx, (SNID, SNID_idx) in enumerate(zip(SNIDs, SNID_idxs)):

        fig = plt.figure()
        gs = gridspec.GridSpec(4, 1)

        #########################
        # BEFORE ANY PROCESSING
        #########################

        ax = plt.subplot(gs[0])

        # Loop over data and plot
        df_tmp = df[df.object_id == SNID]
        d_flux = defaultdict(list)
        first_time = df_tmp.mjd.values[0]
        unique_flts = set([DICT_PLASTICC_FILTERS[f] for f in df_tmp.passband.unique()])
        for i in range(df_tmp.shape[0]):
            flux, fluxerr, flt, time = df_tmp[
                ["flux", "flux_err", "passband", "mjd"]
            ].values[i, :]
            flt = DICT_PLASTICC_FILTERS[flt]
            d_flux[f"FLUXCAL_{flt}"].append(flux)
            d_flux[f"FLUXCALERR_{flt}"].append(fluxerr)
            d_flux[f"time_{flt}"].append(time - first_time)

        for i, flt in enumerate(sorted(unique_flts)):
            ax.errorbar(
                d_flux[f"time_{flt}"],
                d_flux[f"FLUXCAL_{flt}"],
                d_flux[f"FLUXCALERR_{flt}"],
                label=flt,
                color=f"C{d_rev[flt]}",
            )
        # ax.legend(loc="best")

        #########################
        # After preprocessing
        #########################

        ax = plt.subplot(gs[1])
        # Loop over data and plot
        df_tmp = df_preproc[df_preproc.SNID == SNID]
        d_flux = defaultdict(list)
        first_time = df_tmp.MJD.values[0]
        unique_flts = df_tmp.FLT.unique()
        ax = plt.subplot(gs[1])
        for i in range(df_tmp.shape[0]):
            flux, fluxerr, flt, time = df_tmp[
                ["FLUXCAL", "FLUXCALERR", "FLT", "MJD"]
            ].values[i, :]
            d_flux[f"FLUXCAL_{flt}"].append(flux)
            d_flux[f"FLUXCALERR_{flt}"].append(fluxerr)
            d_flux[f"time_{flt}"].append(time - first_time)

        for i, flt in enumerate(sorted(unique_flts)):
            ax.errorbar(
                d_flux[f"time_{flt}"],
                d_flux[f"FLUXCAL_{flt}"],
                d_flux[f"FLUXCALERR_{flt}"],
                label=flt,
                color=f"C{d_rev[flt]}",
            )
        # ax.legend(loc="best")

        ################################
        # After preprocessing and pivot
        #################################

        ax = plt.subplot(gs[2])
        # Loop over data and plot
        df_tmp_pivot = df_pivot[df_pivot.SNID == SNID]

        df_tmp_pivot["time"] = df_tmp_pivot["delta_time"].cumsum()

        for i, FLT in enumerate(sorted(PLASTICC_FILTERS)):
            arr_flux = df_tmp_pivot[df_tmp_pivot[f"FLUXCAL_{FLT}"] != 0][
                f"FLUXCAL_{FLT}"
            ].values
            arr_fluxerr = df_tmp_pivot[df_tmp_pivot[f"FLUXCAL_{FLT}"] != 0][
                f"FLUXCALERR_{FLT}"
            ].values
            arr_time = df_tmp_pivot[df_tmp_pivot[f"FLUXCAL_{FLT}"] != 0]["time"].values
            if len(arr_flux) == 0:
                continue
            ax.errorbar(
                arr_time,
                arr_flux,
                yerr=arr_fluxerr,
                label=f"Filter {FLT}",
                color=f"C{d_rev[FLT]}",
            )

        ax.set_aspect("auto")

        #########################
        # After HDF5
        #########################

        with h5py.File(settings.hdf5_file_name, "r") as hf:

            features = hf["features"][:].astype(str)
            n_features = len(features)

            ax = plt.subplot(gs[3])

            SNID = hf["SNID"][SNID_idx]
            data = hf["data"][SNID_idx].reshape(-1, n_features)

            df_tmp = pd.DataFrame(data, columns=features)
            df_tmp["time"] = df_tmp.delta_time.cumsum()

            for i, FLT in enumerate(PLASTICC_FILTERS):
                arr_flux = df_tmp[df_tmp[f"FLUXCAL_{FLT}"] != 0][
                    f"FLUXCAL_{FLT}"
                ].values
                arr_fluxerr = df_tmp[df_tmp[f"FLUXCAL_{FLT}"] != 0][
                    f"FLUXCALERR_{FLT}"
                ].values
                arr_time = df_tmp[df_tmp[f"FLUXCAL_{FLT}"] != 0]["time"].values
                ax.errorbar(
                    arr_time,
                    arr_flux,
                    yerr=arr_fluxerr,
                    label=f"Filter {FLT}",
                    color=f"C{d_rev[FLT]}",
                )

            ax.set_title(f"{SNID}", fontsize=18)
            ax.legend(loc="best")
            ax.set_aspect("auto")

        plt.savefig(f"truc_fig_{fig_idx}.svg")
        plt.clf()
        plt.close()
