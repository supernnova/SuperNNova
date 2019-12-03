import os
import yaml
import shutil
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from natsort import natsorted
from functools import partial
from astropy.table import Table

from supernnova.utils import data_utils
from supernnova.utils import logging_utils
from plots import datasets_plots

from constants import SNTYPES, LIST_FILTERS, OFFSETS, OFFSETS_STR, FILTER_DICT, MIN_DT


def process_phot_file(file_path, preprocessed_dir, list_filters):
    """
    """

    # Load the PHOT and HEAD files
    dat = Table.read(file_path, format="fits")
    header = Table.read(file_path.replace("PHOT", "HEAD"), format="fits")
    df = dat.to_pandas()
    df_header = header.to_pandas()

    # Keep only columns of interest
    keep_col = ["MJD", "FLUXCAL", "FLUXCALERR", "FLT"]
    df = df[keep_col].copy()

    keep_col_header = [
        "SNID",
        "PEAKMJD",
        "HOSTGAL_PHOTOZ",
        "HOSTGAL_PHOTOZ_ERR",
        "HOSTGAL_SPECZ",
        "HOSTGAL_SPECZ_ERR",
        "SIM_REDSHIFT_CMB",
        "SIM_PEAKMAG_z",
        "SIM_PEAKMAG_g",
        "SIM_PEAKMAG_r",
        "SIM_PEAKMAG_i",
        "SNTYPE",
    ]
    keep_col_header = [k for k in keep_col_header if k in df_header.keys()]
    df_header = df_header[keep_col_header].copy()
    df_header["SNID"] = df_header["SNID"].astype(np.int64)

    # New light curves are identified by MJD == -777.0
    # Last line may be a line with MJD = -777.
    # Remove it so that it does not interfere with arr_ID below
    if df.MJD.values[-1] == -777.0:
        df = df.drop(df.index[-1])

    arr_ID = np.zeros(len(df), dtype=np.int64)
    arr_idx = np.where(df["MJD"].values == -777.0)[0]
    arr_idx = np.hstack((np.array([0]), arr_idx, np.array([len(df)])))
    # Fill in arr_ID
    for counter in range(1, len(arr_idx)):
        start, end = arr_idx[counter - 1], arr_idx[counter]
        # index starts at zero
        arr_ID[start:end] = df_header.SNID.iloc[counter - 1]
    df["SNID"] = arr_ID

    df = df.set_index("SNID")
    df_header = df_header.set_index("SNID")
    # join df and header
    df = df.join(df_header).reset_index()

    #############################################
    # Miscellaneous data processing
    #############################################
    df = df[keep_col + keep_col_header].copy()
    # filters have a trailing white space which we remove
    df.FLT = df.FLT.apply(lambda x: x.rstrip()).values.astype(str)
    # keep only filters we are going to use for classification
    df = df[df["FLT"].isin(list_filters)]
    # Drop the delimiter lines
    df = df[df.MJD != -777.000]
    # Reset the index (it is no longer continuous after dropping lines)
    df.reset_index(inplace=True, drop=True)
    # Add delta time
    df = data_utils.compute_delta_time(df)
    # Remove rows post large delta time in the same light curve(delta_time > 150)
    # df = data_utils.remove_data_post_large_delta_time(df)

    # Save for future use
    basename = os.path.basename(file_path)
    df.to_pickle(f"{preprocessed_dir}/{basename.replace('.FITS', '.pickle')}")

    # getting SNIDs for SNe with Host_spec
    host_spe = df[df["HOSTGAL_SPECZ"] > 0]["SNID"].unique().tolist()

    return host_spe


def preprocess_data(config):
    """
    """

    raw_dir = config["raw_dir"]
    processed_dir = config["processed_dir"]
    raw_format = config["raw_format"]
    preprocessed_dir = config["preprocessed_dir"]
    max_workers = max(1, multiprocessing.cpu_count() - 2)

    # Get the list of FITS files
    list_files = natsorted(map(str, Path(raw_dir).glob(f"*PHOT.{raw_format}*")))
    list_Ia = [f for f in list_files if "_Ia" in f]
    list_nonIa = [f for f in list_files if "_NONIa" in f]

    list_files = list_Ia[:2] + list_nonIa[:2]

    process_fn = partial(
        process_phot_file, preprocessed_dir=preprocessed_dir, list_filters=LIST_FILTERS
    )

    pool = multiprocessing.Pool(max_workers)
    n_files = len(list_files)
    chunk_size = min(len(list_files), 10)
    # Loop over chunks of files
    host_spe_tmp = []
    for idx in tqdm(range(0, n_files, chunk_size), desc="Preprocess", ncols=100):
        # Process each file in the chunk in parallel
        host_spe_tmp += pool.map(process_fn, list_files[idx : idx + chunk_size])

    host_spe = [item for sublist in host_spe_tmp for item in sublist]
    pd.DataFrame(host_spe, columns=["SNID"]).to_pickle(
        f"{processed_dir}/hostspe_SNID.pickle"
    )
    logging_utils.print_green("Finished preprocessing")


def pivot_dataframe_single(filename, list_filters, df_salt, sntypes):
    """
    """

    df = pd.read_pickle(filename)
    arr_MJD = df.MJD.values
    arr_delta_time = df.delta_time.values
    # Loop over times to create grouped MJD:
    # if filters are acquired within less than 0.33 MJD (~8 hours) of each other
    # they get assigned the same time
    min_dt = MIN_DT
    time_last_change = 0
    arr_grouped_MJD = np.zeros_like(arr_MJD)
    for i in range(len(df)):
        time = arr_MJD[i]
        dt = arr_delta_time[i]
        # 2 possibilities to update the time
        # dt == 0 (it"s a new light curve)
        # time - time_last_change > min_dt
        if dt == 0 or (time - time_last_change) > min_dt:
            arr_grouped_MJD[i] = time
            time_last_change = time
        else:
            arr_grouped_MJD[i] = arr_grouped_MJD[i - 1]
    # Add grouped delta time to dataframe
    df["grouped_MJD"] = np.array(arr_grouped_MJD)

    # Some filters  may appear multiple times with the same grouped MJD within same light curve
    # When this happens, we select the one with lowest FLUXCALERR
    df = df.sort_values("FLUXCALERR").groupby(["SNID", "grouped_MJD", "FLT"]).first()
    # We then reset the index
    df = df.reset_index()
    # Compute PEAKMJDNORM = PEAKMJD in days since the start of the light curve
    df["PEAKMJDNORM"] = df["PEAKMJD"] - df["MJD"]
    # The correct PEAKMJDNORM is the first one hence the use of first after groupby
    df_PEAKMJDNORM = df[["SNID", "PEAKMJDNORM"]].groupby("SNID").first().reset_index()
    # Remove PEAKMJDNORM
    df = df.drop("PEAKMJDNORM", 1)
    # Add PEAKMJDNORM back to df with a merge on SNID
    df = df.merge(df_PEAKMJDNORM, how="left", on="SNID")
    # drop columns that won"t be used onwards
    df = df.drop(["MJD", "delta_time"], 1)

    group_features_list = [
        "SNID",
        "grouped_MJD",
        "PEAKMJD",
        "PEAKMJDNORM",
        "SIM_REDSHIFT_CMB",
        "SNTYPE",
        "SIM_PEAKMAG_z",
        "SIM_PEAKMAG_g",
        "SIM_PEAKMAG_r",
        "SIM_PEAKMAG_i",
    ] + [k for k in df.keys() if "HOST" in k]
    # check if keys are in header
    group_features_list = [k for k in group_features_list if k in df.keys()]
    # Pivot so that for a given MJD, we have info on all available fluxes / error
    df = pd.pivot_table(df, index=group_features_list, columns=["FLT"])

    # Flatten columns
    df.columns = ["_".join(col).strip() for col in df.columns.values]
    # Reset index to get grouped_MJD and target as columns
    df.reset_index(df.index.names, inplace=True)
    # Rename grouped_MJD to MJD
    df.rename(columns={"grouped_MJD": "MJD"}, inplace=True)

    # New column to indicate which channel (r,g,z,i) is present
    # The column will read ``rg`` if r,g are present; ``rgz`` if r,g and z are present, etc.
    for flt in list_filters:
        df[flt] = np.where(df["FLUXCAL_%s" % flt].isnull(), "", flt)
    df["FLT"] = df[list_filters[0]]
    for flt in list_filters[1:]:
        df["FLT"] += df[flt]
    # Ensure combination is written in natural sorted order
    df["FLT"] = df.FLT.apply(lambda x: "".join(natsorted(list(x))))
    # Drop some irrelevant columns
    df = df.drop(list_filters, 1)
    # Finally replace NaN with 0
    df = df.fillna(0)
    # Add delta_time back. We removed all delta time columns above as they get
    # filled with NaN during pivot. It is clearer to recompute delta time once the pivot is complete
    df = data_utils.compute_delta_time(df)

    # Cast columns to float32
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)

    if df_salt is not None:
        df_salt["salt"] = 1
        df = df.merge(df_salt[["SNID", "mB", "c", "x1", "salt"]], on="SNID", how="left")
        df["salt"] = df["salt"].fillna(0)
    else:
        df["salt"] = 0

    df.drop(columns="MJD", inplace=True)
    # Save to pickle
    dump_filename = os.path.splitext(filename)[0] + "_pivot.pickle"
    df.to_pickle(dump_filename)


def pivot_dataframe_batch(list_files, config):
    """
    """

    fitopt_file = config.get("fitopt_file", None)
    max_workers = max(1, multiprocessing.cpu_count() - 2)

    pool = multiprocessing.Pool(max_workers)
    n_files = len(list_files)
    chunk_size = min(len(list_files), 10)

    # load FITOPT file on which we will base our splits
    df_salt = (
        data_utils.load_fitfile(fitopt_file, SNTYPES)
        if fitopt_file is not None
        else None
    )

    process_fn = partial(
        pivot_dataframe_single,
        list_filters=LIST_FILTERS,
        df_salt=df_salt,
        sntypes=SNTYPES,
    )

    # Loop over chunks of files
    for idx in tqdm(range(0, n_files, chunk_size), desc="Pivot", ncols=100):
        # Process each file in the chunk in parallel
        pool.map(process_fn, list_files[idx : idx + chunk_size])

    pool.close()

    logging_utils.print_green("Finished pivot")


@logging_utils.timer("Data processing")
def make_dataset(config_path):

    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    preprocessed_dir = config["preprocessed_dir"]
    processed_dir = config["processed_dir"]
    hdf5_file = (Path(processed_dir) / "database.h5").as_posix()

    # Clean up data folders
    for folder in [preprocessed_dir, processed_dir]:
        shutil.rmtree(folder, ignore_errors=True)
        Path(folder).mkdir(exist_ok=True, parents=True)

    # Preprocess dataset
    preprocess_data(config)

    # Pivot dataframe
    list_files = natsorted(map(str, Path(f"{preprocessed_dir}").glob("*PHOT*")))
    pivot_dataframe_batch(list_files, config)

    # Aggregate the pivoted dataframe
    list_files = natsorted(map(str, Path(f"{preprocessed_dir}").glob("*pivot.pickle*")))
    df = pd.concat([pd.read_pickle(f) for f in list_files], axis=0).reset_index(
        drop=True
    )

    # Save to HDF5
    data_utils.save_to_HDF5(
        df, hdf5_file, LIST_FILTERS, OFFSETS, OFFSETS_STR, FILTER_DICT
    )

    # Save plots to visualize the distribution of some of the data features
    SNinfo_df = data_utils.load_HDF5_SNinfo(config["processed_dir"])
    if np.any(SNinfo_df.salt.values == 1):
        fig_dir = Path(config["fig_dir"])
        fig_dir.mkdir(parents=True, exist_ok=True)
        datasets_plots(SNinfo_df, SNTYPES, fig_dir)

    # Clean preprocessed directory
    shutil.rmtree(preprocessed_dir)

    logging_utils.print_green("Finished making dataset")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dataset building")
    parser.add_argument("config_path", help="Path to yml config gile")

    args = parser.parse_args()

    make_dataset(args.config_path)
