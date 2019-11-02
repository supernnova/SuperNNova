import os
import glob
import h5py
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from astropy.table import Table
from collections import namedtuple, OrderedDict

from . import logging_utils


LogStandardized = namedtuple("LogStandardized", ["arr_min", "arr_mean", "arr_std"])


def sntype_decoded(target, settings):
    """Match the target class (integer in {0, ..., 6} to the name
    of the class, i.e. something like "SN Ia" or "SN CC"

    Args:
        target (int): specifies the classification target
        settings (ExperimentSettings): custom class to hold hyperparameters

    Returns:
        (str) the name of the class

    """
    if settings.nb_classes > 3:
        SNtype = list(settings.sntypes.values())[target]
    elif settings.nb_classes == 3:
        if target == 0:
            SNtype = f"SN {list(settings.sntypes.values())[0]}"
        elif target == 1:
            SNtype = f"SN CC Ix"
        else:
            SNtype = "SN CC IIx"
    else:
        list_types = list(set([x for x in settings.sntypes.values()]))
        if target == 0:
            if "Ia" in list_types:
                SNtype = "SN Ia"
            else:
                SNtype = f"SN {list(settings.sntypes.values())[0]}"
        else:
            if "Ia" in list_types:
                SNtype = f"SN {'|'.join(set([k for k in settings.sntypes.values() if 'Ia' not in k]))}"
            else:
                SNtype = f"SN {'|'.join(list(settings.sntypes.values())[1:])}"
    return SNtype


def tag_type(df, sntypes, type_column="TYPE"):
    """Create classes based on a type columns

    Depending on the number of classes (2 or all), we create distinct
    target columns

    Args:
        df (pandas.DataFrame): the input dataframe
        type_column (str): the type column in df

    Returns:
        (pandas.DataFrame) the dataframe, with new target columns
    """

    # 2 classes
    # taking the first type vs. others
    list_types = list(set([x for x in sntypes.values()]))
    if "Ia" in list_types:
        df[type_column] = df[type_column].astype(str)
        # get keys of Ias, the rest tag them as CC
        keys_ia = [key for (key, value) in sntypes.items() if value == "Ia"]
        df["target_2classes"] = df[type_column].apply(
            lambda x: 0 if x in keys_ia else 1
        )
    else:
        arr_temp = df[type_column].values.copy()
        df["target_2classes"] = (arr_temp != int(list(sntypes.keys())[0])).astype(
            np.uint8
        )

    # All classes
    # check if all types are given in input dictionary
    tmp = df[~df[type_column].isin(sntypes.keys())][type_column]
    if len(tmp) > 0:
        logging_utils.print_red("Not all sntypes are given in input")
        logging_utils.print_red(f"missing: {tmp.unique()}")
        logging_utils.print_red(f"tagging them as class {len(sntypes)}")

    classes_to_use = {}
    for i, k in enumerate(sntypes.keys()):
        classes_to_use[k] = i
    for kk in tmp.unique():
        classes_to_use[kk] = len(sntypes)

    df[f"target_{len(sntypes)}classes"] = df[type_column].apply(
        lambda x: classes_to_use[x]
    )

    return df


def load_fitfile(fitopt_file, sntypes):
    """Load the FITOPT file as a pandas dataframe

    Pickle it for future use (it is faster to load as a pickled dataframe)

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
        verbose (bool): whether to display logging message. Default: ``True``

    Returns:
        (pandas.DataFrame) dataframe with FITOPT data
    """
    df = pd.read_csv(
        fitopt_file, index_col=False, comment="#", delimiter=" ", skipinitialspace=True
    )
    df = tag_type(df, sntypes)

    # Rename CID to SNID
    # SNID is CID in FITOPT000.FITRES
    df = df.rename(columns={"CID": "SNID"})

    return df


def process_header(file_path, sntypes, columns=None):
    """Read the HEAD FIT file, add target columns and return
    in pandas DataFrame format

    Args:
        file_path (str): the  path to the header FIT file
        columns (lsit): list of columns to keep. Default: ``None``

    Returns:
        (pandas.DataFrame) the dataframe, with new target columns
    """

    extension = os.path.splitext(str(file_path))[-1]

    if "csv" in extension:
        df = pd.read_csv(file_path)
    elif "FITS" in extension:
        dat = Table.read(file_path, format="fits")
        df = dat.to_pandas()
    else:
        msg = f"Invalid file extension. Got {extension}, should be [FITS|csv]"
        raise RuntimeError(msg)

    df = tag_type(df, sntypes, type_column="SNTYPE")

    if columns is not None:
        df = df[columns]

    return df


def compute_delta_time(df):
    """Compute the delta time between two consecutive observations

    Args:
        df (pandas.DataFrame): dataframe holding lightcurve data

    Returns:
        (pandas.DataFrame) dataframe holding lightcurve data with delta_time features
    """

    df["delta_time"] = df["MJD"].diff()
    # Fill the first row with 0 to replace NaN
    df.delta_time = df.delta_time.fillna(0)
    try:
        IDs = df.SNID.values
    # Deal with the case where lightcrv_ID is the index
    except AttributeError:
        assert df.index.name == "SNID"
        IDs = df.index.values
    # Find idxs of rows where a new light curve start then zero delta_time
    idxs = np.where(IDs[:-1] != IDs[1:])[0] + 1
    arr_delta_time = df.delta_time.values
    arr_delta_time[idxs] = 0
    df["delta_time"] = arr_delta_time

    return df


def remove_data_post_large_delta_time(df):
    """
    Remove rows in the same light curve after a gap > 150 days
    Reason: If no signal has been saved in a time frame of 150 days,
    it is unlikely there is much left afterwards

    Args:
        df (pandas.DataFrame): dataframe holding lightcurve data

    Returns:
        (pandas.DataFrame) dataframe where large delta time rows have been removed
    """

    # Identify indices where delta time is large
    list_to_remove = []
    idx_high = np.where(df.delta_time.values > 150)[0]
    # Identify the lightcurve ID where this happens
    IDs = df.SNID.values
    # Loop over indices and remove row if they belong to the same light curve
    for idx in idx_high:
        ID = IDs[idx]
        same_lc = True
        while same_lc:
            list_to_remove.append(idx)
            idx += 1
            new_ID = IDs[idx]
            same_lc = new_ID == ID
    df = df.drop(list_to_remove)
    # Reset index to account for dropped rows
    df.reset_index(inplace=True, drop=True)

    return df


def load_HDF5_SNinfo(processed_dir):
    """
    """

    file_name = f"{processed_dir}/database.h5"

    with h5py.File(file_name, "r") as hf:

        print("WARNING COL TO KEEP")
        print("WARNING COL TO KEEP")
        print("WARNING COL TO KEEP")

        data = hf["metadata"][:]
        columns = hf["metadata"].attrs["columns"]

        df_SNinfo = pd.DataFrame(data, columns=columns)
        df_SNinfo["SNID"] = hf["SNID"][:]
        df_SNinfo["SNTYPE"] = hf["SNTYPE"][:]

    return df_SNinfo


def log_standardization(arr):
    """Normalization strategy for the fluxes and fluxes error

    - Log transform the data
    - Mean and std dev normalization

    Args:
        arr (np.array): data to normalize

    Returns:
        (LogStandardized) namedtuple holding normalization data
    """

    arr_min = np.min(arr)
    arr_log = np.log(-arr_min + arr + 1e-5)
    arr_mean = arr_log.mean()
    arr_std = arr_log.std()

    return LogStandardized(arr_min=arr_min, arr_mean=arr_mean, arr_std=arr_std)


def save_to_HDF5(
    df, hdf5_file, sntypes, list_filters, offsets, offsets_str, filter_dict
):
    """Saved processed dataframe to HDF5

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
        df (pandas.DataFrame): dataframe holding processed data

    """

    # Compute how many unique nights of data taking existed around PEAKMJD
    df["time"] = df[["SNID", "delta_time"]].groupby("SNID").cumsum()
    for offset, suffix in zip(offsets, offsets_str):
        new_column = f"PEAKMJD{suffix}_unique_nights"
        df_night = (
            df[df["time"] < df["PEAKMJDNORM"] + offset][["PEAKMJDNORM", "SNID"]]
            .groupby("SNID")
            .count()
            .rename(columns={f"PEAKMJDNORM": new_column})
            .reset_index()
        )
        df = df.merge(df_night, how="left", on="SNID")

    # Compute how many occurences of a specific filter around PEAKMJD
    for flt in list_filters:
        # Check presence / absence of the filter at all time steps
        df[f"has_{flt}"] = df.FLT.str.contains(flt).astype(int)
        for offset, suffix in zip(offsets, offsets_str):
            new_column = f"PEAKMJD{suffix}_num_{flt}"
            df_flt = (
                df[df["time"] < df["PEAKMJDNORM"] + offset][[f"has_{flt}", "SNID"]]
                .groupby("SNID")
                .sum()
                .astype(int)
                .rename(columns={f"has_{flt}": new_column})
                .reset_index()
            )
            df = df.merge(df_flt, how="left", on="SNID")
        df.drop(columns=f"has_{flt}", inplace=True)

    list_training_features = [f"FLUXCAL_{f}" for f in list_filters]
    list_training_features += [f"FLUXCALERR_{f}" for f in list_filters]
    list_training_features += ["delta_time"]
    list_features_to_normalize = [f for f in list_training_features if "FLUX" in f] + [
        "delta_time"
    ]

    list_metadata_features = [
        "mB",
        "c",
        "x1",
        "SIM_REDSHIFT_CMB",
        "SIM_PEAKMAG_z",
        "SIM_PEAKMAG_g",
        "SIM_PEAKMAG_r",
        "SIM_PEAKMAG_i",
    ]
    list_metadata_features += [f for f in df.columns.values if "PEAKMJD" in f]
    list_metadata_features += [f for f in df.columns.values if "HOSTGAL" in f]
    list_metadata_features = [k for k in list_metadata_features if k in df.keys()]

    # Get the list of lightcurve IDs
    ID = df.SNID.values
    # Find out when ID changes => find start and end idx of each lightcurve
    idx_change = np.where(ID[1:] != ID[:-1])[0] + 1
    idx_change = np.hstack(([0], idx_change, [len(df)]))
    list_start_end = [(s, e) for s, e in zip(idx_change[:-1], idx_change[1:])]
    # N.B. We could use df.loc[SNID], more elegant but much slower

    # Shuffle
    np.random.shuffle(list_start_end)

    # Drop features we no longer need
    df.drop(columns=["time", "SNID"], inplace=True)

    # Save hdf5 file
    with h5py.File(hdf5_file, "w") as hf:

        n_samples = len(list_start_end)

        # Fill non variable length arrays
        start_idxs = [i[0] for i in list_start_end]
        arr_metadata = df[list_metadata_features].values[start_idxs].astype(np.float32)
        hf.create_dataset("metadata", data=arr_metadata)
        hf["metadata"].attrs["columns"] = list_metadata_features
        hf.create_dataset("SNID", data=ID[start_idxs].astype(np.int64))
        hf.create_dataset(
            "SNTYPE", data=df["SNTYPE"].values[start_idxs].astype(np.int64)
        )

        df = df.drop(columns=list_metadata_features)
        df = df.drop(columns=["SNTYPE"])

        ########################
        # Normalize per feature
        ########################
        logging_utils.print_green("Compute normalizations")
        gnorm = hf.create_group("normalizations")

        # using normalization per feature
        for feat in list_features_to_normalize:
            # Log transform plus mean subtraction and standard dev subtraction
            log_standardized = log_standardization(df[feat].values)
            # Store normalization parameters
            gnorm.create_dataset(f"{feat}/min", data=log_standardized.arr_min)
            gnorm.create_dataset(f"{feat}/mean", data=log_standardized.arr_mean)
            gnorm.create_dataset(f"{feat}/std", data=log_standardized.arr_std)

        #####################################
        # Normalize flux and fluxerr globally
        #####################################
        logging_utils.print_green("Compute global normalizations")
        gnorm = hf.create_group("normalizations_global")

        ################
        # FLUX features
        #################
        flux_features = [f"FLUXCAL_{f}" for f in list_filters]
        flux_log_standardized = log_standardization(df[flux_features].values)
        # Store normalization parameters
        gnorm.create_dataset(f"FLUXCAL/min", data=flux_log_standardized.arr_min)
        gnorm.create_dataset(f"FLUXCAL/mean", data=flux_log_standardized.arr_mean)
        gnorm.create_dataset(f"FLUXCAL/std", data=flux_log_standardized.arr_std)

        ###################
        # FLUXERR features
        ###################
        fluxerr_features = [f"FLUXCALERR_{f}" for f in list_filters]
        fluxerr_log_standardized = log_standardization(df[fluxerr_features].values)
        # Store normalization parameters
        gnorm.create_dataset(f"FLUXCALERR/min", data=fluxerr_log_standardized.arr_min)
        gnorm.create_dataset(f"FLUXCALERR/mean", data=fluxerr_log_standardized.arr_mean)
        gnorm.create_dataset(f"FLUXCALERR/std", data=fluxerr_log_standardized.arr_std)

        ####################################
        # Save the rest of the data to hdf5
        ####################################

        logging_utils.print_green("Save non-data features to HDF5")

        # This type allows one to store flat arrays of variable
        # length inside an HDF5 group
        data_type = h5py.special_dtype(vlen=np.dtype("float32"))

        hf.create_dataset("data", (n_samples,), dtype=data_type)

        df["FLT"] = df["FLT"].map(filter_dict).astype(np.float32)

        list_training_features += ["FLT"]
        hf["data"].attrs["columns"] = list_training_features
        hf["data"].attrs["n_features"] = len(list_training_features)

        # Save training features to hdf5
        logging_utils.print_green("Save data features to HDF5")
        arr_feat = df[list_training_features].values
        for idx, idx_pair in enumerate(
            tqdm(list_start_end, desc="Filling hdf5", ncols=100)
        ):
            arr = arr_feat[idx_pair[0] : idx_pair[1]]
            hf["data"][idx] = np.ravel(arr)
