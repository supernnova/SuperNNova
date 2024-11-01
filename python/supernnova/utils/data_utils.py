import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.table import Table
from collections import namedtuple

from . import logging_utils

OFFSETS = [-2, -1, 0, 1, 2]
OOD_TYPES = ["random", "reverse", "shuffle", "sin"]
OFFSETS_STR = ["-7", "-2", "-1", "", "+1", "+2", "+30"]
OFFSETS_VAL = []
for v in OFFSETS_STR:
    if v == "":
        OFFSETS_VAL.append(0)
    else:
        OFFSETS_VAL.append(int(v.replace("+", "")))

LogStandardized = namedtuple("LogStandardized", ["arr_min", "arr_mean", "arr_std"])


def load_pandas_from_fit(fit_file_path):
    """Load a FIT file and cast it to a PANDAS dataframe

    Args:
        fit_file_path (str): path to FIT file

    Returns:
        (pandas.DataFrame) load dataframe from FIT file
    """
    dat = Table.read(fit_file_path, format="fits")
    df = dat.to_pandas()

    return df


def sntype_decoded(target, settings, simplify=False):
    """Match the target class (integer in {0, ..., 6} to the name
    of the class, i.e. something like "SN Ia" or "SN CC"

    Args:
        target (int): specifies the classification target
        settings (ExperimentSettings): custom class to hold hyperparameters
        simplify (Boolean): if True do not show all classes

    Returns:
        (str) the name of the class

    """
    if settings.nb_classes > 2:
        used = set()
        unique_classes = [
            x
            for x in settings.sntypes.values()
            if x not in used and (used.add(x) or True)
        ]
        SNtype = list(unique_classes)[target]
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
            if simplify:
                SNtype = "non SN Ia"
    return SNtype


def tag_type(df, settings, type_column="TYPE"):
    """Create classes based on a type columns

    Depending on the number of classes (2 or all), we create distinct
    target columns

    Args:
        df (pandas.DataFrame): the input dataframe
        settings (ExperimentSettings): controls experiment hyperparameters
        type_column (str): the type column in df

    Returns:
        (pandas.DataFrame) the dataframe, with new target columns
    """

    # SNTYPE checks
    if type_column not in df.keys():
        if settings.data_testing:
            df[settings.sntype_var] = np.ones(len(df)).astype(int)
        else:
            logging_utils.print_red(
                "Please provide SNTYPE with data (else use data_testing option)"
            )
            raise Exception

    # 2 classes: Ia vs non Ia
    list_types = list(set([x for x in settings.sntypes.values()]))
    if "Ia" in list_types:
        df[type_column] = df[type_column].astype(str)
        # get keys of Ias, the rest tag them as CC
        keys_ia = [key for (key, value) in settings.sntypes.items() if value == "Ia"]
        df["target_2classes"] = df[type_column].apply(
            lambda x: 0 if x in keys_ia else 1
        )
    else:
        arr_temp = df[type_column].values.copy()
        df["target_2classes"] = (
            arr_temp != int(list(settings.sntypes.keys())[0])
        ).astype(np.uint8)

    # All classes
    used = set()
    unique_classes = [
        x for x in settings.sntypes.values() if x not in used and (used.add(x) or True)
    ]
    classes_to_use = dict([(y, x) for x, y in enumerate(unique_classes)])
    map_keys_to_classes = {}
    for k, v in settings.sntypes.items():
        map_keys_to_classes[k] = classes_to_use[v]

    # check if all types are given in input dictionary
    tmp = df[~df[type_column].isin(settings.sntypes.keys())]
    n = 0
    if len(tmp) > 0:
        logging_utils.print_red(
            "Missing sntypes",
            f"{tmp[type_column].unique()} binary tagged as class 1",
        )
        logging_utils.print_red("nb_classes !=2 will NOT work")
        extra_tag = max(map_keys_to_classes.values()) + 1
        for mtyp in tmp[type_column].unique():
            map_keys_to_classes[mtyp] = extra_tag
    len(unique_classes) + n

    df[f"target_{len(unique_classes)}classes"] = df[type_column].apply(
        lambda x: map_keys_to_classes[x]
    )

    return df


def load_fitfile(settings, verbose=True):
    """Load the FITOPT file as a pandas dataframe

    Pickle it for future use (it is faster to load as a pickled dataframe)

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
        verbose (bool): whether to display logging message. Default: ``True``

    Returns:
        (pandas.DataFrame) dataframe with FITOPT data
    """
    if verbose:
        logging_utils.print_green("Loading FITRES file")

    if os.access(f"{settings.preprocessed_dir}/FITOPT000.FITRES.pickle", os.R_OK):
        df = pd.read_pickle(f"{settings.preprocessed_dir}/FITOPT000.FITRES.pickle")
        if verbose:
            print(f"Loaded {settings.preprocessed_dir}/FITOPT000.FITRES.pickle")

    elif os.access(f"{settings.fits_dir}/FITOPT000.FITRES", os.R_OK) or os.access(
        f"{settings.fits_dir}/FITOPT000.FITRES.gz", os.R_OK
    ):
        fit_name = (
            f"{settings.fits_dir}/FITOPT000.FITRES"
            if os.access(f"{settings.fits_dir}/FITOPT000.FITRES", os.R_OK)
            else f"{settings.fits_dir}/FITOPT000.FITRES.gz"
        )
        df = pd.read_csv(
            fit_name, index_col=False, comment="#", delimiter=" ", skipinitialspace=True
        )
        df = tag_type(df, settings)

        # Rename CID to SNID
        # SNID is CID in FITOPT000.FITRES
        if "SNID" not in df.keys():
            df = df.rename(columns={"CID": "SNID"})

        # Save to pickle for later use and fast reload
        df.to_pickle(f"{settings.preprocessed_dir}/FITOPT000.FITRES.pickle", protocol=4)
        if verbose:
            print(f"Loaded {fit_name}")
    else:
        # returning empty df
        df = pd.DataFrame()
        if verbose:
            logging_utils.print_yellow("Warning: No FITRES file to load")

    return df


def process_header_FITS(file_path, settings, columns=None):
    """Read the HEAD FIT file, add target columns and return
    in pandas DataFrame format

    Args:
        file_path (str): the  path to the header FIT file
        settings (ExperimentSettings): controls experiment hyperparameters
        columns (lsit): list of columns to keep. Default: ``None``

    Returns:
        (pandas.DataFrame) the dataframe, with new target columns
    """

    # Data
    df = load_pandas_from_fit(file_path)

    try:
        df["SNID"] = df["SNID"].str.decode("utf-8")
    except Exception:
        df["SNID"] = df["SNID"].astype(str)

    df[settings.sntype_var] = df[settings.sntype_var].astype(str)

    df = tag_type(df, settings, type_column=settings.sntype_var)

    if columns is not None:
        df = df[columns]

    return df


def process_header_csv(file_path, settings, columns=None):
    """Read the HEAD csv file, add target columns and return
    in pandas DataFrame format

    Args:
        file_path (str): the  path to the header FIT file
        settings (ExperimentSettings): controls experiment hyperparameters
        columns (lsit): list of columns to keep. Default: ``None``

    Returns:
        (pandas.DataFrame) the dataframe, with new target columns
    """

    # Data
    df = pd.read_csv(file_path)
    df = tag_type(df, settings, type_column=settings.sntype_var)

    if columns is not None:
        df = df[columns]
    return df


# TODO: double-check the following function that been commented out before deleting. It seems only related to randomforest.
# def add_redshift_features(settings, df):
#     """Add redshift features to pandas dataframe.

#     Args:
#         settings (ExperimentSettings): controls experiment hyperparameters
#         df (str): pandas DataFrame with FIT data

#     Returns:
#         (pandas.DataFrame) the dataframe, possibly with added redshift features
#     """

#     # check if we use host redshift as feature
#     host_features = [f for f in settings.randomforest_features if "HOST" in f]
#     use_redshift = len(host_features) > 0

#     if use_redshift > 0:
#         logging_utils.print_green("Adding redshift features...")

#         columns_to_read = ["SNID"] + host_features
#         # reading from batch pickles
#         list_files = natsorted(glob.glob(f"{settings.preprocessed_dir}/*_PHOT.pickle"))
#         # Check file with redshift features exist
#         error_msg = "Preprocessed_file not found. Call python run.py --data"
#         assert os.path.isfile(list_files[0]), error_msg

#         extra_info_df = pd.concat(
#             [pd.read_pickle(f)[columns_to_read] for f in list_files]
#         )

#         # In extra_info_df, there are many SNID duplicates as each row corresponds to a time step in a given curve
#         # We use groupby + first to only select the first row of each lightcurve
#         # Then we can merge knowing there won't be SNID duplicates in extra_info_df
#         extra_info_df = (
#             extra_info_df.groupby("SNID")[host_features].first().reset_index()
#         )

#         # Add redshift info to df
#         df = df.merge(extra_info_df, how="left", on="SNID")

#     return df


def compute_delta_time(df):
    """Compute the delta time between two consecutive observations

    Args:
        df (pandas.DataFrame): dataframe holding lightcurve data

    Returns:
        (pandas.DataFrame) dataframe holding lightcurve data with delta_time features
    """
    # in case the photometyr is not time sorted
    df = df.sort_values(["SNID", "MJD"])
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


def load_HDF5_SNinfo(settings):
    """Load physical information related to the created database of lightcurves

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters

    Returns:
        (pandas.DataFrame) dataframe holding physics information about the dataset
    """

    file_name = f"{settings.processed_dir}/database.h5"

    dict_SNinfo = {}
    with h5py.File(file_name, "r") as hf:

        columns_to_keep = ["SNID", settings.sntype_var, "mB", "c", "x1"]

        columns_to_keep += [c for c in hf.keys() if "SIM_" in c]
        columns_to_keep += [c for c in hf.keys() if "dataset_" in c]
        columns_to_keep += [c for c in hf.keys() if "PEAK" in c]

        for key in columns_to_keep:
            dict_SNinfo[key] = hf[key][:]

    df_SNinfo = pd.DataFrame(dict_SNinfo)
    # bytes
    if isinstance(df_SNinfo["SNID"].values[0], bytes):
        df_SNinfo["SNID"] = df_SNinfo["SNID"].str.decode("utf8")
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

    if arr_min < -2000:
        logging_utils.print_yellow(
            f"Warning: extreme data values {arr_min}",
            "clipping normalization min to -2000",
        )
        arr_min = -2000

    return LogStandardized(arr_min=arr_min, arr_mean=arr_mean, arr_std=arr_std)


def save_to_HDF5(settings, df):
    """Saved processed dataframe to HDF5

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
        df (pandas.DataFrame): dataframe holding processed data

    """
    # One hot encode filter information and Normalize features
    list_training_features = [f"FLUXCAL_{f}" for f in settings.list_filters]
    list_training_features += [f"FLUXCALERR_{f}" for f in settings.list_filters]
    list_training_features += [
        "delta_time",
        "HOSTGAL_PHOTOZ",
        "HOSTGAL_PHOTOZ_ERR",
        "HOSTGAL_SPECZ",
        "HOSTGAL_SPECZ_ERR",
    ]
    if settings.additional_train_var:
        list_training_features += list(settings.additional_train_var)

    list_misc_features = [
        "PEAKMJD",
        settings.sntype_var,
        "mB",
        "c",
        "x1",
        "SIM_REDSHIFT_CMB",
        "SIM_PEAKMAG_z",
        "SIM_PEAKMAG_g",
        "SIM_PEAKMAG_r",
        "SIM_PEAKMAG_i",
    ]

    if settings.photo_window_var not in list_misc_features:
        list_misc_features += settings.photo_window_var

    list_misc_features = [k for k in list_misc_features if k in df.keys()]

    assert df.index.name == "SNID", "Must set SNID as index"

    # Get the list of lightcurve IDs
    ID = df.index.values
    # Find out when ID changes => find start and end idx of each lightcurve
    idx_change = np.where(ID[1:] != ID[:-1])[0] + 1
    idx_change = np.hstack(([0], idx_change, [len(df)]))
    list_start_end = [(s, e) for s, e in zip(idx_change[:-1], idx_change[1:])]
    # N.B. We could use df.loc[SNID], more elegant but much slower

    # Filter list start end so we get only light curves with at least 3 points
    # except when creating testing data (we want to classify all lcs even w. 1-2 epochs)
    if not settings.data_testing:
        list_start_end = list(filter(lambda x: x[1] - x[0] >= 3, list_start_end))

    # Shuffle
    np.random.shuffle(list_start_end)

    # Save hdf5 file
    with h5py.File(settings.hdf5_file_name, "w") as hf:

        n_samples = len(list_start_end)
        used = set()
        unique_classes = [
            x
            for x in settings.sntypes.values()
            if x not in used and (used.add(x) or True)
        ]
        list_classes = list(set([2, len(unique_classes)]))
        list_names = ["target", "dataset_photometry", "dataset_saltfit"]

        # These arrays can be filled in one shot
        start_idxs = [i[0] for i in list_start_end]
        shuffled_ID = ID[start_idxs]
        hf.create_dataset("SNID", data=shuffled_ID, dtype=h5py.special_dtype(vlen=str))
        df_SNID = pd.DataFrame(shuffled_ID, columns=["SNID"])
        logging_utils.print_green("Saving misc features")
        for feat in list_misc_features:
            if feat == settings.sntype_var:
                dtype = np.dtype("int32")
            else:
                dtype = np.dtype("float32")
            hf.create_dataset(feat, data=df[feat].values[start_idxs], dtype=dtype)
            df.drop(columns=feat, inplace=True)

        logging_utils.print_green("Saving class")
        for c_ in list_classes:
            for name in list_names:
                field_name = f"{name}_{c_}classes"
                hf.create_dataset(
                    field_name,
                    data=df[field_name].values[start_idxs],
                    dtype=np.dtype("int8"),
                )
                df.drop(columns=field_name, inplace=True)

        df["time"] = df[["delta_time"]].groupby(df.index).cumsum()
        df = df.reset_index()

        logging_utils.print_green("Saving unique nights")
        # Compute how many unique nights of data taking existed around PEAKMJD
        for offset, suffix in zip(OFFSETS, OFFSETS_STR):
            new_column = f"PEAKMJD{suffix}_unique_nights"
            df_nights = (
                df[df["time"] < df["PEAKMJDNORM"] + offset][["PEAKMJDNORM", "SNID"]]
                .groupby("SNID")
                .count()
                .astype(np.uint8)
                .rename(columns={"PEAKMJDNORM": new_column})
                .reset_index()
            )

            hf.create_dataset(
                new_column,
                data=df_SNID.merge(df_nights, on="SNID", how="left")[new_column].values,
                dtype=np.dtype("uint8"),
            )

        logging_utils.print_green("Saving filter occurences")
        # Compute how many occurences of a specific filter around PEAKMJD
        for flt in settings.list_filters:
            # Check presence / absence of the filter at all time steps
            df[f"has_{flt}"] = df.FLT.str.contains(flt).astype(np.uint8)
            for offset, suffix in zip(OFFSETS, OFFSETS_STR):
                new_column = f"PEAKMJD{suffix}_num_{flt}"
                df_flt = (
                    df[df["time"] < df["PEAKMJDNORM"] + offset][[f"has_{flt}", "SNID"]]
                    .groupby("SNID")
                    .sum()
                    .astype(np.uint8)
                    .rename(columns={f"has_{flt}": new_column})
                    .reset_index()
                )
                hf.create_dataset(
                    new_column,
                    data=df_SNID.merge(df_flt, on="SNID", how="left")[
                        new_column
                    ].values,
                    dtype=np.dtype("uint8"),
                )

            df.drop(columns=f"has_{flt}", inplace=True)

        # FInally save PEAKMJDNORM
        hf.create_dataset(
            "PEAKMJDNORM",
            data=df["PEAKMJDNORM"].values[start_idxs],
            dtype=np.dtype("float32"),
        )
        cols_to_drop = [
            k
            for k in ["time", "SNID", "PEAKMJDNORM", settings.photo_window_var]
            if k in df.keys()
        ]
        df.drop(
            columns=list(set(cols_to_drop)),
            inplace=True,
        )

        ########################
        # Normalize per feature
        ########################
        logging_utils.print_green("Compute normalizations")
        gnorm = hf.create_group("normalizations")

        # using normalization per feature
        for feat in settings.training_features_to_normalize:
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
        flux_features = [f"FLUXCAL_{f}" for f in settings.list_filters]
        flux_log_standardized = log_standardization(df[flux_features].values)
        # Store normalization parameters
        gnorm.create_dataset("FLUXCAL/min", data=flux_log_standardized.arr_min)
        gnorm.create_dataset("FLUXCAL/mean", data=flux_log_standardized.arr_mean)
        gnorm.create_dataset("FLUXCAL/std", data=flux_log_standardized.arr_std)

        ###################
        # FLUXERR features
        ###################
        fluxerr_features = [f"FLUXCALERR_{f}" for f in settings.list_filters]
        fluxerr_log_standardized = log_standardization(df[fluxerr_features].values)
        # Store normalization parameters
        gnorm.create_dataset("FLUXCALERR/min", data=fluxerr_log_standardized.arr_min)
        gnorm.create_dataset("FLUXCALERR/mean", data=fluxerr_log_standardized.arr_mean)
        gnorm.create_dataset("FLUXCALERR/std", data=fluxerr_log_standardized.arr_std)

        ####################################
        # Save the rest of the data to hdf5
        ####################################

        logging_utils.print_green("Save non-data features to HDF5")

        # This type allows one to store flat arrays of variable
        # length inside an HDF5 group
        data_type = h5py.special_dtype(vlen=np.dtype("float32"))

        hf.create_dataset("data", (n_samples,), dtype=data_type)

        # If header does not have HOST info fill with empty arrays
        list_to_fill = [
            k for k in list_training_features if k not in df.columns.values.tolist()
        ]
        if len([k for k in list_to_fill if "HOST" not in k]) > 0:
            logging_utils.print_red("missing information in input")
            raise AttributeError
        for key in list_to_fill:
            df[key] = np.zeros(len(df))

        logging_utils.print_green("Fit onehot on FLT")
        assert sorted(df.columns.values.tolist()) == sorted(
            list_training_features + ["FLT"]
        )

        # Fit a one hot encoder for FLT
        # to have the same onehot for all datasets
        # tmp = pd.Series(settings.list_filters_combination).append(df["FLT"])
        tmp = pd.concat(
            [pd.Series(settings.list_filters_combination), df["FLT"]]
        )  # TODO: NEED TO TEST THIS LINE
        tmp_onehot = pd.get_dummies(tmp)
        # this is ok since it goes by length not by index (which I never reset)
        FLT_onehot = tmp_onehot[len(settings.list_filters_combination) :]
        df = pd.concat([df[list_training_features], FLT_onehot], axis=1)
        # store feature names
        list_training_features = df.columns.values.tolist()
        hf.create_dataset(
            "features",
            (len(list_training_features),),
            dtype=h5py.special_dtype(vlen=str),
        )
        hf["features"][:] = list_training_features
        logging_utils.print_green("Saved features:", ",".join(list_training_features))

        # Save training features to hdf5
        logging_utils.print_green("Save data features to HDF5")
        arr_feat = df[list_training_features].values
        hf["data"].attrs["n_features"] = len(list_training_features)
        for idx, idx_pair in enumerate(
            tqdm(list_start_end, desc="Filling hdf5", ncols=100)
        ):
            arr = arr_feat[idx_pair[0] : idx_pair[1]]
            hf["data"][idx] = np.ravel(arr)

        # save data types for training
        try:
            hf["data_types_training"] = np.asarray(settings.data_types_training).astype(
                np.dtype("S100")
            )
        except Exception:
            hf["data_types_training"] = f"{settings.data_types_training}"
