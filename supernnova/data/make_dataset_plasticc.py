import os
import h5py
import glob
import shlex
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from functools import partial

from ..utils import data_utils
from ..utils import logging_utils

from concurrent.futures import ProcessPoolExecutor


def preprocess_plasticc(settings, df):
    """

    """

    # Rename columns
    df = df.rename(
        columns={
            "object_id": "SNID",
            "passband": "FLT",
            "flux": "FLUXCAL",
            "flux_err": "FLUXCALERR",
            "mjd": "MJD",
            "target": "raw_target",
            "hostgal_photoz": "HOSTGAL_PHOTOZ",
            "hostgal_photoz_err": "HOSTGAL_PHOTOZ_ERR",
            "hostgal_specz": "HOSTGAL_SPECZ",
        }
    )
    df.FLT = df.FLT.replace(data_utils.DICT_PLASTICC_FILTERS)

    df["HOSTGAL_PHOTOZ"] = df["HOSTGAL_PHOTOZ"].fillna(-1)
    df["HOSTGAL_PHOTOZ_ERR"] = df["HOSTGAL_PHOTOZ_ERR"].fillna(-1)
    df["HOSTGAL_SPECZ"] = df["HOSTGAL_SPECZ"].fillna(-1)

    df = data_utils.compute_delta_time(df)
    df["rel_time"] = df[["SNID", "delta_time"]].groupby("SNID").cumsum()

    if "raw_target" in df.columns.values.tolist():
        df["target"] = df.raw_target.replace(data_utils.DICT_PLASTICC_CLASS)

    return df


def pivot_dataframe(df):
    """
    """

    list_filters = data_utils.PLASTICC_FILTERS

    arr_MJD = df.MJD.values
    arr_delta_time = df.delta_time.values

    # Loop over times to create grouped MJD:
    # if filters are acquired within less than 0.33 MJD (~8 hours) of each other
    # they get assigned the same time
    min_dt = 0.33
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

    # Some filters may appear multiple times with the same grouped MJD within same light curve
    # When this happens, we select the one with lowest FLUXCALERR
    df = df.sort_values("FLUXCALERR").groupby(["SNID", "grouped_MJD", "FLT"]).first()
    # We then reset the index
    df = df.reset_index()

    # drop columns that won"t be used onwards
    df = df.drop(
        [
            "MJD",
            "delta_time",
            "rel_time",
            "ra",
            "decl",
            "gal_b",
            "gal_l",
            "detected",
            "mwebv",
            "ddf",
            "distmod",
        ],
        1,
    )

    group_features_list = ["SNID", "grouped_MJD"]
    redshift_features = [k for k in df.keys() if "HOST" in k]

    if "target" in df.columns.values.tolist():
        group_features_list += ["target", "raw_target"]

    # Create a df of redshift information.
    # This is because sometimes, redshift is not available (NaN)
    # NaN create subtle bugs when pivoting !
    # We'll add the redshift info back to the pivoted dataframe afterwards
    # For a given lightcurve, redshift is fixed, so we can groupby and apply first
    # to create a dataframe containing one line per SNID, with the redshift
    df_redshift = df[["SNID"] + redshift_features].groupby("SNID").first()

    df = df.drop(redshift_features, axis=1)

    # Pivot so that for a given MJD, we have info on all available fluxes / error
    SNID_before_pivot = set(df.SNID.unique().tolist())
    df = pd.pivot_table(df, index=group_features_list, columns=["FLT"])

    # Flatten columns
    df.columns = ["_".join(col).strip() for col in df.columns.values]
    # Reset index to get grouped_MJD and target as columns
    cols_to_reset_list = [c for c in df.index.names if c != "SNID"]
    df.reset_index(cols_to_reset_list, inplace=True)
    # Rename grouped_MJD to MJD
    df.rename(columns={"grouped_MJD": "MJD"}, inplace=True)

    SNID_after_pivot = set(np.unique(df.index.values).tolist())
    assert SNID_before_pivot == SNID_after_pivot

    df = df.join(df_redshift, how="left")

    # New column to indicate which filter is present
    # The column will read ``rg`` if r,g are present; ``rgz`` if r,g and z are present, etc.
    for flt in list_filters:
        df[flt] = np.where(df[f"FLUXCAL_{flt}"].isnull(), "", flt)
    df["FLT"] = df[list_filters[0]]
    for flt in list_filters[1:]:
        df["FLT"] += df[flt]
    # Drop some irrelevant columns
    df = df.drop(list_filters, 1)
    # Finally replace NaN with 0
    df = df.fillna(0)
    # Add delta_time back. We removed all delta time columns above as they get
    # filled with NaN during pivot. It is clearer to recompute delta time once the pivot is complete
    df = data_utils.compute_delta_time(df)

    # Cast columns to float32 to save space
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)

    df.drop(columns="MJD", inplace=True)

    return df


def save_to_HDF5(settings, df, hdf5_file_name=None, verbose=True):
    """
    """
    list_training_features = settings.training_features_to_normalize + [
        "HOSTGAL_PHOTOZ",
        "HOSTGAL_PHOTOZ_ERR",
        "HOSTGAL_SPECZ",
    ]

    list_training_features
    assert df.index.name == "SNID", "SNID must be set as index"

    # Get the list of lightcurve IDs
    ID = df.index.values
    # Find out when ID changes => find start and end idx of each lightcurve
    idx_change = np.where(ID[1:] != ID[:-1])[0] + 1
    idx_change = np.hstack(([0], idx_change, [len(df)]))
    list_start_end = [(s, e) for s, e in zip(idx_change[:-1], idx_change[1:])]
    # N.B. We could use df.loc[SNID], more elegant but much slower

    logging_utils.print_green(
        f"Found {len(np.unique(ID))} unique lightcurves in dataset", verbose=verbose
    )

    # # Shuffle
    # np.random.shuffle(list_start_end)

    # Save hdf5 file
    if hdf5_file_name is None:
        hdf5_file_name = settings.hdf5_file_name
    with h5py.File(hdf5_file_name, "w") as hf:

        n_samples = len(list_start_end)

        # These arrays can be filled in one shot
        start_idxs = [i[0] for i in list_start_end]
        shuffled_ID = ID[start_idxs]
        hf.create_dataset(
            "SNID", data=shuffled_ID.astype(np.int32), dtype=np.dtype("int32")
        )

        logging_utils.print_green("Saving class", verbose=verbose)
        # only save target, splits, normalizations when building the training database
        if "target" in df.columns.values.tolist():

            # Save target info
            class_columns = ["target", "raw_target"]
            for class_column in class_columns:
                hf.create_dataset(
                    class_column,
                    data=df[class_column].values[start_idxs],
                    dtype=np.dtype("int64"),
                )
                # 0 if trainig set, 1 if valid set
                split_idxs = np.random.binomial(
                    1, 0.1, size=(len(list_start_end),)
                ).astype(np.int64)
                df = df.drop(class_column, axis=1)

            # Save dataset split
            logging_utils.print_green("Saving splits", verbose=verbose)
            hf.create_dataset("dataset", data=split_idxs, dtype=np.dtype("int64"))

            ########################
            # Normalize per feature
            ########################
            logging_utils.print_green("Compute normalizations", verbose=verbose)
            gnorm = hf.create_group("normalizations")

            # using normalization per feature
            for feat in settings.training_features_to_normalize:

                # Log transform plus mean subtraction and standard dev subtraction
                log_standardized = data_utils.log_standardization(df[feat].values)
                # Store normalization parameters
                gnorm.create_dataset(f"{feat}/min", data=log_standardized.arr_min)
                gnorm.create_dataset(f"{feat}/mean", data=log_standardized.arr_mean)
                gnorm.create_dataset(f"{feat}/std", data=log_standardized.arr_std)

            #####################################
            # Normalize flux and fluxerr globally
            #####################################
            logging_utils.print_green("Compute global normalizations", verbose=verbose)
            gnorm = hf.create_group("normalizations_global")

            ################
            # FLUX features
            #################
            flux_features = [f"FLUXCAL_{f}" for f in data_utils.PLASTICC_FILTERS]
            flux_log_standardized = data_utils.log_standardization(
                df[flux_features].values
            )
            # Store normalization parameters
            gnorm.create_dataset(f"FLUXCAL/min", data=flux_log_standardized.arr_min)
            gnorm.create_dataset(f"FLUXCAL/mean", data=flux_log_standardized.arr_mean)
            gnorm.create_dataset(f"FLUXCAL/std", data=flux_log_standardized.arr_std)

            ###################
            # FLUXERR features
            ###################
            fluxerr_features = [f"FLUXCALERR_{f}" for f in data_utils.PLASTICC_FILTERS]
            fluxerr_log_standardized = data_utils.log_standardization(
                df[fluxerr_features].values
            )
            # Store normalization parameters
            gnorm.create_dataset(
                f"FLUXCALERR/min", data=fluxerr_log_standardized.arr_min
            )
            gnorm.create_dataset(
                f"FLUXCALERR/mean", data=fluxerr_log_standardized.arr_mean
            )
            gnorm.create_dataset(
                f"FLUXCALERR/std", data=fluxerr_log_standardized.arr_std
            )

        ####################################
        # Save the rest of the data to hdf5
        ####################################

        # This type allows one to store flat arrays of variable
        # length inside an HDF5 group
        data_type = h5py.special_dtype(vlen=np.dtype("float32"))

        hf.create_dataset("data", (n_samples,), dtype=data_type)

        # Remove FLT column
        df = df.drop("FLT", axis=1)

        # store feature names
        list_training_features = df.columns.values.tolist()
        hf.create_dataset(
            "features",
            (len(list_training_features),),
            dtype=h5py.special_dtype(vlen=str),
        )
        hf["features"][:] = list_training_features
        logging_utils.print_green(
            "Saved features:", ",".join(list_training_features), verbose=verbose
        )

        # Save training features to hdf5
        logging_utils.print_green("Save data features to HDF5", verbose=verbose)
        arr_feat = df[list_training_features].values
        hf["data"].attrs["n_features"] = len(list_training_features)
        iterator = (
            enumerate(tqdm(list_start_end, desc="Filling hdf5", ncols=100))
            if verbose
            else enumerate(list_start_end)
        )
        for idx, idx_pair in iterator:
            arr = arr_feat[idx_pair[0] : idx_pair[1]]
            hf["data"][idx] = np.ravel(arr)


@logging_utils.timer("Plasticc Training data Processing")
def make_dataset(settings):
    """
    """

    df = pd.read_csv(os.path.join(settings.raw_dir, "training_set.csv"))
    df_meta = pd.read_csv(os.path.join(settings.raw_dir, "training_set_metadata.csv"))
    df = df.merge(df_meta, on="object_id", how="left")

    df = preprocess_plasticc(settings, df)
    df = pivot_dataframe(df)
    save_to_HDF5(settings, df)


def process_test_chunk(filename, settings):

    df = pd.read_pickle(filename)

    if len(df) == 0:
        return

    # Apply processing
    hdf5_file_name = Path(settings.processed_dir) / Path(
        os.path.basename(filename)
    ).with_suffix(".h5")
    df = preprocess_plasticc(settings, df)
    df = pivot_dataframe(df)
    save_to_HDF5(settings, df, hdf5_file_name, verbose=False)


@logging_utils.timer("Plasticc Test data Processing")
def make_test_dataset(settings):

    # Clear preproc dir
    for f in glob.glob(settings.preprocessed_dir + "/*"):
        logging_utils.print_green("Removing", f)
        os.remove(f)

    # Clear data dir of its test set constituents
    for f in glob.glob(settings.processed_dir + "/test_set_*.h5"):
        logging_utils.print_green("Removing", f)
        os.remove(f)

    num_chunks = 100

    # Load metadata
    df_meta = pd.read_csv(os.path.join(settings.raw_dir, "test_set_metadata.csv"))

    # We first cache the test dataset in multiple sub-chunks with pickle
    # This is reasonably fast

    # Count number of lines in the test set
    num_lines = int(
        subprocess.check_output(
            shlex.split(f"sed -n '$=' {os.path.join(settings.raw_dir, 'test_set.csv')}")
        )
        .decode()
        .rstrip()
    )
    chunksize = num_lines // num_chunks
    reader = pd.read_csv(
        os.path.join(settings.raw_dir, "test_set.csv"), sep=",", chunksize=chunksize
    )
    # Get the first chunk
    df_temp = next(reader)

    with tqdm(total=num_chunks, desc="Caching test data", ncols=100) as pbar:
        for chunk_idx, df_chunk in enumerate(reader):
            # Find out if df_chunk has an overlap with df_temp

            last_id = df_temp.object_id.values[-1]
            first_id = df_chunk.object_id.values[0]

            if last_id == first_id:
                df_temp = pd.concat(
                    [df_temp, df_chunk[df_chunk.object_id == last_id]]
                ).reset_index(drop=True)

            df_temp.merge(df_meta, on="object_id", how="left").to_pickle(
                os.path.join(
                    settings.preprocessed_dir, f"test_set_chunk_{chunk_idx}.pickle"
                )
            )

            df_temp = df_chunk[df_chunk.object_id != last_id].reset_index(drop=True)

            pbar.update(1)

    # Dump the last chunk
    if len(df_temp) != 0:
        df_temp.merge(df_meta, on="object_id", how="left").to_pickle(
            os.path.join(
                settings.preprocessed_dir, f"test_set_chunk_{chunk_idx + 1}.pickle"
            )
        )

    # Now we process all of the chunks in parallel
    list_pickles = glob.glob(
        os.path.join(settings.preprocessed_dir, "test_set_*.pickle")
    )

    process_fn = partial(process_test_chunk, settings=settings)

    # Split list pickles into chunks to get a progress bar with tqdm
    num_elem = len(list_pickles)
    chunk_size = min(10, num_elem)
    num_chunks = num_elem / chunk_size
    list_chunks = np.array_split(np.arange(num_elem), num_chunks)
    max_workers = multiprocessing.cpu_count() - 1
    for chunk_idxs in tqdm(list_chunks, desc="Processing test data", ncols=100):

        start_idx, end_idx = chunk_idxs[0], chunk_idxs[-1] + 1

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            executor.map(process_fn, list_pickles[start_idx:end_idx])

    list_SNID = []
    for f in glob.glob(os.path.join(settings.processed_dir, "test_set*.h5")):
        with h5py.File(f, "r") as hf:
            list_SNID += np.ravel(hf["SNID"][:]).tolist()

    print(len(np.unique(list_SNID)))
    print(len(np.unique(df_meta.object_id.values)))
