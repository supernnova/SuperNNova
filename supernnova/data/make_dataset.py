import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from natsort import natsorted
from functools import partial
from astropy.table import Table
from concurrent.futures import ProcessPoolExecutor

from ..utils import data_utils
from ..utils import logging_utils
from ..paper.superNNova_plots import datasets_plots


def build_traintestval_splits(settings):
    """Build dataset split in the following way

    - Downsample each class so that it has the same cardinality as the lowest cardinality class
    - Randomly assign lightcurves to a 80/10/10 train test val split (except Out-of-distribution data 1/1/98)

    OOD:
        set almost all samples to testing and does not use saltfits for selection.
        Will use the complete sample for testing, does not require settings.data_prefix to be changed.

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
    """

    logging_utils.print_green(f"Computing splits")

    # load FITOPT file on which we will base our splits
    df_salt = data_utils.load_fitfile(settings)
    df_salt["is_salt"] = 1

    list_files = natsorted(
        glob.glob(os.path.join(settings.raw_dir, f"{settings.data_prefix}*HEAD.FITS"))
    )
    print("List files", list_files)

    # Read and process files faster with ProcessPoolExecutor
    max_workers = multiprocessing.cpu_count()
    photo_columns = ["SNID"] + [
        f"target_{nb_classes}classes"
        for nb_classes in [2, 3, len(settings.sntypes.keys())]
    ]
    process_fn = partial(
        data_utils.process_header_FITS,
        settings=settings,
        columns=photo_columns + ["SNTYPE"],
    )
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list_df = executor.map(process_fn, list_files)

    # Load df_photo
    df_photo = pd.concat(list_df)
    df_photo["SNID"] = df_photo["SNID"].astype(int)

    # Check all SNID in df_salt are also in df_photo
    assert np.all(np.in1d(df_salt.SNID.values, df_photo.SNID.values))

    # Merge left on df_photo
    df = df_photo.merge(df_salt[["SNID", "is_salt"]], on=["SNID"], how="left")
    # Some curves are in photo and not in salt, these curves have is_salt = NaN
    # We replace the NaN with 0
    df["is_salt"] = df["is_salt"].fillna(0).astype(int)

    # Save dataset stats
    list_stat = []

    # Save a dataframe to record train/test/val split for
    # binary, ternary and all-classes classification
    for dataset in ["saltfit", "photometry"]:
        for nb_classes in [2, 3, len(settings.sntypes.keys())]:
            print()
            logging_utils.print_green(
                f"Computing {dataset} splits for {nb_classes}-way classification"
            )

            # Randomly sample SNIDs such that all class have the same number of occurences
            if dataset == "saltfit":
                g = df[df.is_salt == 1].groupby(f"target_{nb_classes}classes")
            else:
                g = df.groupby(f"target_{nb_classes}classes")

            # Line below: we have grouped df by target, we find out which of those
            # group has the smallest size with g.size().min(), then we sample randomly
            # from this group and reset the index. We then sample with frac=1 to shuffle
            # the whole dataset. Otherwise, the classes are sorted and the train/test/val
            # splits are incorrect.
            g = (
                g.apply(lambda x: x.sample(g.size().min()))
                .reset_index(drop=True)
                .sample(frac=1)
            )

            all_SNIDs = df.SNID.values
            sampled_SNIDs = g["SNID"].values
            n_samples = len(sampled_SNIDs)

            # Now create train/test/validation indices
            if settings.data_training:
                SNID_train = sampled_SNIDs[: int(0.99 * n_samples)]
                SNID_val = sampled_SNIDs[int(0.99 * n_samples): int(0.995 * n_samples)]
                SNID_test = sampled_SNIDs[int(0.995 * n_samples):]
            elif settings.data_testing:
                SNID_val = sampled_SNIDs[: int(0.99 * n_samples)]
                SNID_train = sampled_SNIDs[int(0.99 * n_samples): int(0.995 * n_samples)]
                SNID_test = sampled_SNIDs[int(0.995 * n_samples):]
            else:
                SNID_train = sampled_SNIDs[: int(0.8 * n_samples)]
                SNID_val = sampled_SNIDs[int(0.8 * n_samples): int(0.9 * n_samples)]
                SNID_test = sampled_SNIDs[int(0.9 * n_samples):]

            # Find the indices of our train test val splits
            idxs_train = np.where(np.in1d(all_SNIDs, SNID_train))[0]
            idxs_val = np.where(np.in1d(all_SNIDs, SNID_val))[0]
            idxs_test = np.where(np.in1d(all_SNIDs, SNID_test))[0]

            # Create a new column that will state to which data split
            # a given SNID will be assigned
            # train: 0, val: 1, test:2 others: -1
            arr_dataset = -np.ones(len(df)).astype(int)
            arr_dataset[idxs_train] = 0
            arr_dataset[idxs_val] = 1
            arr_dataset[idxs_test] = 2

            df[f"dataset_{dataset}_{nb_classes}classes"] = arr_dataset

            # Display classes balancing in the dataset, for each split
            logging_utils.print_bright("Dataset composition")
            for split_name, idxs in zip(
                ["Training", "Validation", "Test"], [idxs_train, idxs_val, idxs_test]
            ):
                # We count the number of occurence in each class and each split with
                # pandas.value_counts
                d_occurences = (
                    df[f"target_{nb_classes}classes"]
                    .iloc[idxs]
                    .value_counts()
                    .sort_values()
                    .to_dict()
                )
                d_occurences_SNTYPE = (
                    df["SNTYPE"].iloc[idxs].value_counts().sort_values().to_dict()
                )
                total_samples = sum(d_occurences.values())
                total_samples_str = logging_utils.str_to_yellowstr(total_samples)

                str_ = f"# samples {total_samples_str} "
                for c_, n_samples in d_occurences.items():
                    class_str = logging_utils.str_to_yellowstr(c_)
                    class_fraction = f"{100 *(n_samples/total_samples):.2g}%"
                    class_fraction_str = logging_utils.str_to_yellowstr(class_fraction)
                    str_ += f"Class {class_str}: {class_fraction_str} samples "

                list_stat.append(
                    [
                        dataset,
                        nb_classes,
                        split_name,
                        total_samples,
                        d_occurences,
                        d_occurences_SNTYPE,
                    ]
                )

                logging_utils.print_green(f"{split_name} set", str_)

    # Save to pickle
    df.to_pickle(f"{settings.processed_dir}/{settings.data_prefix}_SNID.pickle")

    # Save stats for publication
    df_stats = pd.DataFrame(
        np.array(list_stat),
        columns=[
            "dataset",
            "nb_classes",
            "split",
            "total_samples",
            "occurences_per_class",
            "ocurrences_per_SNTYPE",
        ],
    )
    df_stats.to_csv(
        os.path.join(settings.stats_dir, f"{settings.data_prefix}_data_stats.csv"),
        index=False,
    )
    paper_df = pd.DataFrame()
    for dataset in df_stats["dataset"].unique():
        for classes in df_stats["nb_classes"].unique():
            paper_df[f"{dataset} {classes} classes"] = (
                df_stats[
                    (df_stats["dataset"] == dataset)
                    & (df_stats["nb_classes"] == classes)
                ]["ocurrences_per_SNTYPE"]
                .apply(pd.Series)
                .sum()
            )
    keep_cols = ["saltfit 2 classes", "photometry 2 classes"]
    paper_df["SN"] = np.array([settings.sntypes[i] for i in paper_df.index])
    paper_df.index = paper_df["SN"]
    paper_df = paper_df[keep_cols]
    paper_df = paper_df.sort_index()
    # save to
    with open(
        os.path.join(settings.latex_dir, f"{settings.data_prefix}_data_stats.tex"), "w"
    ) as tf:
        tf.write(paper_df.to_latex())

    logging_utils.print_green("Done")


def process_single_FITS(file_path, settings):
    """
    Carry out preprocessing on FITS file and save results to pickle.
    Pickle is preferred to csv as it is faster to read and write.

    - Join column from header files
    - Select columns that will be useful laer on
    - Compute SNID to tag each light curve
    - Compute delta times between measures
    - Filter preprocessing
    - Removal of delimiter rows

    Args:
        file_path (str): path to ``.FITS`` file
        settings (ExperimentSettings): controls experiment hyperparameters

    """
    # Load the PHOT file
    df = data_utils.load_pandas_from_fit(file_path)
    # Last line may be a line with MJD = -777.
    # Remove it so that it does not interfere with arr_ID below
    if df.MJD.values[-1] == -777.0:
        df = df.drop(df.index[-1])
    # Keep only columns of interest
    keep_col = ["MJD", "FLUXCAL", "FLUXCALERR", "FLT"]
    df = df[keep_col].copy()

    # Load the companion HEAD file
    header = Table.read(file_path.replace("PHOT", "HEAD"), format="fits")
    df_header = header.to_pandas()
    # Keep only columns of interest
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
    df_header = df_header[keep_col_header].copy()
    df_header["SNID"] = df_header["SNID"].astype(np.int32)
    #############################################
    # Compute SNID for df and join with df_header
    #############################################
    arr_ID = np.zeros(len(df), dtype=np.int32)
    # New light curves are identified by MJD == -777.0
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
    # Drop the delimiter lines
    df = df[df.MJD != -777.000]
    # Reset the index (it is no longer continuous after dropping lines)
    df.reset_index(inplace=True, drop=True)
    # Add delta time
    df = data_utils.compute_delta_time(df)
    # Remove rows post large delta time in the same light curve(delta_time > 150)
    df = data_utils.remove_data_post_large_delta_time(df)

    #############################################
    # Add class and dataset information
    #############################################
    df_SNID = pd.read_pickle(
        f"{settings.processed_dir}/{settings.data_prefix}_SNID.pickle"
    )
    # Check all SNID in df are in df_SNID
    assert np.all(np.in1d(df.SNID.values, df_SNID.SNID.values))
    # Merge left on df: len(df) will not change and will now include
    # relevant columns from df_SNID
    merge_columns = ["SNID"]
    for c_ in [2, 3, len(settings.sntypes.keys())]:
        merge_columns += [f"target_{c_}classes"]
        for dataset in ["photometry", "saltfit"]:
            merge_columns += [f"dataset_{dataset}_{c_}classes"]
    df = df.merge(df_SNID[merge_columns], on=["SNID"], how="left")

    # Save for future use
    basename = os.path.basename(file_path)
    df.to_pickle(f"{settings.preprocessed_dir}/{basename.replace('.FITS', '.pickle')}")

    # getting SNIDs for SNe with Host_spec
    host_spe = df[df["HOSTGAL_SPECZ"] > 0]["SNID"].unique().tolist()

    return host_spe


def preprocess_data(settings):
    """Preprocess the FITS data

    - Use multiprocessing/threading to speed up data processing
    - Preprocess every FIT file in the raw data dir
    - Also save a DataFrame of Host Spe for publication plots

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters

    """

    # Get the list of FITS files
    list_files = natsorted(
        glob.glob(os.path.join(settings.raw_dir, f"{settings.data_prefix}*PHOT.FITS"))
    )
    logging_utils.print_green("List to preprocess ", list_files)
    # Parameters of multiprocessing below
    max_workers = multiprocessing.cpu_count()
    parallel_fn = partial(process_single_FITS, settings=settings)

    # Split list files in chunks of size 10 or less
    # to get a progress bar and alleviate memory constraints
    num_elem = len(list_files)
    num_chunks = num_elem // 10 + 1
    list_chunks = np.array_split(np.arange(num_elem), num_chunks)
    # Loop over chunks of files
    host_spe_tmp = []
    for chunk_idx in tqdm(list_chunks, desc="Preprocess", ncols=100):
        # Process each file in the chunk in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            start, end = chunk_idx[0], chunk_idx[-1] + 1
            # Need to cast to list because executor returns an iterator
            host_spe_tmp += list(executor.map(parallel_fn, list_files[start:end]))
    # Save host spe for plotting and performance tests
    host_spe = [item for sublist in host_spe_tmp for item in sublist]
    pd.DataFrame(host_spe, columns=["SNID"]).to_pickle(
        f"{settings.processed_dir}/hostspe_SNID.pickle"
    )
    logging_utils.print_green("Finished preprocessing")


def pivot_dataframe_single(filename, settings):
    """
    Carry out pivot: we will group time-wise close observations on the same row
    and each row in the dataframe will show a value for each of the flux and flux
    error column

    - All observations withing 8 hours of each other are assigned the same MJD
    - Results are cached with pickle

    Args:
        filename (str): path to a ``.pickle`` file containing pre-processed data
        settings (ExperimentSettings): controls experiment hyperparameters

    """
    list_filters = data_utils.FILTERS

    assert len(list_filters) > 0

    df = pd.read_pickle(filename)
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

    # Some filters (i, r, g, z) may appear multiple times with the same grouped MJD within same light curve
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
    class_columns = []
    for c_ in [2, 3, len(settings.sntypes.keys())]:
        class_columns += [f"target_{c_}classes"]
        for dataset in ["photometry", "saltfit"]:
            class_columns += [f"dataset_{dataset}_{c_}classes"]

    group_features_list = (
        [
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
        ]
        + [k for k in df.keys() if "HOST" in k]
        + class_columns
    )
    # Pivot so that for a given MJD, we have info on all available fluxes / error
    df = pd.pivot_table(df, index=group_features_list, columns=["FLT"])

    # Flatten columns
    df.columns = ["_".join(col).strip() for col in df.columns.values]
    # Reset index to get grouped_MJD and target as columns
    cols_to_reset_list = [c for c in df.index.names if c != "SNID"]
    df.reset_index(cols_to_reset_list, inplace=True)
    # Rename grouped_MJD to MJD
    df.rename(columns={"grouped_MJD": "MJD"}, inplace=True)

    # New column to indicate which channel (r,g,z,i) is present
    # The column will read ``rg`` if r,g are present; ``rgz`` if r,g and z are present, etc.
    for flt in list_filters:
        df[flt] = np.where(df["FLUXCAL_%s" % flt].isnull(), "", flt)
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

    # Cast columns to float32, int32 to save space
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
        elif "classes" in c and df[c].dtype == np.int64:
            df[c] = df[c].astype(np.int8)

    # Add some extra columns from the FITOPT file
    df_salt = data_utils.load_fitfile(settings, verbose=False).set_index("SNID")
    df = df.join(df_salt[["mB", "c", "x1"]], how="left")

    df.drop(columns="MJD", inplace=True)

    # Save to pickle
    dump_filename = os.path.splitext(filename)[0] + "_pivot.pickle"
    df.to_pickle(dump_filename)


def pivot_dataframe_batch(list_files, settings):
    """
    - Use multiprocessing/threading to speed up data processing
    - Pivot every file in list_files and cache the result with pickle

    Args:
        list_files (list): list of ``.pickle`` files containing pre-processed data
        settings (ExperimentSettings): controls experiment hyperparameters

    """
    # Split list files in chunks of size 10 or less
    # to get a progress bar and alleviate memory constraints
    num_elem = len(list_files)
    num_chunks = num_elem // 10 + 1
    list_chunks = np.array_split(np.arange(num_elem), num_chunks)

    # Parameters of multiprocessing below
    max_workers = multiprocessing.cpu_count()
    # Loop over chunks of files
    for chunk_idx in tqdm(list_chunks, desc="Pivoting dataframes", ncols=100):
        parallel_fn = partial(pivot_dataframe_single, settings=settings)
        # Process each file in the chunk in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            start, end = chunk_idx[0], chunk_idx[-1] + 1
            executor.map(parallel_fn, list_files[start:end])

    logging_utils.print_green("Finished pivot")


@logging_utils.timer("Data processing")
def make_dataset(settings):
    """Main function for data processing

    - Create the train test val splits
    - Preprocess all the FITs data, then pivot
    - Save all of the processed data to a single HDF5 database

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
    """

    # Clean up data folders
    if settings.overwrite is True:
        for folder in [settings.preprocessed_dir, settings.processed_dir]:
            # Dont throw error if folder exists with exist_ok Flag.
            for f in glob.glob(f"{folder}/{settings.data_prefix}*"):
                os.remove(f)

    # split dataset in train test and validation
    build_traintestval_splits(settings)

    # Preprocess dataset
    preprocess_data(settings)

    # Pivot dataframe
    list_files = natsorted(
        glob.glob(f"{settings.preprocessed_dir}/{settings.data_prefix}*PHOT*")
    )
    pivot_dataframe_batch(list_files, settings)

    # Aggregate the pivoted dataframe
    list_files = natsorted(
        glob.glob(
            os.path.join(
                settings.preprocessed_dir, f"{settings.data_prefix}*pivot.pickle"
            )
        )
    )
    logging_utils.print_green("Concatenating pivot")
    df = pd.concat([pd.read_pickle(f) for f in list_files], axis=0)

    # Save to HDF5
    data_utils.save_to_HDF5(settings, df)

    # Save plots to visualize the distribution of some of the data features
    SNinfo_df = data_utils.load_HDF5_SNinfo(settings)
    datasets_plots(SNinfo_df, settings)

    logging_utils.print_green("Finished making dataset")
