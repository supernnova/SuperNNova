import os
import glob
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
from supernnova.paper.superNNova_plots import datasets_plots

from constants import SNTYPES, LIST_FILTERS, OFFSETS, OFFSETS_STR, FILTER_DICT


# def build_data_splits(config):
#     """Build dataset split in the following way

#     - Downsample each class so that it has the same cardinality as the lowest cardinality class
#     - Randomly assign lightcurves to a 80/10/10 train test val split (except Out-of-distribution data 1/1/98)

#     OOD:
#         Will use the complete sample for testing, does not require settings.

#     Args:
#         settings (ExperimentSettings): controls experiment hyperparameters
#     """

#     raw_dir = config["raw_dir"]
#     processed_dir = config["raw_dir"]
#     raw_format = config["raw_format"]
#     fitopt_file = config["fitopt_file"]
#     max_workers = max(1, multiprocessing.cpu_count() - 2)
#     photo_columns = ["SNID"] + [
#         f"target_{nb_classes}classes" for nb_classes in list([2, len(sntypes)])
#     ]

#     logging_utils.print_green(f"Computing splits")

#     # Get list of data files
#     list_files = natsorted(Path(raw_dir).glob(f"*HEAD.{raw_format}*"))
#     print("List files", "\n".join(list_files))

#     # Load files into pandas Dataframes
#     process_fn = partial(
#         data_utils.process_header, sntypes=sntypes, columns=photo_columns + ["SNTYPE"]
#     )
#     pool = multiprocessing.Pool(max_workers)
#     list_df = pool.map(process_fn, list_files)
#     pool.close()

#     # Load df_photo
#     df_photo = pd.concat(list_df)
#     df_photo["SNID"] = df_photo["SNID"].astype(int)

#     # load FITOPT file on which we will base our splits
#     if fitopt_file is not None:
#         df_salt = data_utils.load_fitfile(fitopt_file, sntypes)
#     else:
#         df_salt["SNID"] = df_photo["SNID"].copy()
#     df_salt["is_salt"] = 1

#     # Check all SNID in df_salt are also in df_photo
#     assert np.all(np.in1d(df_salt.SNID.values, df_photo.SNID.values))

#     # Merge left on df_photo
#     df = df_photo.merge(df_salt[["SNID", "is_salt"]], on=["SNID"], how="left")
#     # Some curves are in photo and not in salt, these curves have is_salt = NaN
#     # We replace the NaN with 0
#     df["is_salt"] = df["is_salt"].fillna(0).astype(int)

#     # Save dataset stats
#     list_stat = []

#     # Save a dataframe to record train/test/val split for
#     # binary, ternary and all-classes classification
#     for dataset in ["saltfit", "photometry"]:
#         for nb_classes in list([2, len(sntypes.keys())]):
#             logging_utils.print_green(
#                 f"Computing {dataset} splits for {nb_classes}-way classification"
#             )
#             # Randomly sample SNIDs such that all class have the same number of occurences
#             if dataset == "saltfit":
#                 g = df[df.is_salt == 1].groupby(f"target_{nb_classes}classes")
#             else:
#                 g = df.groupby(f"target_{nb_classes}classes")

#             # Line below: we have grouped df by target, we find out which of those
#             # group has the smallest size with g.size().min(), then we sample randomly
#             # from this group and reset the index. We then sample with frac=1 to shuffle
#             # the whole dataset. Otherwise, the classes are sorted and the train/test/val
#             # splits are incorrect.
#             if settings.data_testing:
#                 # when just classifying data balancing is not necessary
#                 g = g.apply(lambda x: x).reset_index(drop=True).sample(frac=1)
#             else:
#                 g = (
#                     g.apply(lambda x: x.sample(g.size().min()))
#                     .reset_index(drop=True)
#                     .sample(frac=1)
#                 )

#             all_SNIDs = df.SNID.values
#             sampled_SNIDs = g["SNID"].values
#             n_samples = len(sampled_SNIDs)

#             # Now create train/test/validation indices
#             if settings.data_training:
#                 SNID_train = sampled_SNIDs[: int(0.99 * n_samples)]
#                 SNID_val = sampled_SNIDs[int(0.99 * n_samples) : int(0.995 * n_samples)]
#                 SNID_test = sampled_SNIDs[int(0.995 * n_samples) :]
#             elif settings.data_testing:
#                 SNID_test = sampled_SNIDs[:]
#                 # the train and val sets wont be used in this case
#                 SNID_train = sampled_SNIDs[0]
#                 SNID_val = sampled_SNIDs[0]
#             else:
#                 SNID_train = sampled_SNIDs[: int(0.8 * n_samples)]
#                 SNID_val = sampled_SNIDs[int(0.8 * n_samples) : int(0.9 * n_samples)]
#                 SNID_test = sampled_SNIDs[int(0.9 * n_samples) :]

#             # Find the indices of our train test val splits
#             idxs_train = np.where(np.in1d(all_SNIDs, SNID_train))[0]
#             idxs_val = np.where(np.in1d(all_SNIDs, SNID_val))[0]
#             idxs_test = np.where(np.in1d(all_SNIDs, SNID_test))[0]

#             # Create a new column that will state to which data split
#             # a given SNID will be assigned
#             # train: 0, val: 1, test:2 others: -1
#             arr_dataset = -np.ones(len(df)).astype(int)
#             arr_dataset[idxs_train] = 0
#             arr_dataset[idxs_val] = 1
#             arr_dataset[idxs_test] = 2

#             df[f"dataset_{dataset}_{nb_classes}classes"] = arr_dataset

#             # Display classes balancing in the dataset, for each split
#             logging_utils.print_bright("Dataset composition")
#             for split_name, idxs in zip(
#                 ["Training", "Validation", "Test"], [idxs_train, idxs_val, idxs_test]
#             ):
#                 # We count the number of occurence in each class and each split with
#                 # pandas.value_counts
#                 d_occurences = (
#                     df[f"target_{nb_classes}classes"]
#                     .iloc[idxs]
#                     .value_counts()
#                     .sort_values()
#                     .to_dict()
#                 )
#                 d_occurences_SNTYPE = (
#                     df["SNTYPE"].iloc[idxs].value_counts().sort_values().to_dict()
#                 )
#                 total_samples = sum(d_occurences.values())
#                 total_samples_str = logging_utils.str_to_yellowstr(total_samples)

#                 str_ = f"# samples {total_samples_str} "
#                 for c_, n_samples in d_occurences.items():
#                     class_str = logging_utils.str_to_yellowstr(c_)
#                     class_fraction = f"{100 *(n_samples/total_samples):.2g}%"
#                     class_fraction_str = logging_utils.str_to_yellowstr(class_fraction)
#                     str_ += f"Class {class_str}: {class_fraction_str} samples "

#                 list_stat.append(
#                     [
#                         dataset,
#                         nb_classes,
#                         split_name,
#                         total_samples,
#                         d_occurences,
#                         d_occurences_SNTYPE,
#                     ]
#                 )

#                 logging_utils.print_green(f"{split_name} set", str_)
#     # Save to pickle
#     df.to_pickle(f"{processed_dir}/SNID.pickle")

#     logging_utils.print_green("Done")


def process_phot_file(file_path, preprocessed_dir, list_filters):
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

    # TODO
    # #############################################
    # # Photometry window init
    # #############################################
    # if settings.photo_window_files:
    #     if Path(settings.photo_window_files[0]).exists():
    #         # load fits file
    #         df_peak = pd.read_csv(
    #             settings.photo_window_files[0],
    #             comment="#",
    #             delimiter=" ",
    #             skipinitialspace=True,
    #         )
    #         df_peak["SNID"] = df_peak["CID"].astype(int)
    #         try:
    #             df_peak = df_peak[["SNID", settings.photo_window_var]]
    #         except Exception:
    #             logging_utils.print_red("Provide a correct photo_window variable")
    #             raise Exception
    #         # merge with header
    #         df_header = pd.merge(df_header, df_peak, on="SNID")
    #     else:
    #         logging_utils.print_red("Provide a valid photo_window_file")

    #############################################
    # Compute SNID for df and join with df_header
    #############################################

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

    # TODO
    # #############################################
    # # Photometry window selection
    # #############################################
    # if settings.photo_window_files:
    #     df["window_time_cut"] = True
    #     mask = df["MJD"] != -777.00
    #     df["window_delta_time"] = df["MJD"] - df[settings.photo_window_var]
    #     df.loc[mask, "window_time_cut"] = df["window_delta_time"].apply(
    #         lambda x: True
    #         if (x > 0 and x < settings.photo_window_max)
    #         else (True if (x <= 0 and x > settings.photo_window_min) else False)
    #     )
    #     df = df[df["window_time_cut"] == True]

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

    # TODO photometry / saltfit /nb classes

    # Save for future use
    basename = os.path.basename(file_path)
    df.to_pickle(f"{preprocessed_dir}/{basename.replace('.FITS', '.pickle')}")

    # getting SNIDs for SNe with Host_spec
    host_spe = df[df["HOSTGAL_SPECZ"] > 0]["SNID"].unique().tolist()

    return host_spe


def preprocess_data(config):
    """Preprocess the FITS data

    - Use multiprocessing/threading to speed up data processing
    - Preprocess every FIT file in the raw data dir
    - Also save a DataFrame of Host Spe for publication plots

    Args:
        config (dict): experiment hyperparameters

    """

    raw_dir = config["raw_dir"]
    processed_dir = config["raw_dir"]
    raw_format = config["raw_format"]
    preprocessed_dir = config["preprocessed_dir"]
    max_workers = max(1, multiprocessing.cpu_count() - 2)

    logging_utils.print_green(f"Computing splits")

    # Get the list of FITS files
    # TODO support csv
    list_files = natsorted(map(str, Path(raw_dir).glob(f"*PHOT.{raw_format}*")))
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


def pivot_dataframe_single(filename, list_filters, fitopt_file, sntypes):
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

    # TODO class columns removed

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

    # load FITOPT file on which we will base our splits
    if fitopt_file is not None:
        df_salt = data_utils.load_fitfile(fitopt_file, sntypes)
    else:
        # if no fits file we populate with dummies
        # logging_utils.print_yellow(f"Creating dummy mB,c,x1")
        df_salt = pd.DataFrame()
        df_salt["SNID"] = np.array(df.index.unique())
        df_salt["mB"] = np.zeros(len(df.index.unique()))
        df_salt["c"] = np.zeros(len(df.index.unique()))
        df_salt["x1"] = np.zeros(len(df.index.unique()))

    df = df.merge(df_salt[["SNID", "mB", "c", "x1"]], on="SNID", how="left")

    df.drop(columns="MJD", inplace=True)
    # Save to pickle
    dump_filename = os.path.splitext(filename)[0] + "_pivot.pickle"
    df.to_pickle(dump_filename)


def pivot_dataframe_batch(list_files, config):
    """
    - Use multiprocessing/threading to speed up data processing
    - Pivot every file in list_files and cache the result with pickle

    Args:
        list_files (list): list of ``.pickle`` files containing pre-processed data
        settings (ExperimentSettings): controls experiment hyperparameters

    """

    fitopt_file = config["fitopt_file"]
    max_workers = max(1, multiprocessing.cpu_count() - 2)

    pool = multiprocessing.Pool(max_workers)
    n_files = len(list_files)
    chunk_size = min(len(list_files), 10)

    process_fn = partial(
        pivot_dataframe_single,
        list_filters=LIST_FILTERS,
        fitopt_file=fitopt_file,
        sntypes=SNTYPES,
    )

    process_fn(list_files[0])

    # Loop over chunks of files
    for idx in tqdm(range(0, n_files, chunk_size), desc="Preprocess", ncols=100):
        # Process each file in the chunk in parallel
        pool.map(process_fn, list_files[idx : idx + chunk_size])

    logging_utils.print_green("Finished pivot")


@logging_utils.timer("Data processing")
def make_dataset(config_path):

    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    preprocessed_dir = config["preprocessed_dir"]
    processed_dir = config["processed_dir"]
    hdf5_file = config["hdf5_file"]

    # Clean up data folders
    for folder in [preprocessed_dir, processed_dir]:
        if config.get("overwrite", True):
            shutil.rmtree(folder, ignore_errors=True)
        Path(folder).mkdir(exist_ok=True, parents=True)

    # # split dataset in train test and validation
    # build_data_splits(config)

    # Preprocess dataset
    preprocess_data(config)

    # Pivot dataframe
    list_files = natsorted(map(str, Path(f"{preprocessed_dir}").glob("*PHOT*")))
    pivot_dataframe_batch(list_files, config)

    # Aggregate the pivoted dataframe
    list_files = natsorted(map(str, Path(f"{preprocessed_dir}").glob("*pivot.pickle*")))
    logging_utils.print_green("Concatenating pivot")

    df = pd.concat([pd.read_pickle(f) for f in list_files], axis=0).reset_index(
        drop=True
    )

    # Save to HDF5
    data_utils.save_to_HDF5(
        df, hdf5_file, SNTYPES, LIST_FILTERS, OFFSETS, OFFSETS_STR, FILTER_DICT
    )

    # Save plots to visualize the distribution of some of the data features
    try:
        SNinfo_df = data_utils.load_HDF5_SNinfo(settings)
        datasets_plots(SNinfo_df, settings)
    except Exception:
        logging_utils.print_yellow(
            "Warning: can't do data plots if no saltfit for this dataset"
        )

    # Clean preprocessed directory
    shutil.rmtree(preprocessed_dir)

    logging_utils.print_green("Finished making dataset")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dataset building")
    parser.add_argument("config_path", help="Path to yml config gile")

    args = parser.parse_args()

    make_dataset(args.config_path)
