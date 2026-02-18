import os
import re
import json
import glob
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from natsort import natsorted
from functools import partial
from astropy.table import Table
from concurrent.futures import ProcessPoolExecutor

from collections import OrderedDict

from ..utils import data_utils
from ..utils import logging_utils
from ..paper.superNNova_plots import datasets_plots


def process_fn(inputs):
    fn, fil = inputs
    return fn(fil)


def powers_of_two(x):
    powers = []
    i = 1
    while i <= x:
        if i & x:
            powers.append(i)
        i <<= 1
    return powers


def build_traintestval_splits(settings):
    """Build dataset split in the following way

    - Downsample each class so that it has the same cardinality as the lowest cardinality class
    - Randomly assign lightcurves to a 80/10/10 train test val split (except Out-of-distribution data 1/1/98)

    OOD:
        Will use the complete sample for testing, does not require settings.

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
    """

    logging_utils.print_green("Computing splits")

    # Read and process files faster with ProcessPoolExecutor
    max_workers = multiprocessing.cpu_count() - 2
    photo_columns = ["SNID"] + [
        f"target_{nb_classes}classes"
        # for nb_classes in list(set([2, len(settings.sntypes.keys())]))
        for nb_classes in list(
            set([2, len(set([k for k in dict(settings.sntypes).values()]))])
        )
    ]

    # Load headers
    # either in HEAD.FITS or csv format
    list_files = natsorted(Path(settings.raw_dir).glob("**/*HEAD*"))
    list_fmt = [re.search(r"(FITS|csv)", fil.name).group() for fil in list_files]
    list_files = [str(fil) for fil in list_files]

    print("List files", list_files)
    # use parallelization to speed up processing
    if not settings.debug:
        process_fn_FITS = partial(
            data_utils.process_header_FITS,
            settings=settings,
            columns=photo_columns + [settings.sntype_var],
        )
        process_fn_csv = partial(
            data_utils.process_header_csv,
            settings=settings,
            columns=photo_columns + [settings.sntype_var],
        )

        list_fn = []
        for fmt in list_fmt:
            if fmt == "csv":
                list_fn.append(process_fn_csv)
            elif fmt == "FITS":
                list_fn.append(process_fn_FITS)

        list_pairs = list(zip(list_fn, list_files))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            list_df = executor.map(process_fn, list_pairs)

    else:
        logging_utils.print_yellow("Beware debugging mode (loop over files)")
        list_df = []
        for fil, fmat in zip(list_files, list_fmt):
            if fmat == "FITS":
                list_df.append(
                    data_utils.process_header_FITS(
                        fil, settings, columns=photo_columns + [settings.sntype_var]
                    )
                )
            else:
                list_df.append(
                    data_utils.process_header_csv(
                        fil, settings, columns=photo_columns + [settings.sntype_var]
                    )
                )

    # Load df_photo
    df_photo = pd.concat(list_df)
    df_photo["SNID"] = df_photo["SNID"].astype(str).str.strip()
    # load FITOPT file on which we will base our splits
    df_salt = data_utils.load_fitfile(settings)
    if len(df_salt) < 1:
        # if no fits file we include all lcs
        logging_utils.print_yellow("All lcs used for salt and photometry samples")
        df_salt = pd.DataFrame()
        df_salt["SNID"] = df_photo["SNID"].str.strip()
    df_salt["is_salt"] = 1
    # correct format SNID
    df_salt["SNID"] = df_salt["SNID"].astype(str).str.strip()
    # Check all SNID in df_salt are also in df_photo
    try:
        assert np.all(df_salt.SNID.isin(df_photo.SNID))
    except Exception:
        logging_utils.print_red(
            " BEWARE! This is not the fits file for this photometry "
        )
        print(logging_utils.str_to_redstr("   do point at the correct --fits_dir "))
        print(
            logging_utils.str_to_redstr(
                "   (cheat: or an empty folder to override use of salt2fits) "
            )
        )
        import sys

        sys.exit(1)
    # Merge left on df_photo
    df = df_photo.merge(df_salt[["SNID", "is_salt"]], on=["SNID"], how="left")
    # Drop duplicates in case of multiple entries for the same SNID
    df = df.drop_duplicates(subset="SNID")
    # Some curves are in photo and not in salt, these curves have is_salt = NaN
    # We replace the NaN with 0
    df["is_salt"] = df["is_salt"].fillna(0).astype(int)

    # Save dataset stats
    list_stat = []

    # Save a dataframe to record train/test/val split for
    # binary, ternary and all-classes classification
    for dataset in ["saltfit", "photometry"]:
        for nb_classes in list(
            set([2, len(set([k for k in dict(settings.sntypes).values()]))])
        ):
            logging_utils.print_green(
                f"Computing {dataset} splits for {nb_classes}-way classification"
            )
            # Randomly sample SNIDs such that all class have the same number of occurences
            if dataset == "saltfit":
                g = df[df.is_salt == 1].groupby(
                    f"target_{nb_classes}classes", group_keys=False
                )
            else:
                g = df.groupby(f"target_{nb_classes}classes", group_keys=False)
            dic_targets = (
                g[settings.sntype_var].apply(lambda x: list(np.unique(x))).to_dict()
            )
            print(f"target {settings.sntype_var}")
            settings.data_types_training = [
                (
                    f"{k} {settings.sntypes[v[0]]} {[int(dt) for dt in dic_targets[k]]}"
                    if v[0] in settings.sntypes.keys()
                    else f"{k} other {[int(dt) for dt in dic_targets[k]]}"
                )
                for k, v in dic_targets.items()
            ]
            print(settings.data_types_training)

            if settings.testing_ids:
                if Path(settings.testing_ids).suffix == ".csv":
                    df_ids_test = pd.read_csv(settings.testing_ids)
                    try:
                        ids_test = df_ids_test["SNID"].astype(str).values
                    except Exception:
                        logging_utils.print_red(
                            f"Provide a {settings.testing_ids} with SNID column"
                        )
                        raise ValueError
                elif Path(settings.testing_ids).suffix == ".npy":
                    ids_test = np.load(settings.testing_ids)
                    ids_test = [f"{k}" for k in ids_test]
                else:
                    logging_utils.print_red("Provide a csv or numpy testing_ids file")
                    raise ValueError

                g_wo_test = df[~df.SNID.isin(ids_test)].groupby(
                    f"target_{nb_classes}classes", group_keys=False
                )
                g_test = df[df.SNID.isin(ids_test)].groupby(
                    f"target_{nb_classes}classes", group_keys=False
                )
            # Line below: we have grouped df by target, we find out which of those
            # group has the smallest size with g.size().min(), then we sample randomly
            # from this group and reset the index. We then sample with frac=1 to shuffle
            # the whole dataset. Otherwise, the classes are sorted and the train/test/val
            # splits are incorrect.
            if settings.data_testing:
                # when just classifying data balancing is not necessary
                g = g.apply(lambda x: x).reset_index(drop=True).sample(frac=1)
            elif settings.testing_ids:
                g_wo_test = (
                    g_wo_test.apply(lambda x: x).reset_index(drop=True).sample(frac=1)
                )
                g_test = g_test.apply(lambda x: x).reset_index(drop=True).sample(frac=1)
            else:
                g = (
                    g.apply(lambda x: x.sample(g.size().min()), include_groups=False)
                    .reset_index(drop=True)
                    .sample(frac=1)
                )
            if settings.testing_ids:
                sampled_SNIDs_wo_test = g_wo_test["SNID"].values
                n_samples = len(sampled_SNIDs_wo_test)
                SNID_train = sampled_SNIDs_wo_test[: int(0.9 * n_samples)]
                SNID_val = sampled_SNIDs_wo_test[int(0.9 * n_samples) : int(n_samples)]
                sampled_SNIDs_test = g_test["SNID"].values
                SNID_test = sampled_SNIDs_test[:]
            else:
                sampled_SNIDs = g["SNID"].values
                n_samples = len(sampled_SNIDs)
                # Now create train/test/validation indices
                if settings.data_training:
                    SNID_train = sampled_SNIDs[: int(0.99 * n_samples)]
                    SNID_val = sampled_SNIDs[
                        int(0.99 * n_samples) : int(0.995 * n_samples)
                    ]
                    SNID_test = sampled_SNIDs[int(0.995 * n_samples) :]
                elif settings.data_testing:
                    SNID_test = sampled_SNIDs[:]
                    # the train and val sets wont be used in this case
                    SNID_train = [sampled_SNIDs[0]]
                    SNID_val = [sampled_SNIDs[0]]
                else:
                    SNID_train = sampled_SNIDs[: int(0.8 * n_samples)]
                    SNID_val = sampled_SNIDs[
                        int(0.8 * n_samples) : int(0.9 * n_samples)
                    ]
                    SNID_test = sampled_SNIDs[int(0.9 * n_samples) :]

            # Find the indices of our train test val splits
            idxs_train = np.where(df.SNID.isin(SNID_train))[0]
            idxs_val = np.where(df.SNID.isin(SNID_val))[0]
            idxs_test = np.where(df.SNID.isin(SNID_test))[0]
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
                    df[settings.sntype_var]
                    .iloc[idxs]
                    .value_counts()
                    .sort_values()
                    .to_dict()
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
    df.to_pickle(f"{settings.processed_dir}/SNID.pickle", protocol=4)

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

    if len(df) < 1:
        logging_utils.print_red("Do not provide empty photometry file", file_path)
    # Last line may be a line with MJD = -777.
    # Remove it so that it does not interfere with arr_ID below
    if df.MJD.values[-1] == -777.0:
        df = df.drop(df.index[-1])

    # Keep only columns of interest
    keep_col = ["MJD", "FLUXCAL", "FLUXCALERR", "FLT"]
    # BAND and FLT are exchangeable
    if "FLT" not in df.keys() and "BAND" in df.keys():
        df = df.rename(columns={"BAND": "FLT"})
    df = (
        df[keep_col + [settings.phot_reject]].copy()
        if settings.phot_reject
        else df[keep_col].copy()
    )
    # filters have a trailing white space which we remove and convert to string
    df.FLT = df.FLT.apply(
        lambda x: (
            x.decode("utf-8").rstrip() if isinstance(x, bytes) else str(x).rstrip()
        )
    )
    # Load the companion HEAD file
    header = Table.read(file_path.replace("PHOT", "HEAD"), format="fits")
    df_header = header.to_pandas()
    try:
        df_header["SNID"] = df_header["SNID"].str.decode("utf-8")
    except Exception:
        df_header["SNID"] = df_header["SNID"].astype(str)
    # Keep only columns of interest
    # Hack for using the final redshift not the galaxy
    if settings.redshift_label != "none":
        logging_utils.print_yellow("Changed redshift to", settings.redshift_label)
        df_header["HOSTGAL_SPECZ"] = df_header[settings.redshift_label]
        df_header["HOSTGAL_SPECZ_ERR"] = df_header[f"{settings.redshift_label}_ERR"]
        df_header["SIM_REDSHIFT_CMB"] = df_header[settings.redshift_label]

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
        settings.sntype_var,
    ]
    if settings.photo_window_var not in keep_col_header:
        keep_col_header += [settings.photo_window_var]
    if settings.additional_train_var:
        keep_col_header += list(settings.additional_train_var)
    # check if keys are in header
    keep_col_header = [k for k in keep_col_header if k in df_header.keys()]
    df_header = df_header[keep_col_header].copy()
    df_header["SNID"] = df_header["SNID"].astype(str).str.strip()

    #############################################
    # Photometry window init
    #############################################
    if settings.photo_window_files:
        if Path(settings.photo_window_files[0]).exists():
            # load fits file
            df_peak = pd.read_csv(
                settings.photo_window_files[0],
                comment="#",
                delimiter=" ",
                skipinitialspace=True,
            )
            if "SNID" not in df_peak.keys():
                df_peak["SNID"] = df_peak["CID"].astype(str)
            else:
                df_peak["SNID"] = df_peak["SNID"].astype(str)

            try:
                df_peak = df_peak[["SNID", settings.photo_window_var]]
            except Exception:
                logging_utils.print_red("Provide a correct photo_window variable")
                raise Exception
            # merge with header
            df_header_tmp = pd.merge(df_header, df_peak, on="SNID")
            if len(df_header) == len(df_header_tmp):
                df_header = df_header_tmp
            else:
                raise Exception
            if len(df_header) < 1:
                logging_utils.print_red(
                    "Provide a matching photo_window_file (not a common SNID found) "
                )
                raise Exception
        else:
            if settings.photo_window_files[0] == "HEAD":
                # if using a variable from header file
                if settings.photo_window_var in df_header.keys():
                    pass
                else:
                    logging_utils.print_red(
                        "Provide a valid peak key in header or a photo_window_file"
                    )
                    logging_utils.print_red(
                        f"Currently {settings.photo_window_var} {settings.photo_window_files}"
                    )
            else:
                logging_utils.print_red(
                    "Provide a valid peak key in header or a photo_window_file"
                )
                logging_utils.print_red(
                    f"Currently {settings.photo_window_var} {settings.photo_window_files}"
                )
    #############################################
    # Compute SNID for df and join with df_header
    #############################################
    arr_ID = np.chararray(len(df), itemsize=15)
    # New light curves are identified by MJD == -777.0
    arr_idx = np.where(df["MJD"].values == -777.0)[0]
    arr_idx = np.hstack((np.array([0]), arr_idx, np.array([len(df)])))
    # Fill in arr_ID
    for counter in range(1, len(arr_idx)):
        start, end = arr_idx[counter - 1], arr_idx[counter]
        # index starts at zero
        arr_ID[start:end] = df_header.SNID.iloc[counter - 1]
    df["SNID"] = arr_ID.astype(str)
    df["SNID"] = df["SNID"].str.strip()
    df = df.set_index("SNID")
    df_header["SNID"] = df_header["SNID"].str.strip()
    df_header = df_header.set_index("SNID")
    # join df and header
    df = df.join(df_header).reset_index()

    #############################################
    # Photometry window & quality (flag) selection
    #############################################
    # window
    if settings.photo_window_files:
        df["window_time_cut"] = True
        mask = df["MJD"] != -777.00
        df["window_delta_time"] = df["MJD"] - df[settings.photo_window_var]
        df.loc[mask, "window_time_cut"] = df["window_delta_time"].apply(
            lambda x: (
                True
                if (x > 0 and x < settings.photo_window_max)
                else (True if (x <= 0 and x > settings.photo_window_min) else False)
            )
        )
        df = df[df["window_time_cut"]]
    # quality
    if settings.phot_reject:
        # only valid for powers of two combinations
        tmp = len(df.SNID.unique())
        tmp2 = len(df)
        df["phot_reject"] = df[settings.phot_reject].apply(
            lambda x: (
                False
                if len(
                    set(settings.phot_reject_list).intersection(set(powers_of_two(x)))
                )
                > 0
                else True
            )
        )
        df = df[df["phot_reject"]]

        if settings.debug:
            logging_utils.print_blue("Phot reject", file_path)
            logging_utils.print_blue(f"SNID {tmp} to {len(df.SNID.unique())}")
            logging_utils.print_blue(f"Phot {tmp2} to {len(df)}")

    #############################################
    # Miscellaneous data processing
    #############################################
    df = df[keep_col + keep_col_header].copy()

    # keep only filters we are going to use for classification
    df = df[df["FLT"].isin(settings.list_filters)]
    if len(df) < 1:
        logging_utils.print_red(
            "No light curve left after filtering by filters, check your settings"
        )
        raise ValueError

    # Drop the delimiter lines
    df = df[df.MJD != -777.000]
    # Reset the index (it is no longer continuous after dropping lines)
    df = df.reset_index(drop=True)
    # Add delta time
    df = data_utils.compute_delta_time(df)
    # Remove rows post large delta time in the same light curve(delta_time > 150)
    # df = data_utils.remove_data_post_large_delta_time(df)
    #############################################
    # Add class and dataset information
    #############################################
    df_SNID = pd.read_pickle(f"{settings.processed_dir}/SNID.pickle")
    # Check all SNID in df are in df_SNID
    assert np.all(df.SNID.isin(df_SNID.SNID))
    # Merge left on df: len(df) will not change and will now include
    # relevant columns from df_SNID
    merge_columns = ["SNID"]
    # for c_ in list(set([2, len(settings.sntypes.keys())])):
    distinct_classes = len(set([k for k in dict(settings.sntypes).values()]))
    for c_ in list(set([2, distinct_classes])):
        merge_columns += [f"target_{c_}classes"]
        for dataset in ["photometry", "saltfit"]:
            merge_columns += [f"dataset_{dataset}_{c_}classes"]
    df = df.merge(df_SNID[merge_columns], on=["SNID"], how="left")

    # Save for future use
    basename = os.path.basename(file_path)
    folder_name = Path(file_path.split(f"{settings.raw_dir}/")[-1]).parent
    if folder_name != Path("."):
        prefix = str(folder_name).replace("/", "_")
        basename = f"{prefix}_{basename}"

    df.to_pickle(
        f"{settings.preprocessed_dir}/{basename.replace('.FITS', '.pickle').replace('.gz','')}",
        protocol=4,
    )

    # getting SNIDs for SNe with Host_spec
    host_spe = df[df["HOSTGAL_SPECZ"] > 0]["SNID"].unique().tolist()

    print(len(df), "rows in preprocessed file", file_path)
    return host_spe


def process_single_csv(file_path, settings):
    """
    Carry out preprocessing on csv file and save results to pickle.
    Pickle is preferred to csv as it is faster to read and write.

    - Compute delta times between measures
    - Filter preprocessing

    Args:
        file_path (str): path to ``.csv`` file
        settings (ExperimentSettings): controls experiment hyperparameters

    """
    # Load the PHOT file
    df = pd.read_csv(file_path)
    if len(df) < 1:
        logging_utils.print_red("Do not provide empty photometry file", file_path)
        raise ValueError

    # Keep only columns of interest
    keep_col = ["SNID", "MJD", "FLUXCAL", "FLUXCALERR", "FLT"]
    df = df[keep_col].copy()
    df["SNID"] = df["SNID"].astype(str)
    df = df.set_index("SNID")

    # Load the companion HEAD file
    df_header = pd.read_csv(file_path.replace("PHOT", "HEAD"))

    if settings.redshift_label != "none":
        logging_utils.print_yellow("Changed redshift to", settings.redshift_label)
        df_header["HOSTGAL_SPECZ"] = df_header[settings.redshift_label]
        df_header["HOSTGAL_SPECZ_ERR"] = df_header[f"{settings.redshift_label}_ERR"]
        df_header["SIM_REDSHIFT_CMB"] = df_header[settings.redshift_label]

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
        settings.sntype_var,
    ]
    if settings.photo_window_var not in keep_col_header:
        keep_col_header += [settings.photo_window_var]
    if settings.additional_train_var:
        keep_col_header += list(settings.additional_train_var)
        print(f"Adding additional variables to dataset {settings.additional_train_var}")
    # check if keys are in header
    keep_col_header = [k for k in keep_col_header if k in df_header.keys()]
    df_header = df_header[keep_col_header].copy()
    df_header["SNID"] = df_header["SNID"].astype(str)
    df_header["SNID"] = df_header["SNID"].str.strip()
    df_header = df_header.set_index("SNID")
    df = df.join(df_header).reset_index()

    if settings.photo_window_files:
        df["window_time_cut"] = True
        mask = df["MJD"] != -777.00
        df["window_delta_time"] = df["MJD"] - df[settings.photo_window_var]
        df.loc[mask, "window_time_cut"] = df["window_delta_time"].apply(
            lambda x: (
                True
                if (x > 0 and x < settings.photo_window_max)
                else (True if (x <= 0 and x > settings.photo_window_min) else False)
            )
        )
        df = df[df["window_time_cut"] is True]
    # quality
    if settings.phot_reject:

        tmp = len(df.SNID.unique())
        tmp2 = len(df)
        df["phot_reject"] = df[settings.phot_reject].apply(
            lambda x: (
                False
                if len(
                    set(settings.phot_reject_list).intersection(set(powers_of_two(x)))
                )
                > 0
                else True
            )
        )
        df = df[df["phot_reject"] is True]

        if settings.debug:
            logging_utils.print_blue("Phot reject", file_path)
            logging_utils.print_blue(f"SNID {tmp} to {len(df.SNID.unique())}")
            logging_utils.print_blue(f"Phot {tmp2} to {len(df)}")

    #############################################
    # Miscellaneous data processing
    #############################################
    df = df[list(set(keep_col + keep_col_header))].copy()
    # filters have a trailing white space which we remove
    df.FLT = df.FLT.apply(lambda x: x.rstrip()).values.astype(str)
    # keep only filters we are going to use for classification
    df = df[df["FLT"].isin(settings.list_filters)]
    # Drop the delimiter lines
    df = df[df.MJD != -777.000]
    # Reset the index (it is no longer continuous after dropping lines)
    df = df.reset_index(drop=True)
    # Add delta time
    df = data_utils.compute_delta_time(df)
    # Remove rows post large delta time in the same light curve(delta_time > 150)
    # df = data_utils.remove_data_post_large_delta_time(df)

    #############################################
    # Add class and dataset information
    #############################################
    df_SNID = pd.read_pickle(f"{settings.processed_dir}/SNID.pickle")
    # Check all SNID in df are in df_SNID
    assert np.all(df.SNID.isin(df_SNID.SNID))
    # Merge left on df: len(df) will not change and will now include
    # relevant columns from df_SNID
    merge_columns = ["SNID"]
    # for c_ in list(set([2, len(settings.sntypes.keys())])):
    distinct_classes = len(set([k for k in dict(settings.sntypes).values()]))
    for c_ in list(set([2, distinct_classes])):
        merge_columns += [f"target_{c_}classes"]
        for dataset in ["photometry", "saltfit"]:
            merge_columns += [f"dataset_{dataset}_{c_}classes"]
    df = df.merge(df_SNID[merge_columns], on=["SNID"], how="left")

    # Save for future use
    basename = os.path.basename(file_path)
    df.to_pickle(
        f"{settings.preprocessed_dir}/{basename.replace('.csv', '.pickle').replace('.gz','')}",
        protocol=4,
    )

    # getting SNIDs for SNe with Host_spec
    host_spe = (
        df[df["HOSTGAL_SPECZ"] > 0]["SNID"].unique().tolist()
        if "HOSTGAL_SPECZ" in df.keys()
        else []
    )

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
    # Load headers
    # either in HEAD.FITS or csv format
    list_files = natsorted(Path(settings.raw_dir).glob("**/*PHOT*"))
    list_fmt = [re.search(r"(FITS|csv)", fil.name).group() for fil in list_files]
    list_files = [str(fil) for fil in list_files]

    if not settings.debug:
        process_fn_FITS = partial(process_single_FITS, settings=settings)
        process_fn_csv = partial(process_single_csv, settings=settings)

        list_fn = []
        for fmt in list_fmt:
            if fmt == "csv":
                list_fn.append(process_fn_csv)
            elif fmt == "FITS":
                list_fn.append(process_fn_FITS)

    logging_utils.print_green("List to preprocess ", list_files)
    max_workers = multiprocessing.cpu_count() - 2

    host_spe_tmp = []
    # use parallelization to speed up processing
    # Split list files in chunks of size 10 or less
    # to get a progress bar and alleviate memory constraints
    num_elem = len(list_files)
    num_chunks = num_elem // 10 + 1
    list_chunks = np.array_split(np.arange(num_elem), num_chunks)

    # # Loop over chunks of files
    if not settings.debug:
        for chunk_idx in tqdm(list_chunks, desc="Preprocess", ncols=100):
            # Process each file in the chunk in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                start, end = chunk_idx[0], chunk_idx[-1] + 1
                # Need to cast to list because executor returns an iterator
                # host_spe_tmp += list(executor.map(parallel_fn, list_files[start:end]))
                list_pairs = list(zip(list_fn[start:end], list_files[start:end]))
                host_spe_tmp += list(executor.map(process_fn, list_pairs))

    else:
        logging_utils.print_yellow("Beware debugging mode (loop over files)")
        # for debugging only (parallelization needs to be commented)
        for i in range(len(list_files)):
            out = (
                process_single_FITS(list_files[i], settings)
                if "FITS" in list_files[i]
                else process_single_csv(list_files[i], settings)
            )
            host_spe_tmp.append(out)
    # Save host spe for plotting and performance tests
    host_spe = [item for sublist in host_spe_tmp for item in sublist]
    pd.DataFrame(host_spe, columns=["SNID"]).to_pickle(
        f"{settings.processed_dir}/hostspe_SNID.pickle", protocol=4
    )
    logging_utils.print_green("Finished preprocessing")


def pivot_dataframe_single(filename, settings):

    df = pd.read_pickle(filename)
    df = pivot_dataframe_single_from_df(df, settings)

    # Save to pickle
    dump_filename = filename.split(".pickle")[0] + "_pivot.pickle"
    df.to_pickle(dump_filename, protocol=4)


def pivot_dataframe_single_from_df(df, settings):
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
    list_filters = settings.list_filters

    assert len(list_filters) > 0

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
    df = df.drop(columns="PEAKMJDNORM")
    # Add PEAKMJDNORM back to df with a merge on SNID
    df = df.merge(df_PEAKMJDNORM, how="left", on="SNID")
    # drop columns that won"t be used onwards
    df = df.drop(columns=["MJD", "delta_time"])
    class_columns = []
    # for c_ in list(set([2, len(settings.sntypes.keys())])):
    distinct_classes = len(set([k for k in dict(settings.sntypes).values()]))
    for c_ in list(set([2, distinct_classes])):
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
            settings.sntype_var,
            "SIM_PEAKMAG_z",
            "SIM_PEAKMAG_g",
            "SIM_PEAKMAG_r",
            "SIM_PEAKMAG_i",
        ]
        + [k for k in df.keys() if "HOST" in k]
        + class_columns
    )
    if settings.photo_window_var not in group_features_list:
        group_features_list += [settings.photo_window_var]
    if settings.additional_train_var:
        group_features_list += list(settings.additional_train_var)
    # check if keys are in header
    group_features_list = [k for k in group_features_list if k in df.keys()]
    # Pivot so that for a given MJD, we have info on all available fluxes / error
    df = pd.pivot_table(df, index=group_features_list, columns=["FLT"])

    # Flatten columns
    df.columns = ["_".join(col).strip() for col in df.columns.values]
    # Reset index to get grouped_MJD and target as columns
    cols_to_reset_list = [c for c in df.index.names if c != "SNID"]
    df = df.reset_index(cols_to_reset_list)
    # Rename grouped_MJD to MJD
    df = df.rename(columns={"grouped_MJD": "MJD"})

    # New column to indicate which channel (r,g,z,i) is present
    # The column will read ``rg`` if r,g are present; ``rgz`` if r,g and z are present, etc.
    # fix missing filters
    missing_filters = [k for k in list_filters if f"FLUXCAL_{k}" not in df.columns]
    for f in missing_filters:
        df[f"FLUXCAL_{f}"] = np.nan
        df[f"FLUXCALERR_{f}"] = np.nan
    for flt in list_filters:
        df[flt] = np.where(df["FLUXCAL_%s" % flt].isnull(), "", flt)
    df["FLT"] = df[list_filters[0]]
    for flt in list_filters[1:]:
        df["FLT"] += df[flt]
    # Drop some irrelevant columns
    df = df.drop(columns=list_filters)
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
    df_salt = data_utils.load_fitfile(settings, verbose=False)
    if len(df_salt) > 1:
        df_salt = df_salt.set_index("SNID")
    else:
        # if no fits file we populate with dummies
        # logging_utils.print_yellow(f"Creating dummy mB,c,x1")
        df_salt = pd.DataFrame()
        df_salt["SNID"] = np.array(df.index.unique())
        df_salt["mB"] = np.zeros(len(df.index.unique()))
        df_salt["c"] = np.zeros(len(df.index.unique()))
        df_salt["x1"] = np.zeros(len(df.index.unique()))
        df_salt = df_salt.set_index("SNID")
    df = df.join(df_salt[["mB", "c", "x1"]], how="left")

    df = df.drop(columns="MJD")

    return df


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
    if not settings.debug:
        max_workers = multiprocessing.cpu_count() - 2
        # use parallelization to speed up processing
        # Loop over chunks of files
        for chunk_idx in tqdm(list_chunks, desc="Pivoting dataframes", ncols=100):
            parallel_fn = partial(pivot_dataframe_single, settings=settings)
            # Process each file in the chunk in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                start, end = chunk_idx[0], chunk_idx[-1] + 1
                executor.map(parallel_fn, list_files[start:end])
    else:
        logging_utils.print_yellow("Beware debugging mode (loop over pivot)")
        # for debugging only, process one file only
        for fil in list_files:
            pivot_dataframe_single(fil, settings)

    logging_utils.print_green("Finished pivot")


def parse_sntypes_from_readme(raw_dir):
    """Parse GENTYPE_TO_NAME block from .README files in raw_dir.

    Looks for files matching *.README in raw_dir and extracts supernova
    type mappings from the GENTYPE_TO_NAME block.  For each GENTYPE number
    N found, two entries are created: N and N+100 (the photo-ID convention
    used by SNANA simulations).

    The expected block format is::

        GENTYPE_TO_NAME:  # GENTYPE-integer (non)Ia transient-Name FITS-prefix
          1:   Ia       SALT3              SNIaMODEL00
          20:  nonIa    SNIIP              NONIaMODEL03

    Column mapping (after splitting each data line on whitespace):

    * Column 1 – GENTYPE number (the key, e.g. ``1:``)
    * Column 2 – Ia / nonIa category
    * Column 3 – transient-Name (e.g. ``SNIIP``)

    For Ia types (column 2 == "Ia") the type name is taken from column 2
    directly ("Ia").  For non-Ia types the type name is taken from column 3
    (the transient-Name, e.g. "SNIIP").

    Args:
        raw_dir (str): Path to the raw data directory.

    Returns:
        OrderedDict or None: Parsed ``{sntype_number: type_name}`` mapping,
        or *None* when no README is found or the block is absent / empty.
    """
    readme_files = natsorted(Path(raw_dir).glob("*.README"))
    if not readme_files:
        return None

    # Use the first README found
    readme_file = readme_files[0]

    sntypes = OrderedDict()
    in_gentype_block = False

    with open(readme_file, "r") as f:
        for line in f:
            stripped = line.strip()

            # Detect start of GENTYPE_TO_NAME block
            if stripped.startswith("GENTYPE_TO_NAME:"):
                in_gentype_block = True
                continue

            if not in_gentype_block:
                continue

            # Skip comment lines inside the block
            if stripped.startswith("#"):
                continue

            # Empty line or new section key → end of block
            if not stripped:
                break

            parts = stripped.split()
            # Data lines look like "N: (non)Ia transient-Name FITS-prefix"
            if not (parts and parts[0].endswith(":") and parts[0][:-1].isdigit()):
                break  # not a data line → end of block

            if len(parts) < 3:
                continue  # malformed line, skip

            gentype_num = int(parts[0].rstrip(":"))

            # For Ia types the category name *is* the type name;
            # for non-Ia types use the transient-Name (column 3).
            if parts[1].lower() == "ia":
                type_name = parts[1]  # e.g. "Ia"
            else:
                type_name = parts[2]  # e.g. "SNIIP", "SNIIn"

            # Add both N and N+100 (photo-ID convention)
            sntypes[str(gentype_num)] = type_name
            sntypes[str(gentype_num + 100)] = type_name

    if not sntypes:
        return None

    return sntypes


def resolve_sntypes(settings):
    """Resolve settings.sntypes when not explicitly provided by the user.

    Priority order:

    1. Explicit ``--sntypes`` on CLI / config → already set, nothing to do.
    2. ``.README`` file in ``raw_dir`` → parse ``GENTYPE_TO_NAME`` block.
    3. Built-in ``DEFAULT_SNTYPES`` fallback.

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
    """
    from .. import conf  # local import to avoid circular dependency

    if settings.sntypes is not None:
        # User explicitly provided --sntypes, respect it
        return

    # Try to extract from .README in raw_dir
    readme_sntypes = parse_sntypes_from_readme(settings.raw_dir)
    if readme_sntypes is not None:
        settings.sntypes = readme_sntypes
        logging_utils.print_green(
            "Extracted --sntypes from .README:",
            json.dumps(settings.sntypes),
        )
    else:
        settings.sntypes = conf.DEFAULT_SNTYPES.copy()
        logging_utils.print_yellow(
            "No --sntypes provided and no .README found in raw_dir,",
            "using built-in defaults",
        )


def detect_contaminant_types(settings):
    """Pre-scan raw data files to detect types not in settings.sntypes.

    Any type found in the data but missing from settings.sntypes is
    automatically added as 'contaminant'. This must run before any
    column-name computation so that target column names are consistent
    throughout the pipeline.

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
    """
    list_files = natsorted(Path(settings.raw_dir).glob("**/*HEAD*"))
    all_types = set()
    for fil in list_files:
        fmt = re.search(r"(FITS|csv)", fil.name).group()
        if fmt == "csv":
            df_head = pd.read_csv(fil, usecols=[settings.sntype_var])
        else:
            df_head = Table.read(str(fil), format="fits").to_pandas()
        all_types.update(df_head[settings.sntype_var].astype(str).unique())

    missing_types = [t for t in sorted(all_types) if t not in settings.sntypes]
    if missing_types:
        logging_utils.print_yellow(
            "Missing sntypes",
            f"{missing_types} assigned to 'contaminant' class",
        )
        for mtyp in missing_types:
            settings.sntypes[mtyp] = "contaminant"

    # Warn about sntypes keys not found in the data but do NOT remove them.
    # Phantom keys are harmless: groupby only creates groups for values present
    # in the data, so downsampling and splits work correctly. Keeping them
    # preserves the class structure (target_Nclasses column name and indices),
    # which is essential for compatibility when using a model trained on a
    # dataset that contained all types.
    phantom_keys = [k for k in list(settings.sntypes.keys()) if k not in all_types]
    if phantom_keys:
        logging_utils.print_yellow(
            "Unused sntypes",
            f"Keys {phantom_keys} not found in data (kept for class structure consistency)",
        )


@logging_utils.timer("Data processing")
def make_dataset(settings):
    """Main function for data processing

    - Create the train test val splits
    - Preprocess all the FITs data, then pivot
    - Save all of the processed data to a single HDF5 database

    Args:
        settings (ExperimentSettings): controls experiment hyperparameters
    """
    # Resolve sntypes: CLI > .README in raw_dir > built-in defaults.
    resolve_sntypes(settings)

    # Detect types in data not listed in sntypes and assign as contaminant.
    # This must happen before build_traintestval_splits so that column names
    # (target_Nclasses, dataset_*_Nclasses) are computed correctly.
    detect_contaminant_types(settings)

    # Clean up data folders
    if settings.overwrite is True:
        for folder in [settings.preprocessed_dir, settings.processed_dir]:
            # Dont throw error if folder exists with exist_ok Flag.
            for f in glob.glob(f"{folder}/*"):
                os.remove(f)
            # Save cli args
            settings._save_to_json(settings.processed_dir)

    # split dataset in train test and validation
    build_traintestval_splits(settings)

    # Preprocess dataset
    preprocess_data(settings)

    # Pivot dataframe
    list_files = natsorted(glob.glob(f"{settings.preprocessed_dir}/*PHOT*"))
    pivot_dataframe_batch(list_files, settings)

    # Aggregate the pivoted dataframe
    list_files = natsorted(
        glob.glob(os.path.join(settings.preprocessed_dir, "*pivot.pickle*"))
    )
    logging_utils.print_green("Concatenating pivot")

    df = pd.concat([pd.read_pickle(f) for f in list_files], axis=0)
    # Save to HDF5
    data_utils.save_to_HDF5(settings, df)

    # Save plots to visualize the distribution of some of the data features
    try:
        SNinfo_df = data_utils.load_HDF5_SNinfo(settings)
        datasets_plots(SNinfo_df, settings)
    except Exception:
        logging_utils.print_yellow(
            "Warning: can't do data plots if no saltfit for this dataset"
        )

    # Clean preprocessed directory
    if settings.debug:
        logging_utils.print_red("Debugging mode, keeping preprocessed data")
    else:
        shutil.rmtree(settings.preprocessed_dir)

    logging_utils.print_green("Finished making dataset")
