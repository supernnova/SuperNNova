import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import metrics
import supernnova.utils.logging_utils as lu
from supernnova.validation.validate_onthefly import classify_lcs, get_settings
import time

"""
    Example code on how to run on the fly classifications
    - Need to load a pre-trained model
    - Either provide a list with data or a Pandas DataFrame
"""


def manual_lc():
    """Manually provide data"""
    # this is the format you can use to provide light-curves
    df = pd.DataFrame()
    # supernova IDs
    df["SNID"] = ["1", "1", "2", "2"]
    # time in MJD
    df["MJD"] = [57433.4816, 57436.4815, 33444, 33454]
    # FLux and errors
    df["FLUXCAL"] = [2.0, 3, 200, 300]
    df["FLUXCALERR"] = [0.1, 0.2, 0.1, 0.2]
    # bandpasses
    df["FLT"] = ["g", "r", "g", "r"]
    # redshift is not required if classifying without it
    df["HOSTGAL_SPECZ"] = [0.12, 0.12, 0.5, 0.5]
    df["HOSTGAL_PHOTOZ"] = [0.1, 0.1, 0.5, 0.5]
    df["HOSTGAL_SPECZ_ERR"] = [0.001, 0.001, 0.001, 0.001]
    df["HOSTGAL_PHOTOZ_ERR"] = [0.01, 0.01, 0.01, 0.01]
    df["MWEBV"] = [0.01, 0.01, 0.01, 0.01]

    return df


def load_lc_csv(filename, settings):
    """Read light-curve(s) in csv format

    Args:
        filename (str): data file
        settings (str): model file settings

    Returns:
        df (pd.DataFrame): dataframe with lc and metadata
    """

    if "HEAD" in filename:
        df_meta = pd.read_csv(filename)
        df_phot = pd.read_csv(filename.replace("HEAD", "PHOT"))
        df = df_phot.merge(df_meta, how="left", on="SNID")
    else:
        df = pd.read_csv(filename)

    if settings.redshift_label != "none":
        z_label = settings.redshift_features[0].replace("_ERR", "")
        df[z_label] = df[settings.redshift_label]
        df[f"{z_label}_ERR"] = df[f"{settings.redshift_label}_ERR"]

    cols = [
        "SNID",
        "MJD",
        "FLUXCAL",
        "FLUXCALERR",
        "FLT",
        "HOSTGAL_PHOTOZ",
        "HOSTGAL_PHOTOZ_ERR",
        "HOSTGAL_SPECZ",
        "HOSTGAL_SPECZ_ERR",
    ]
    if settings.additional_train_var:
        cols += settings.additional_train_var
    missing_cols = [k for k in cols if k not in df.keys()]
    if len(missing_cols) > 0:
        lu.print_red(f"Missing {len(missing_cols)} columns", missing_cols)
    df = df.sort_values(by=["MJD"])
    df["SNID"] = df["SNID"].astype(int).astype(str)

    return df


def reformat_to_df(pred_probs, ids=None):
    """Reformat SNN predictions to a DataFrame

    Args:
        pred_probs (arr): probabilities array
        ids (str, list): ids

    Returns:
        df (pd.DataFrame): reformatted predictions
    """
    num_inference_samples = 1

    d_series = {}
    for i in range(pred_probs[0].shape[1]):
        d_series["SNID"] = []
        d_series[f"prob_class{i}"] = []
    for idx, value in enumerate(pred_probs):
        d_series["SNID"] += [ids[idx]] if len(ids) > 0 else idx
        value = value.reshape((num_inference_samples, -1))
        value_dim = value.shape[1]
        for i in range(value_dim):
            d_series[f"prob_class{i}"].append(value[:, i][0])
    preds_df = pd.DataFrame.from_dict(d_series)

    # get predicted class
    preds_df["pred_class"] = np.argmax(pred_probs, axis=-1).reshape(-1)

    return preds_df


if __name__ == "__main__":
    """Wrapper to get predictions on the fly with SNN

    Raises:
        ValueError: when model/data is not properly matched
    """
    parser = argparse.ArgumentParser(
        description="Classification using pre-trained model"
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="tests/onthefly_model/vanilla_S_0_CLF_2_R_none_photometry_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean.pt",
        help="path to pre-trained SuperNNova model",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="device to be used [cuda,cpu]"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="tests/onthefly_lc/example_lc.csv",
        help="filename or path to classify",
    )
    parser.add_argument(
        "--redshift_label",
        type=str,
        default="none",
        help="label to be used as redshift",
    )
    parser.add_argument(
        "--manual_lc",
        action="store_true",
        help="if manual lc to be used",
    )

    args = parser.parse_args()

    # Load model configuration
    settings = get_settings(args.model_file)
    # override model redshift_label to input data redshift
    settings.redshift_label = args.redshift_label

    start = time.time()
    start_cpu = time.process_time()

    # Input data
    # options: csv or manual data, choose one
    if args.manual_lc:
        df = manual_lc()
        outname = "manual_lc.csv"
    else:
        if "csv" in args.filename:
            df = load_lc_csv(args.filename, settings)
            outname = f"Predictions_{Path(args.filename).name}"
        else:
            try:
                list_df = []
                to_search = f"{args.filename}/*HEAD*csv"
                if len(to_search) < 1:
                    to_search = f"{args.filename}/*csv"

                for fil in glob.glob(to_search):
                    list_df.append(load_lc_csv(fil, settings))
                df = pd.concat(list_df)
                outname = (
                    f"{args.filename}/Predictions_{Path(args.model_file).name}.csv"
                )
            except Exception:
                print("Provide a csv file or folder with csv")
                raise ValueError

    # Obtain predictions for full light-curve
    # Format: batch, nb_inference_samples, nb_classes
    # Beware, ids are resorted while obtaining predictions!
    ids_preds, pred_probs = classify_lcs(df, args.model_file, args.device)

    # ________________________
    # Optional
    #
    # reformat and save
    preds_df = reformat_to_df(pred_probs, ids=ids_preds)
    preds_df.to_csv(outname)

    # get accuracy
    # use types in training_log.json
    dic_types = {
        0: [10, 11, 12, 20, 21, 25, 26, 27, 30, 31, 32, 35, 36, 37],
        1: [40, 42, 45, 46, 59],
        2: [51, 82, 84, 87, 88, 89],
        3: [60],
        4: [80, 83, 90, 91],
    }
    try:
        preds_df = preds_df.merge(df[["SNID", "SNTYPE"]].drop_duplicates(), on="SNID")
        preds_df["target"] = preds_df["SNTYPE"].apply(
            lambda x: 0
            if x in dic_types[0]
            else (
                1
                if x in dic_types[1]
                else (2 if x in dic_types[2] else (3 if x in dic_types[3] else 4))
            )
        )
        print(
            "Balanced accuracy",
            round(
                100
                * metrics.balanced_accuracy_score(
                    preds_df["target"], preds_df["pred_class"]
                ),
                2,
            ),
        )
    except Exception:
        # silent error
        a = 0

    end = time.time()
    end_cpu = time.process_time()
    print("total running time: ", end - start)
    print("total cpu time: ", end_cpu - start_cpu)
