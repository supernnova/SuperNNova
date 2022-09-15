import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import supernnova.utils.logging_utils as lu
from supernnova.validation.validate_onthefly import classify_lcs

"""
    Example code on how to run on the fly classifications
    - Need to laod a pre-trained model
    - Either provide a list with data or a Pandas DataFrame
"""

# Data columns to provide
# HOSTGAL redshifts only required if classification with redshift is used
COLUMN_NAMES = [
    "SNID",
    "MJD",
    "FLUXCAL",
    "FLUXCALERR",
    "FLT",
    "HOSTGAL_PHOTOZ",
    "HOSTGAL_SPECZ",
    "HOSTGAL_PHOTOZ_ERR",
    "HOSTGAL_SPECZ_ERR",
    "MWEBV",
]


def manual_lc():
    """Manually provide data
    """
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


def load_lc_csv(filename):
    """Read light-curve(s) in csv format

    Args:
        filename (str): data file
    """
    df = pd.read_csv(filename)

    missing_cols = [k for k in COLUMN_NAMES if k not in df.keys()]
    lu.print_red(f"Missing {len(missing_cols)} columns", missing_cols)
    lu.print_yellow(f"filling with zeros")
    lu.print_yellow(f"HOSTGAL required only for classification with redshift")
    for k in missing_cols:
        df[k] = np.zeros(len(df))
    df = df.sort_values(by=["MJD"])
    df["SNID"] = df["SNID"].astype(int).astype(str)

    return df


def reformat_to_df(pred_probs, ids=None):
    """ Reformat SNN predictions to a DataFrame

    # TO DO: suppport nb_inference != 1
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
    """ Wrapper to get predictions on the fly with SNN

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
        help="device to be used [cuda,cpu]",
    )

    args = parser.parse_args()

    # Input data
    # options: csv or manual data, choose one
    # df = load_lc_csv(args.filename)
    df = manual_lc()

    # Obtain predictions for full light-curve
    # Format: batch, nb_inference_samples, nb_classes
    # Beware, ids are resorted while obtaining predictions!
    ids_preds, pred_probs = classify_lcs(df, args.model_file, args.device)

    # ________________________
    # Optional
    #
    # reformat to df
    preds_df = reformat_to_df(pred_probs, ids=df.SNID.unique())
    preds_df.to_csv(f"Predictions_{Path(args.filename).name}")

    print(preds_df)
    # To implement
    # Early prediction visualization
