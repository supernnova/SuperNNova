import numpy as np
import argparse
import pandas as pd
from supernnova.validation.validate_onthefly import classify_lcs


def manual_lc():
    """Manually provide data
    """
    df = pd.DataFrame()
    df["MJD"] = [57433.4816, 57436.4815, 33444, 33454]
    df["FLUXCAL"] = [2.0, 3, 200, 300]
    df["FLUXCALERR"] = [0.1, 0.2, 0.1, 0.2]
    df["FLT"] = ["g", "r", "g", "r"]
    df["SNID"] = ["1", "1", "2", "2"]
    df["HOSTGAL_SPECZ"] = [0.12, 0.12, 0.5, 0.5]
    df["HOSTGAL_PHOTZ"] = [0.1, 0.1, 0.5, 0.5]

    return df


def load_lc_csv(filename):
    df = pd.read_csv(filename, usecols=["SNID", "FLUXCAL", "FLUXCALERR", "FLT", "MJD"],)
    df["HOSTGAL_SPECZ"] = np.zeros(len(df))
    df["HOSTGAL_PHOTZ"] = np.zeros(len(df))
    df = df.sort_values(by=["MJD"])
    df["SNID"] = df["SNID"].astype(int).astype(str)

    return df


if __name__ == "__main__":
    """ Wrapper to get predictions on the fly with SNN

    """

    parser = argparse.ArgumentParser(
        description="Classification using pre-trained model"
    )
    parser.add_argument(
        "--model_file", type=str, help="path to pre-trained SuperNNova model"
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

    # data
    # csv or manual data, choose one
    # df = load_lc_csv(args.filename)
    df = manual_lc()

    # Obtain predictions for full light-curve
    # Output format: batch, nb_inference_samples, nb_classes
    preds = classify_lcs(df, args.model_file, args.device)

    import ipdb

    ipdb.set_trace()
