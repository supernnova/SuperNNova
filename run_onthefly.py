import numpy as np
import argparse
import pandas as pd
from supernnova.validation.validate_onthefly import classify_lcs


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
    df["HOSTGAL_PHOTZ"] = [0.1, 0.1, 0.5, 0.5]

    return df


def load_lc_csv(filename):
    df = pd.read_csv(filename, usecols=["SNID", "FLUXCAL", "FLUXCALERR", "FLT", "MJD"],)
    df["HOSTGAL_SPECZ"] = np.zeros(len(df))
    df["HOSTGAL_PHOTZ"] = np.zeros(len(df))
    df = df.sort_values(by=["MJD"])
    df["SNID"] = df["SNID"].astype(int).astype(str)

    return df


def reformat_to_df(pred_probs, ids=None):
    # TO DO: suppport nb_inference != 1
    num_inference_samples = 1

    d_series = {}
    for i in range(pred_probs[0].shape[1]):
        d_series["SNID"] = []
        d_series[f"prob_class{i}"] = []
    for idx, value in enumerate(pred_probs):
        d_series["SNID"] += ids[idx] if len(ids) > 0 else idx
        value = value.reshape((num_inference_samples, -1))
        value_dim = value.shape[1]
        for i in range(value_dim):
            d_series[f"prob_class{i}"].append(value[:, i][0])
    preds_df = pd.DataFrame.from_dict(d_series)

    # get predicted class
    try:
        preds_df["pred_class"] = np.argmax(pred_probs, axis=0)[0]
    except Exception:
        preds_df["pred_class"] = np.argmax(pred_probs[0], axis=0)[0]

    return preds_df


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
    # Format: batch, nb_inference_samples, nb_classes
    pred_probs = classify_lcs(df, args.model_file, args.device)

    # reformat to df
    preds_df = reformat_to_df(pred_probs, ids=df.SNID.unique())

    import ipdb

    ipdb.set_trace()
