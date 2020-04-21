import argparse
import pandas as pd
from supernnova.validation.validate_onthefly import classify_lcs

if __name__ == "__main__":
	""" Wrapper to get predictions on the fly with SNN

	"""

    parser = argparse.ArgumentParser(description="Classification using pre-trained model")
    parser.add_argument("--model_file", type=str, help='path to pre-trained SuperNNova model')
    parser.add_argument("--device", type=str, default="cpu", help = 'device to be used [cuda,cpu]')

    args = parser.parse_args()

    # data
    df = pd.DataFrame()
    df["MJD"] = [57433.4816, 57436.4815, 33444, 33454]
    df["FLUXCAL"] = [2.0, 3, 200, 300]
    df["FLUXCALERR"] = [0.1, 0.2, 0.1, 0.2]
    df["FLT"] = ["g", "r", "g", "r"]
    df["SNID"] = ["1", "1", "2", "2"]
    df["HOSTGAL_SPECZ"] = [0.12, 0.12, 0.5, 0.5]
    df["HOSTGAL_PHOTZ"] = [0.1, 0.1, 0.5, 0.5]

    # Obtain predictions
    # Output format: batch, nb_inference_samples, nb_classes
    preds = classify_lcs(df, args.model_file, args.device)
