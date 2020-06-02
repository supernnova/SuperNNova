import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from supernnova.utils import data_utils
from supernnova.conf import get_norm_from_model
from supernnova.utils import experiment_settings
from supernnova.utils import training_utils as tu
from supernnova.validation.validate_rnn import get_batch_predictions
from supernnova.data.make_dataset import pivot_dataframe_single_from_df


def get_settings(model_file):
    """ Define settings from model

    Args:
        model_file (str): complete name with path for model to be used
    Returns:
        settings (ExperimentSettings): custom class to hold hyperparameters
    """

    # Load model
    model_dir = Path(model_file).parent
    cli_file = model_dir / "cli_args.json"
    with open(cli_file, "r") as f:
        cli_args = json.load(f)

    # for on the fly predictions
    cli_args["no_dump"] = True
    cli_args["models_dir"] = Path(model_file).parent.as_posix()

    settings = experiment_settings.ExperimentSettings(cli_args)

    # load normalization from json dump
    settings = get_norm_from_model(model_file, settings)

    return settings


def format_data(df, settings):
    """ Format data into SuperNNova-friendly format

    Args:
        df (pandas.DataFrame): dataframe with light-curves
        settings (ExperimentSettings): custom class to hold hyperparameters
    Returns:
        df (pandas.DataFrame): reformatted data

    """
    # compute delta time
    df = data_utils.compute_delta_time(df)
    # fill dummies
    if "PEAKMJD" not in df.keys():
        df["PEAKMJD"] = np.zeros(len(df))

    # pivot
    df = pivot_dataframe_single_from_df(df, settings)

    # onehot
    # Fit a one hot encoder for FLT
    # to have the same onehot for all datasets
    tmp = pd.Series(settings.list_filters_combination).append(df["FLT"])
    tmp_onehot = pd.get_dummies(tmp)
    # this is ok since it goes by length not by index (which I never reset)
    # beware: this requires index int!
    FLT_onehot = tmp_onehot[len(settings.list_filters_combination) :]
    df = pd.concat([df, FLT_onehot], axis=1)[settings.training_features]

    return df


def classify_lcs(df, model_file, device):
    """ Obtain predictions for light-curves
    Args:
        df (DataFrame): light-curves to classify
        model_file (str): Path+name of model to use for predictions
        device (str): wehter to use cuda or cpu

    Returns:
        idx (list): light-curve indices after classification (they are resorted)
        preds (np.array): predictions for this model (shape= len(idx),model_nb_class)
    """
    # init
    settings = get_settings(model_file)
    settings.use_cuda = True if "cuda" in str(device) else False
    settings.idx_features_to_normalize = [
        i
        for (i, f) in enumerate(settings.training_features)
        if f in settings.training_features_to_normalize
    ]
    settings.random_length = False
    settings.random_redshift = False
    if "vanilla" in settings.pytorch_model_name:
        settings.num_inference_samples = 1

    # format data
    df = format_data(df, settings)

    # get packed data batches
    list_lcs = []
    for idx in df.index.unique():
        sel = df[df.index == idx]
        X_all = sel[settings.training_features].values
        # check if normalization converges
        # using clipping in case of min<model_min
        X_clip = X_all.copy()
        X_clip = np.clip(
            X_clip[:, settings.idx_features_to_normalize],
            settings.arr_norm[:, 0],
            np.inf,
        )
        X_all[:, settings.idx_features_to_normalize] = X_clip

        # Normalize features that need to be normalized
        X_normed = X_all.copy()
        X_normed = tu.normalize_arr(X_normed, settings)
        # format: data, target (filled with zeros), _
        X_tmp = X_normed, 0, "dummy"
        list_lcs.append(X_tmp)

    packed, _, target_tensor, idxs_rev_sort = tu.get_data_batch(
        list_lcs, np.arange(len(list_lcs)), settings
    )

    # load model
    rnn = tu.get_model(settings, len(settings.training_features))
    rnn_state = torch.load(model_file, map_location=lambda storage, loc: storage)
    rnn.load_state_dict(rnn_state)
    rnn.to(device)
    rnn.eval()

    # obtain predictions
    list_preds = []
    for iter_ in range(settings.num_inference_samples):

        arr_preds, _ = get_batch_predictions(rnn, packed, target_tensor)

        # Rever sorting that occurs in get_batch_predictions
        arr_preds = arr_preds[idxs_rev_sort]

        list_preds.append(arr_preds)

    # B, inf_samples, nb_classes
    preds = np.stack(list_preds, axis=1)

    return df.index.unique(), preds
