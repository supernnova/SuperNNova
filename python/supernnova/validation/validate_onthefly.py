import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from supernnova.utils import data_utils
import supernnova.utils.logging_utils as lu
from supernnova.conf import get_norm_from_model
from supernnova.utils import experiment_settings
from supernnova.utils import training_utils as tu
from supernnova.validation.validate_rnn import get_batch_predictions
from supernnova.data.make_dataset import pivot_dataframe_single_from_df


def get_settings(model_file):
    """Define settings from model

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
        # Unset general arguments
        for arg in [
            "data",
            "train_rnn",
            "validate_rnn",
            "explore_lightcurves",
            "dryrun",
            "metrics",
            # "performance",
            "calibration",
            "plot_lcs",
            "prediction_files",
        ]:
            cli_args[arg] = False

    # for on the fly predictions
    cli_args["no_dump"] = True
    cli_args["models_dir"] = Path(model_file).parent.as_posix()

    settings = experiment_settings.ExperimentSettings(cli_args)

    # Backward compatibility (hardcoded...)
    dic_missing_keys = {"sntype_var": "SNTYPE", "additional_train_var": None}
    for k, v in dic_missing_keys.items():
        if k not in cli_args.keys():
            cli_args[k] = v

    settings = experiment_settings.ExperimentSettings(cli_args)

    # load normalization from json dump
    settings = get_norm_from_model(model_file, settings)

    return settings


def format_data(df, settings):
    """Format data into SuperNNova-friendly format

    Args:
        df (pandas.DataFrame): dataframe with light-curves
        settings (ExperimentSettings): custom class to hold hyperparameters
    Returns:
        df (pandas.DataFrame): reformatted data

    """

    # fill missing columns with zeros
    # only if columns are not used for classification
    host_features_model = [k for k in settings.all_features if "HOST" in k]
    host_features_notindata = [k for k in host_features_model if k not in df.keys()]
    if (
        (len(host_features_notindata) > 0 and settings.redshift == "none")
        or (
            settings.redshift == "zspe"
            and "HOSTGAL_SPECZ" not in host_features_notindata
        )
        or settings.redshift == "zpho"
        and "HOSTGAL_PHOTOZ" not in host_features_notindata
    ):
        for hf in host_features_notindata:
            df[hf] = np.zeros(len(df))
    elif len(host_features_notindata) > 0:
        lu.print_red(
            f"Missing features needed for classification {host_features_notindata}"
        )
        raise ValueError

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
    tmp = pd.concat([pd.Series(settings.list_filters_combination), df["FLT"]])
    tmp_onehot = pd.get_dummies(tmp)
    # this is ok since it goes by length not by index (which I never reset)
    # beware: this requires index int!
    FLT_onehot = tmp_onehot[len(settings.list_filters_combination) :]

    df = pd.concat([df, FLT_onehot], axis=1)[settings.all_features]

    ordered_features = [k for k in settings.all_features if k in df.keys()]
    df = df[ordered_features]

    return df


def classify_lcs(df, model_file, device):
    """Obtain predictions for light-curves
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

    # Set the random seed manually for reproducibility.
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)

    if "vanilla" in settings.pytorch_model_name:
        settings.num_inference_samples = 1

    # format data
    df = format_data(df, settings)

    # get packed data batches
    list_lcs = []

    for _, sel in df.groupby(level=0):
        X_all = sel.values
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
        settings.idx_features = [
            i
            for (i, _) in enumerate(settings.all_features)
            if _ in settings.training_features
        ]
        X_normed = X_normed[:, settings.idx_features]
        X_tmp = X_normed, 0, "dummy"

        list_lcs.append(X_tmp)

    packed, _, target_tensor, idxs_rev_sort = tu.get_data_batch(
        list_lcs, np.arange(len(list_lcs)), settings
    )

    # load model
    rnn = tu.get_model(settings, len(settings.training_features))
    rnn_state = torch.load(model_file, map_location=lambda storage, loc: storage)
    try:
        rnn.load_state_dict(rnn_state)
    except Exception:
        if len(settings.training_features) < len(settings.all_features):
            lu.print_red("Model has less features than data (check model filters)")
        else:
            lu.print_red("Check correct model is chosen")
        raise ValueError
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
