import os
import sys
import json
import argparse
from pathlib import Path
from natsort import natsorted
from collections import OrderedDict
from distutils.util import strtobool
from .utils import experiment_settings
from supernnova.utils import logging_utils as lu


def get_args():

    parser = argparse.ArgumentParser(description="SNIa classification")

    parser.add_argument("--seed", type=int, default=0, help="Random seed to be used")

    #######################
    # General parameters
    #######################
    parser.add_argument(
        "--data",
        action="store_true",
        help="Create dataset for ML training",  # write data
    )

    parser.add_argument("--train_rnn", action="store_true", help="Train RNN model")

    parser.add_argument(
        "--validate_rnn", action="store_true", help="Validate RNN model"
    )

    parser.add_argument(
        "--explore_lightcurves",  # use it without using debbug
        action="store_true",
        help="Plot a random selection of lightcurves",
    )
    parser.add_argument(
        "--speed", action="store_true", help="Get RNN speed benchmark"
    )  # test this option!

    parser.add_argument(
        "--monitor_interval",
        type=int,
        default=1,
        help="Monitor validation every monitor_interval epoch",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Use Pred file to compute metrics",  #  test this option!
    )

    # parser.add_argument(
    #     "--performance", # elimiate this option
    #     action="store_true",
    #     help="Get method performance and paper plots",
    # )

    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Plot calibration of trained classifiers",
    )

    parser.add_argument(
        "--plot_lcs",  # test this option
        action="store_true",
        help="Plot lcs with classification probabilities",
    )

    parser.add_argument(
        "--plot_file",  # test this option
        default=None,
        help="Plot subset of lcs in file (csv with SNID column)",
    )

    parser.add_argument(
        "--plot_prediction_distribution",
        action="store_true",
        help="Plot lcs and the histogram of probability for each class",
    )
    parser.add_argument(
        "--model_files", nargs="+", help="Path to model files"
    )  # test it
    parser.add_argument(
        "--prediction_files", nargs="+", help="Path to prediction files"  # test it
    )

    parser.add_argument("--metric_files", nargs="+", help="Path to metric files")

    parser.add_argument(
        "--done_file", default=None, type=str, help="Done or failure file name"
    )

    parser.add_argument(
        "--no_dump", action="store_true", help="No dump database nor preds"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug database creation: one file processed only",
    )

    ########################
    # Data parameters
    ########################
    dir_path = os.path.dirname(os.path.realpath(__file__))
    default_dump_dir = str(Path(dir_path).parent.parent / "snndump")
    parser.add_argument(
        "--dump_dir",
        type=str,
        default=default_dump_dir,
        help="Default path where data and models are dumped",
    )

    parser.add_argument(
        "--fits_dir",  # eliminate it
        type=str,
        default=f"{default_dump_dir}/fits",
        help="Default path where fits to photometry are",
    )

    parser.add_argument(
        "--raw_dir",
        type=str,
        default=f"{default_dump_dir}/raw",
        help="Default path where raw data is",
    )

    parser.add_argument(
        "--redshift",  # change it by Anais
        choices=["none", "zpho", "zspe"],
        default="none",
        help="Host redshift used in classification: none, zpho, zspe",
    )

    parser.add_argument(
        "--norm",
        choices=["none", "perfilter", "global", "cosmo", "cosmo_quantile"],
        default="global",
        help="Feature normalization: global does the same norm for all filters",
    )

    parser.add_argument(
        "--source_data",  # eliminate it
        choices=["saltfit", "photometry"],
        default="photometry",
        help="Data source used to select light-curves for supernnova",
    )

    parser.add_argument(
        "--no_overwrite",
        action="store_true",
        help="If True: do not clean processed_dir and preprocessed_dir when calling `python run.py --data`",
    )

    parser.add_argument(
        "--data_fraction", type=float, default=1.0, help="Fraction of data to use"
    )

    parser.add_argument(
        "--data_training",
        default=False,
        action="store_true",
        help="Create database with mostly training set of 99.5%",
    )

    parser.add_argument(
        "--data_testing",
        default=False,
        action="store_true",
        help="Create database with only validation set",
    )

    parser.add_argument(
        "--testing_ids",  # test it
        default=None,
        help="Filename with SNIDs to be used for testing (.csv with SNID column or .npy)",
    )

    # Photometry window
    parser.add_argument(
        "--photo_window_files",
        nargs="+",
        help="Path to fits with PEAKMJD estimation",  # test it
    )

    parser.add_argument(
        "--photo_window_var",
        type=str,
        default="PKMJDINI",
        help="Variable representing PEAKMJD for photo window (in photo_window_files)",
    )

    parser.add_argument(
        "--photo_window_min", type=int, default=-30, help="Window size before peak"
    )

    parser.add_argument(
        "--photo_window_max", type=int, default=100, help="Window size after peak"
    )

    # Survey configuration
    parser.add_argument(
        "--list_filters",  # test it
        nargs="+",
        default=natsorted(["g", "i", "r", "z"]),
        help="Survey filters",
    )

    # Photometry filtering
    parser.add_argument(
        "--phot_reject",
        type=None,
        help="Variable for photometry flag rejection as a power of 2 (e.g.PHOTFLAG)",
    )

    parser.add_argument(
        "--phot_reject_list",
        nargs="+",
        type=int,
        default=[8, 16, 32, 64, 128, 256, 512],
        help="Bit list to mask (supports only powers of 2)",
    )

    parser.add_argument(
        "--redshift_label",
        type=str,
        default="none",
        help="Redshift label to be used instead of HOSTGAL_SPECZ (with _ERR) and SIM_REDSHIFT",
    )

    ######################
    # RNN  parameters
    ######################
    parser.add_argument(
        "--cyclic", action="store_true", help="Use cyclic learning rate"
    )

    parser.add_argument(
        "--cyclic_phases", nargs=3, default=[5, 10, 15], type=int, help="Cyclic phases"
    )

    parser.add_argument(
        "--random_length",
        choices=[True, False],
        default=True,
        type=lambda x: bool(strtobool(x)),
        help="Use random length sequences for training",
    )

    parser.add_argument(
        "--random_redshift",
        action="store_true",
        help="randomly set spectroscopic redshift to -1 (i.e. unknown)",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0000001,
        help="L2 decay on weights (variational)",
    )

    parser.add_argument(
        "--layer_type",
        default="lstm",
        type=str,
        choices=["lstm", "gru", "rnn"],
        help="recurrent layer type",
    )

    parser.add_argument(
        "--model",
        default="vanilla",  # Anais change the name
        type=str,
        choices=["vanilla", "variational", "bayesian", "bayesian_2"],
        help="recurrent model type",
    )

    parser.add_argument(
        "--use_cuda", action="store_true", help="Use GPU (pytorch backend only)"  # test
    )

    parser.add_argument(
        "--learning_rate", default=1e-3, type=float, help="Learning rate"
    )

    parser.add_argument(
        "--nb_classes",
        default=2,
        type=int,
        help="Number of classification targets",  # test
    )

    parser.add_argument(
        "--sntypes",  # test it
        default=OrderedDict(
            {
                "101": "Ia",
                "120": "IIP",
                "121": "IIn",
                "122": "IIL1",
                "123": "IIL2",
                "132": "Ib",
                "133": "Ic",
            }
        ),
        type=json.loads,
        help="SN classes in sims (put Ia always first)",
    )

    parser.add_argument(
        "--sntype_var",
        type=str,
        default="SNTYPE",
        help="Variable representing event types (e.g. SNTYPE)",
    )

    parser.add_argument(
        "--additional_train_var",  # test
        nargs="+",
        help="Additional training variables",
    )

    parser.add_argument(
        "--nb_epoch", default=90, type=int, help="Number of batches per epoch"
    )

    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")

    parser.add_argument(
        "--hidden_dim", default=32, type=int, help="Hidden layer dimension"
    )

    parser.add_argument(
        "--num_layers", default=2, type=int, help="Number of recurrent layers"
    )

    parser.add_argument("--dropout", default=0.05, type=float, help="Dropout value")

    parser.add_argument(
        "--bidirectional",
        choices=[True, False],
        default=True,
        type=lambda x: bool(strtobool(x)),
        help="Use bidirectional models",
    )

    parser.add_argument(
        "--rnn_output_option",
        default="mean",
        type=str,
        choices=["standard", "mean"],
        help="RNN output options",
    )

    parser.add_argument("--pi", default=0.75, type=float)

    ### change to pytorch higher version
    parser.add_argument("--log_sigma1", default=-1.0, type=float)
    parser.add_argument("--log_sigma2", default=-7.0, type=float)
    parser.add_argument("--rho_scale_lower", default=4.0, type=float)
    parser.add_argument("--rho_scale_upper", default=3.0, type=float)

    # Different parameters for output layer to obtain better uncertainty
    parser.add_argument("--log_sigma1_output", default=-1.0, type=float)
    parser.add_argument("--log_sigma2_output", default=-7.0, type=float)
    parser.add_argument("--rho_scale_lower_output", default=4.0, type=float)
    parser.add_argument("--rho_scale_upper_output", default=3.0, type=float)
    ### pytorch higher version end

    parser.add_argument(
        "--num_inference_samples",
        type=int,
        default=50,
        help="Number of samples to use for Bayesian inference",
    )

    parser.add_argument(
        "--mean_field_inference",
        action="store_true",
        help="Use mean field inference for bayesian models",
    )

    parser.add_argument("--config_file", default=None, type=str, help="YML config file")

    args = parser.parse_args(sys.argv[2:])

    return args


def get_settings(args=None):

    if not args:
        args = get_args()

    # Initialize a settings instance

    settings = experiment_settings.ExperimentSettings(args)

    assert args.rho_scale_lower >= args.rho_scale_upper

    return settings


def get_settings_from_dump(settings, model_or_pred_or_metrics_file):
    # Model settings
    model_dir = Path(model_or_pred_or_metrics_file).parent
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
            "config_file",
        ]:
            cli_args[arg] = False

    # Using dump/fits/raw dir from settings instead of model

    # raw and fits shouldnt change a thing
    cli_args["raw_dir"] = settings.raw_dir
    cli_args["fits_dir"] = settings.fits_dir
    cli_args["dump_dir"] = settings.dump_dir

    # and device
    cli_args["use_cuda"] = settings.use_cuda
    cli_args["device"] = settings.device

    # model files
    cli_args["model_files"] = settings.model_files

    # model files
    cli_args["plot_file"] = settings.plot_file

    # Backward compatibility
    keys_not_in_model_settings = [
        k for k in settings.cli_args.keys() if k not in cli_args.keys()
    ]
    for k in keys_not_in_model_settings:
        cli_args[k] = settings.cli_args[k]

    # Warning for redshift
    if cli_args["redshift"] != settings.redshift:
        lu.print_red("Model forces redshift to be set as", cli_args["redshift"])

    settings = experiment_settings.ExperimentSettings(cli_args)

    # load normalization from json dump
    settings = get_norm_from_model(model_or_pred_or_metrics_file, settings)

    return settings


def get_norm_from_model(model_file, settings):

    import json
    import numpy as np

    norm_file = Path(model_file).parent / "data_norm.json"
    with open(norm_file, "r") as f:
        norm_args = json.load(f)
        list_norm = []
        for f in settings.training_features_to_normalize:
            minv = norm_args[f]["min"]
            meanv = norm_args[f]["mean"]
            stdv = norm_args[f]["std"]
            list_norm.append([minv, meanv, stdv])
    settings.arr_norm = np.array(list_norm)

    return settings
