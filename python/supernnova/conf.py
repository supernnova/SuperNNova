import sys
import json
import yaml
import argparse
from pathlib import Path
from natsort import natsorted
from collections import OrderedDict
from distutils.util import strtobool
from .utils import experiment_settings
from supernnova.utils import logging_utils as lu

# group options to show in print_help for different actions
COMMON_OPTIONS = [
    "--seed",
    "--use_cuda",
    "--dump_dir",
    "--config_file",
    "--no_dump",
    "--help",
]

MAKE_DATA_OPTIONS = COMMON_OPTIONS + [
    "--data_fraction",
    "--data_testing",
    "--data_training",
    "--debug",
    "--norm",
    "--no_overwrite",
    "--raw_dir",
    "--fits_dir",
    "--testing_ids",
    "--photo_window_files",
    "--photo_window_var",
    "--photo_window_min",
    "--photo_window_max",
    "--list_filters",
    "--phot_reject",
    "--phot_reject_list",
    "--redshift",
    "--redshift_label",
    "--explore_lightcurves",
]

TRAIN_RNN_OPTIONS = COMMON_OPTIONS + [
    "--cyclic",
    "--cyclic_phases",
    "--random_length",
    "--random_redshift",
    "--weight_decay",
    "--layer_type",
    "--model",
    "--learning_rate",
    "--nb_classes",
    "--sntypes",
    "--sntype_var",
    "--additional_train_var",
    "--nb_epoch",
    "--batch_size",
    "--hidden_dim",
    "--num_layers",
    "--dropout",
    "--bidirectional",
    "--rnn_output_option",
    "--pi",
    "--log_sigma1",
    "--log_sigma2",
    "--rho_scale_lower",
    "--rho_scale_upper",
    "--log_sigma1_output",
    "--log_sigma2_output",
    "--rho_scale_lower_output",
    "--rho_scale_upper_output",
    "--num_inference_samples",
    "--mean_field_inference",
    "--monitor_interval",
    "--calibration",
    "--plot_file",
]

VALIDATE_RNN_OPTIONS = COMMON_OPTIONS + [
    "--model_files",
    "--plot_lcs",
    "--plot_file",
    "--calibration",
    "--plot_prediction_distribution",
    "--speed",
]

SHOW_OPTIONS = COMMON_OPTIONS + [
    "--model_files",
    "--plot_lcs",
    "--plot_file",
    "--plot_prediction_distribution",
    "--calibration",
]

PERFORMANCE_OPTIONS = COMMON_OPTIONS + [
    "--metrics",
    "--prediction_files",
    "--speed",
    "--done_files",
]

helps = {
    "make_data": MAKE_DATA_OPTIONS,
    "train_rnn": TRAIN_RNN_OPTIONS,
    "validate_rnn": VALIDATE_RNN_OPTIONS,
    "show": SHOW_OPTIONS,
    "performance": PERFORMANCE_OPTIONS,
}

# customize help, so it only print out relevant to the provide command
def generate_command_help(parser, command_arg):
    """Generate a help message for specific command options."""
    help_message = f"usage: snn {command_arg} [options]\n\n"
    help_message += "optional arguments:\n"

    # Filter and sort actions based on their option strings
    command_options = helps[command_arg]
    relevant_actions = [
        action
        for action in parser._actions
        if any(opt in command_options for opt in action.option_strings)
    ]

    sorted_actions = sorted(relevant_actions, key=lambda a: a.option_strings[0])

    for action in sorted_actions:
        option_str = ", ".join(action.option_strings)
        help_descr = f"{option_str:30} {action.help}"
        help_message += f"  {help_descr}\n"
    return help_message


def handle_custom_help(parser, command_arg):
    """Display help messages for provided command"""

    if command_arg in helps.keys():
        print(generate_command_help(parser, command_arg))
    else:
        print("The command {} is not valid.".format(command_arg))
        sys.exit()


class CustomHelpAction(argparse.Action):
    command_arg = None

    def __init__(self, option_strings, dest, help=None):
        super(CustomHelpAction, self).__init__(
            option_strings=option_strings, dest=dest, nargs=0, help=help
        )

    def __call__(self, parser, namespace, values, option_string=None):
        handle_custom_help(parser, self.__class__.command_arg)
        parser.exit()  # Exit after printing the custom help message


def absolute_path(path):
    return str(Path(path).resolve())


def get_args(command_arg):

    CustomHelpAction.command_arg = command_arg

    parser = argparse.ArgumentParser(description="SNIa classification", add_help=False)

    parser.add_argument(
        "--help", action=CustomHelpAction, help="Show custom help message"
    )

    parser.add_argument("--seed", type=int, default=0, help="Random seed to be used")

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
        "--model_files", nargs="+", type=absolute_path, help="Path to model files"
    )  # test it
    parser.add_argument(
        "--prediction_files",
        nargs="+",
        type=absolute_path,
        help="Path to prediction files",  # test it
    )

    parser.add_argument(
        "--metric_files", nargs="+", type=absolute_path, help="Path to metric files"
    )

    parser.add_argument(
        "--done_file",
        default=None,
        type=absolute_path,
        help="Done or failure file name",
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
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # default_dump_dir = str(Path(dir_path).parent.parent / "snndump")
    default_dump_dir = absolute_path("snndump")
    parser.add_argument(
        "--dump_dir",
        type=absolute_path,
        default=default_dump_dir,
        help="Default path where data and models are dumped",
    )

    parser.add_argument(
        "--fits_dir",
        type=absolute_path,
        default=f"{default_dump_dir}/fits",
        help="Default path where fits to photometry are",
    )

    parser.add_argument(
        "--raw_dir",
        type=absolute_path,
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
        "--source_data",
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
        help="Create database with mostly training set of 99.5%%",
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
        type=absolute_path,
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

    parser.add_argument(
        "--config_file", default=None, type=absolute_path, help="YML config file"
    )

    args = parser.parse_args()

    # The following block of code deal with the case when YAML config file is provided alone
    # or together with other options
    if args.config_file:
        # keep track of user provide arguments
        _args = sys.argv[1:]
        _user_namespace, _ = parser._parse_known_args(
            _args, namespace=argparse.Namespace()
        )
        _user_args = vars(_user_namespace)

        # update args from config_file
        yml_args = load_config_file(args.config_file)
        for key in yml_args.keys():
            if hasattr(args, key):
                setattr(args, key, yml_args[key])
                print(getattr(args, key))
            else:
                print("{} is not a valid option.".format(key))

        # update user provided value again in case user provides default value
        if len(_user_args) > 1:
            for key in _user_args.keys():
                setattr(args, key, _user_args[key])

    return args


def get_settings(command_arg, args=None):

    if not args:
        args = get_args(command_arg)

    # Initialize a settings instance

    settings = experiment_settings.ExperimentSettings(args, action=command_arg)
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


def load_config_file(config_file):
    with open(config_file, "r") as f:
        if config_file.endswith(".yml"):
            config = yaml.safe_load(f)
        if config_file.endswith(".json"):
            config = json.load(f)
    return config
