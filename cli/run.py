import sys
import re
import os.path as osp
import numpy as np
from pathlib import Path
from supernnova import conf
from supernnova.utils import logging_utils as lu
from supernnova.visualization import (
    visualize,
    early_prediction,
    prediction_distribution,
)
from supernnova.training import train_rnn
from supernnova.paper import superNNova_plots as sp
from supernnova.data import make_dataset
from supernnova.validation import (
    validate_rnn,
    metrics,
)

ALL_ACTIONS = ("make_data", "train_rnn", "validate_rnn", "show", "performance")

help_msg = """
Available commands:

    make_data        create dataset for ML training
    train_rnn        train RNN model
    validate_rnn     validate RNN model
    show             vitualize different types of plot
    performance      get method performance and paper plots

Type snn <command> --help for usage help on a specific command.
For example, snn make_data --help will list all data creation options.
"""


def print_usage():
    print("Usage: {} <command> <options> <arguments>".format(osp.basename(sys.argv[0])))
    print(help_msg)


def get_action():
    """Pop first argument, check it is a valid action."""
    if len(sys.argv) <= 1:
        print_usage()
        sys.exit(1)
    if sys.argv[1] not in ALL_ACTIONS:
        print_usage()
        sys.exit(1)

    return sys.argv.pop(1)


def make_data_action(settings):
    # Validate command-line arguments
    # explore_lightcurves should be used with debug
    if settings.explore_lightcurves and not settings.debug:
        raise ValueError("--explore_lightcurves must be used with --debug")

    # Build an HDF5 database
    make_dataset.make_dataset(settings)
    lu.print_blue("Finished constructing dataset")


def train_rnn_action(settings):

    # Train
    if settings.cyclic:
        train_rnn.train_cyclic(settings)
    else:
        train_rnn.train(settings)

    # Obtain predictions
    validate_rnn.get_predictions(settings)
    # Compute metrics
    metrics.get_metrics_singlemodel(settings, model_type="rnn")
    # Plot some lightcurves
    early_prediction.make_early_prediction(settings)

    lu.print_blue("Finished rnn training, validating, testing and plotting lcs")


def validate_rnn_action(settings):

    if settings.model_files is None:
        validate_rnn.get_predictions(settings)
        # Compute metrics
        metrics.get_metrics_singlemodel(settings, model_type="rnn")
    else:
        for model_file in settings.model_files:
            # Restore model settings
            model_settings = conf.get_settings_from_dump(settings, model_file)
            if settings.num_inference_samples != model_settings.num_inference_samples:
                model_settings.num_inference_samples = settings.num_inference_samples
            # Get predictions
            prediction_file = validate_rnn.get_predictions(
                model_settings, model_file=model_file
            )
            # Compute metrics
            metrics.get_metrics_singlemodel(
                model_settings,
                prediction_file=prediction_file,
                model_type="rnn",
            )


def show_action(settings):

    if settings.explore_lightcurves:
        visualize.visualize(settings)

    if settings.plot_lcs:
        if settings.model_files:
            for model_file in settings.model_files:
                model_settings = conf.get_settings_from_dump(settings, model_file)
        early_prediction.make_early_prediction(
            model_settings, nb_lcs=100, do_gifs=False
        )

    if settings.plot_prediction_distribution:
        prediction_distribution.plot_prediction_distribution(settings)

    if settings.calibration:
        # Provide a metric_files arguments to carry out plot
        sp.plot_calibration(settings)


def performance_action(settings):

    if settings.metrics:
        for prediction_file in settings.prediction_files:
            # TODO: need to make sure only rnn model file is allowed in this step
            model_type = "rf" if "randomforest" in prediction_file else "rnn"
            metrics.get_metrics_singlemodel(
                conf.get_settings_from_dump(settings, prediction_file),
                prediction_file=prediction_file,
                model_type=model_type,
            )
        lu.print_blue("Finished computing metrics")

    # if settings.performance:
    #     from supernnova.utils import logging_utils
    #     metrics.aggregate_metrics(settings)
    #     lu.print_blue("Finished aggregating performance")
    #     # Stats and plots in paper
    #     st.SuperNNova_stats_and_plots(settings)
    #     lu.print_blue("Finished assembling paper performance")

    # Speed benchmarks
    if settings.speed:
        validate_rnn.get_predictions_for_speed_benchmark(settings)

    if settings.done_file:
        with open(Path(settings.done_file), "w") as the_file:
            the_file.write("SUCCESS\n")


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--help":
        print_usage()

    else:
        # Workaround for optparse limitation: insert -- before first negative
        # number found.
        negint = re.compile("-[0-9]+")
        for n, arg in enumerate(sys.argv):
            if negint.match(arg):
                sys.argv.insert(n, "--")
                break

        actions = {
            "make_data": make_data_action,
            "train_rnn": train_rnn_action,
            "validate_rnn": validate_rnn_action,
            "show": show_action,
            "performance": performance_action,
        }

        action = get_action()

        # Get config parameters
        settings = conf.get_settings(action)
        # setting random seeds
        np.random.seed(settings.seed)

        import torch

        torch.manual_seed(settings.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(settings.seed)

        actions[action](settings)


if __name__ == "__main__":
    main()
