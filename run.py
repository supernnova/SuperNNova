import numpy as np
from pathlib import Path
from supernnova import conf
from supernnova.utils import logging_utils as lu
from supernnova.visualization import (
    visualize,
    early_prediction,
    prediction_distribution,
)
from supernnova.training import train_rnn, train_randomforest
from supernnova.paper import superNNova_plots as sp
from supernnova.paper import superNNova_thread as st
from supernnova.data import make_dataset
from supernnova.validation import (
    validate_rnn,
    validate_randomforest,
    metrics,
)
from supernnova.utils import logging_utils


if __name__ == "__main__":

    try:

        # Get conf parameters
        settings = conf.get_settings()
        # setting random seeds
        np.random.seed(settings.seed)
        import torch

        torch.manual_seed(settings.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(settings.seed)

        ################
        # DATA
        ################
        if settings.data:
            # Build an HDF5 database
            make_dataset.make_dataset(settings)
            lu.print_blue("Finished constructing dataset")

        ################
        # TRAINING
        ################
        if settings.train_rnn:

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

        if settings.train_rf:

            train_randomforest.train(settings)
            # Obtain predictions
            validate_randomforest.get_predictions(settings)
            # Compute metrics
            metrics.get_metrics_singlemodel(settings, model_type="rf")

            lu.print_blue("Finished rf training, validating and testing")

        ################
        # VALIDATION
        ################
        if settings.validate_rnn:

            if settings.model_files is None:
                validate_rnn.get_predictions(settings)
                # Compute metrics
                metrics.get_metrics_singlemodel(settings, model_type="rnn")
            else:
                for model_file in settings.model_files:
                    # Restore model settings
                    model_settings = conf.get_settings_from_dump(
                        settings,
                        model_file,
                        override_source_data=settings.override_source_data,
                    )
                    if (
                        settings.num_inference_samples
                        != model_settings.num_inference_samples
                    ):
                        model_settings.num_inference_samples = (
                            settings.num_inference_samples
                        )
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

        if settings.validate_rf:

            if settings.model_files is None:
                validate_randomforest.get_predictions(settings)
                # Compute metrics
                metrics.get_metrics_singlemodel(settings, model_type="rf")
            else:
                for model_file in settings.model_files:
                    # Restore model settings
                    model_settings = conf.get_settings_from_dump(
                        settings,
                        model_file,
                        override_source_data=settings.override_source_data,
                    )
                    # Get predictions
                    prediction_file = validate_randomforest.get_predictions(
                        model_settings, model_file=model_file
                    )
                    # Compute metrics
                    metrics.get_metrics_singlemodel(
                        model_settings, prediction_file=prediction_file, model_type="rf"
                    )

        ##################################
        # VISUALIZE
        ##################################
        if settings.explore_lightcurves:
            if settings.debug:
                visualize.visualize(settings)
            else:
                logging_utils.print_red("Use --debug --data for explore_lightcurves")

        if settings.plot_lcs:
            if settings.model_files:
                for model_file in settings.model_files:
                    model_settings = conf.get_settings_from_dump(
                        settings,
                        model_file,
                        override_source_data=settings.override_source_data,
                    )
            early_prediction.make_early_prediction(
                model_settings, nb_lcs=100, do_gifs=False
            )

        if settings.plot_prediction_distribution:
            prediction_distribution.plot_prediction_distribution(settings)

        if settings.science_plots:
            # Provide a prediction_files argument to carry out plot
            lu.print_yellow("Will fail if --prediction_files not specified")
            sp.science_plots(settings, onlycnf=True)

        if settings.calibration:
            # Provide a metric_files arguments to carry out plot
            sp.plot_calibration(settings)

        ##################################
        # PERFORMANCE
        ##################################

        if settings.metrics:
            for prediction_file in settings.prediction_files:
                model_type = "rf" if "randomforest" in prediction_file else "rnn"
                metrics.get_metrics_singlemodel(
                    conf.get_settings_from_dump(settings, prediction_file),
                    prediction_file=prediction_file,
                    model_type=model_type,
                )
            lu.print_blue("Finished computing metrics")

        if settings.performance:
            metrics.aggregate_metrics(settings)
            lu.print_blue("Finished aggregating performance")
            # Stats and plots in paper
            # st.SuperNNova_stats_and_plots(settings)
            # lu.print_blue("Finished assembling paper performance")

        # Speec benchmarks
        if settings.speed:
            validate_rnn.get_predictions_for_speed_benchmark(settings)

        if settings.done_file:
            with open(Path(settings.done_file), "w") as the_file:
                the_file.write("SUCCESS\n")

    except Exception as e:
        settings = conf.get_settings()
        if settings.done_file:
            with open(Path(settings.done_file), "w") as the_file:
                the_file.write("FAILURE\n")
        raise e
