import numpy as np
from pathlib import Path
import supernnova.conf as conf
from supernnova.utils import logging_utils as lu
from supernnova.visualization import (
    visualize,
    visualize_plasticc,
    early_prediction,
    prediction_distribution,
)
from supernnova.training import train_rnn, train_randomforest
from supernnova.paper import superNNova_plots as sp
from supernnova.paper import superNNova_thread as st
from supernnova.data import make_dataset, make_dataset_plasticc
from supernnova.validation import (
    validate_rnn,
    validate_randomforest,
    validate_plasticc,
    metrics,
)


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

            if settings.SWA:
                # reset settings and compute
                model_file = f"{settings.dump_dir}/models/{settings.pytorch_model_name}/{settings.pytorch_model_name}_SWA.pt"

                # Restore model settings
                model_settings = conf.get_settings_from_dump(
                    settings,
                    model_file,
                    override_source_data=settings.override_source_data,
                )
                model_settings.pytorch_model_name = f"{model_settings.pytorch_model_name}_SWA"
                # Get predictions
                prediction_file = validate_rnn.get_predictions(
                    model_settings, model_file=model_file
                )
                # Compute metrics
                metrics.get_metrics_singlemodel(
                    model_settings, prediction_file=prediction_file, model_type="rnn",
                )

                lu.print_blue("Finished SWA")

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
                    if 'SWA' in model_file or settings.SWA:
                        model_settings.pytorch_model_name = f"{model_settings.pytorch_model_name}_SWA"
                    lu.print_yellow('Validation of the SWA model')
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
            visualize.visualize(settings)

        if settings.plot_lcs:
            if settings.model_files:
                for model_file in settings.model_files:
                    settings = conf.get_norm_from_model(model_file, settings)
            early_prediction.make_early_prediction(settings, nb_lcs=20, do_gifs=False)

        if settings.plot_prediction_distribution:
            prediction_distribution.plot_prediction_distribution(settings)

        if settings.science_plots:
            # Provide a prediction_files argument to carry out plot
            sp.science_plots(settings)

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
            st.SuperNNova_stats_and_plots(settings)
            lu.print_blue("Finished assembling paper performance")

        # Speec benchmarks
        if settings.speed:
            validate_rnn.get_predictions_for_speed_benchmark(settings)

        ################
        # PLASTICC
        ################

        if settings.viz_plasticc:
            visualize_plasticc.visualize_plasticc(settings)

        if settings.data_plasticc_train:
            make_dataset_plasticc.make_dataset(settings)

        if settings.data_plasticc_test:
            make_dataset_plasticc.make_test_dataset(settings)

        if settings.train_plasticc:
            if settings.cyclic:
                train_rnn.train_cyclic(settings)
            else:
                train_rnn.train(settings)

        if settings.predict_plasticc:
            validate_plasticc.get_predictions(settings)

        if settings.done_file:
            with open(Path(settings.done_file), "w") as the_file:
                the_file.write("SUCCESS\n")

    except Exception as e:
        settings = conf.get_settings()
        if settings.done_file:
            with open(Path(settings.done_file), "w") as the_file:
                the_file.write("FAILURE\n")
        raise e
