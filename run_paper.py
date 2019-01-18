import os
import json
import torch
import shlex
import argparse
import subprocess
import pandas as pd
from pathlib import Path
from itertools import product
from supernnova.paper.superNNova_plots import plot_speed_benchmark
from supernnova.utils import logging_utils as lu

"""superNNova paper experiments
"""

LIST_SEED = [0, 100, 1000, 55, 30496]


def run_cmd(cmd, debug, seed):
    """Run command
    Using cuda if available
    """

    cmd += f" --seed {seed} "

    if torch.cuda.is_available():
        cmd += " --use_cuda "

    if debug is True:
        # Run for 1 epoch only
        cmd = cmd.replace("--cyclic ", " ")
        cmd = cmd + " --nb_epoch 1 "

        if "num_inference_samples" not in cmd:
            # Make inference faster
            cmd = cmd + "--num_inference_samples 2 "

        if "hidden_dim" not in cmd:
            # Decrease NN size
            cmd = cmd + "--hidden_dim 2 "

    subprocess.check_call(shlex.split(cmd))


def run_data(dump_dir, debug, seed):
    """Create database
    """

    cmd = "python -W ignore run.py --data " f"--dump_dir {dump_dir} "
    run_cmd(cmd, debug, seed)


def run_speed(dump_dir, debug, seed):

    lu.print_green(f"SEED {seed}: SPEED")

    if seed != LIST_SEED[0]:
        return

    models = ["vanilla", "variational", "bayesian"]
    use_cuda = [True, False]
    source_data = "saltfit"
    purpose = "speed"

    for model, use_cuda in product(models, use_cuda):

        # No cuda benchmark if cuda is not available
        if use_cuda is True and not torch.cuda.is_available():
            continue

        cmd = (
            f"python -W ignore run.py --{purpose} "
            f"--model {model} "
            f"--cyclic "
            f"--dump_dir {dump_dir} "
            f"--source_data {source_data} "
        )
        if use_cuda:
            cmd += f" --use_cuda "

        # Call subprocess here because run_cmd otherwise
        # may mess up with gpu options
        if debug is True:
            # Run for 1 epoch only
            cmd = cmd.replace("--cyclic ", " ")
            cmd = cmd + " --nb_epoch 1 "

            if "num_inference_samples" not in cmd:
                # Make inference faster
                cmd = cmd + "--num_inference_samples 2 "

            if "hidden_dim" not in cmd:
                # Decrease NN size
                cmd = cmd + "--hidden_dim 2 "

        subprocess.check_call(shlex.split(cmd))

    # Create plots with the results
    plot_speed_benchmark(dump_dir)


def run_benchmark_cyclic(dump_dir, debug, seed):

    lu.print_green(f"SEED {seed}: BENCHMARK CYCLIC")

    if seed != LIST_SEED[0]:
        return

    models = ["vanilla"]
    training_method = ["cyclic", ""]
    data_fraction = [0.25, 0.5, 1.0]
    cyclic_phases = [["5", "10", "15"], ["10", "20", "25"]]
    source_data = "saltfit"
    purpose = "train_rnn"

    if debug is True:
        cyclic_phases = [["1", "2", "3"]]
        data_fraction = data_fraction[:1]

    for model, training_method, data_fraction in product(
        models, training_method, data_fraction
    ):

        cmd = (
            f"python -W ignore run.py --{purpose} "
            f"--model {model} "
            f"--dump_dir {dump_dir} "
            f"--source_data {source_data} "
            f"--data_fraction {data_fraction} "
        )
        if training_method == "cyclic":
            cmd += "--cyclic "
            for cyclic_phase in cyclic_phases:
                cmd += f"--cyclic_phases {' '.join(cyclic_phase)} "
                # Set debug to False to avoid overriding
                run_cmd(cmd, False, seed)
        else:
            run_cmd(cmd, debug, seed)

    # Load and save results
    list_logs = (Path(dump_dir) / "models").glob("**/training_log.json")
    list_df = []
    for log_file in list_logs:
        with open(log_file, "r") as f:
            df_cyclic = pd.DataFrame.from_dict(json.load(f), orient="index")
            list_df.append(df_cyclic)

    df_cyclic = pd.concat(list_df)

    with open(Path(dump_dir) / "stats/cyclic_stats.tex", "w") as tf:
        tf.write(df_cyclic.to_latex())


def run_baseline_hp(dump_dir, debug, seed):

    lu.print_green(f"SEED {seed}: BASELINE HP")

    if seed != LIST_SEED[0]:
        return

    list_batch_size = [64, 128, 512]
    list_dropout = [0.05, 0.1, 0.2]
    list_num_layers = [1, 2]
    list_layer_type = ["gru", "lstm"]
    list_bidirectional = [True, False]
    list_rnn_output_option = ["standard", "mean"]
    list_random_length = [True, False]
    list_hidden_dim = [16, 32]

    if debug is True:
        list_batch_size = list_batch_size[:1]
        list_dropout = list_dropout[:1]
        list_hidden_dim = list_hidden_dim[:1]

    for (
        batch_size,
        dropout,
        num_layers,
        layer_type,
        bidirectional,
        rnn_output_option,
        random_length,
        hidden_dim,
    ) in product(
        list_batch_size,
        list_dropout,
        list_num_layers,
        list_layer_type,
        list_bidirectional,
        list_rnn_output_option,
        list_random_length,
        list_hidden_dim,
    ):
        cmd = (
            f"python -W ignore run.py --train_rnn "
            f"--dump_dir {dump_dir} "
            f"--cyclic "
            f"--data_fraction 0.2 "
            f"--dropout {dropout} "
            f"--batch_size {batch_size} "
            f"--layer_type {layer_type} "
            f"--num_layers {num_layers} "
            f"--bidirectional {bidirectional} "
            f"--random_length {random_length} "
            f"--rnn_output_option {rnn_output_option} "
            f"--hidden_dim {hidden_dim} "
        )
        run_cmd(cmd, debug, seed)


def run_baseline(dump_dir, debug, seed):
    """Baseline/Random Forest Accuracy vs. number of supernovae
        Default configurations used when not specified
        e.g. source_data(saltfit),modelrnn(vanilla),norm(global)
    """

    lu.print_green(f"SEED {seed}: TRAINING")

    #################################
    # Train baseline models on SALT #
    #################################
    list_data_fraction = [0.5, 1.0] if debug else [0.05, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    list_redshift = [None, "zpho", "zspe"]

    # Train RF models
    for (data_fraction, redshift) in product(list_data_fraction, list_redshift):
        cmd = (
            f"python -W ignore run.py --train_rf "
            f"--data_fraction {data_fraction} "
            f"--dump_dir {dump_dir} "
        )
        if redshift is not None:
            cmd += f" --redshift {redshift} "
        run_cmd(cmd, debug, seed)

    # Train RNN models
    for (data_fraction, redshift) in product(list_data_fraction, list_redshift):
        cmd = (
            f"python -W ignore run.py --train_rnn "
            f"--data_fraction {data_fraction} "
            f"--cyclic "
            f"--dump_dir {dump_dir} "
        )
        if redshift is not None:
            cmd += f" --redshift {redshift} "
        run_cmd(cmd, debug, seed)

    # Train RNN models, varying normalization strategy
    for norm in ["perfilter", "none"]:
        cmd = (
            f"python -W ignore run.py --train_rnn "
            f"--data_fraction 0.5 "
            f"--norm {norm} "
            f"--cyclic "
            f"--dump_dir {dump_dir} "
        )
        run_cmd(cmd, debug, seed)

    #######################################
    # Train baseline models on COMPLETE   #
    # goal: representativeness            #
    #######################################

    for data_fraction, redshift in product([0.43, 0.5], list_redshift):
        cmd = (
            f"python -W ignore run.py --train_rnn "
            f"--data_fraction {data_fraction} "
            f"--cyclic "
            f"--source_data photometry "
            f"--dump_dir {dump_dir} "
        )
        if redshift is not None:
            cmd += f" --redshift {redshift} "
        run_cmd(cmd, debug, seed)

    #######################################
    # Train baseline models on COMPLETE   #
    # goal: multiclass                    #
    #######################################

    list_nb_classes = [2, 3, 7]
    for (redshift, nb_classes) in product(list_redshift, list_nb_classes):
        cmd = (
            f"python -W ignore run.py --train_rnn "
            f"--nb_classes {nb_classes} "
            f"--dump_dir {dump_dir} "
            f"--source_data photometry "
            f"--cyclic "
        )
        if redshift is not None:
            cmd += f" --redshift {redshift} "
        run_cmd(cmd, debug, seed)


def run_variational_hp(dump_dir, debug, seed):
    """Variational dropout and weight decay tests
        Default configurations used when not specified
        e.g. no redshift
    """

    lu.print_green(f"SEED {seed}: VARIATIONAL HP")

    if seed != LIST_SEED[0]:
        return

    list_dropout_values = [0.01, 0.05, 0.1]
    list_weight_decay = [0, 1E-7, 1E-5]

    if debug is True:
        list_dropout_values = list_dropout_values[:1]
        list_weight_decay = list_weight_decay[:1]

    for (dropout_values, weight_decay) in product(
        list_dropout_values, list_weight_decay
    ):
        cmd = (
            f"python -W ignore run.py --train_rnn "
            f"--model variational "
            f"--dump_dir {dump_dir} "
            f"--cyclic "
            f"--data_fraction 0.2 "
            f"--num_inference_samples 10 "
            f"--dropout {dropout_values} "
            f"--weight_decay {weight_decay} "
        )
        run_cmd(cmd, debug, seed)


def run_variational_best(dump_dir, debug, seed):
    """Variational dropout and weight decay tests
        Default configurations used when not specified
        e.g. no redshift
    """

    lu.print_green(f"SEED {seed}: VARIATIONAL BEST")

    list_nb_classes = [2, 3, 7]
    list_redshift = [None, "zpho", "zspe"]
    list_data_fraction = [0.43, 1.0]
    for (nb_classes, redshift, data_fraction) in product(
        list_nb_classes, list_redshift, list_data_fraction
    ):

        # Carry out representativeness only in binary classification
        if data_fraction == 0.43 and nb_classes != 2:
            continue

        cmd = (
            f"python -W ignore run.py --train_rnn "
            f"--model variational "
            f"--source_data photometry "
            f"--dump_dir {dump_dir} "
            f"--cyclic "
            f"--data_fraction {data_fraction} "
            f"--dropout 0.01 "
            f"--weight_decay 1e-7 "
            f"--nb_classes {nb_classes} "
        )
        if redshift is not None:
            cmd += f" --redshift {redshift} "

        run_cmd(cmd, debug, seed)

    list_data_fraction = [0.5, 1.0]
    for (redshift, data_fraction) in product(list_redshift, list_data_fraction):
        cmd = (
            f"python -W ignore run.py --train_rnn "
            f"--model variational "
            f"--source_data saltfit "
            f"--dump_dir {dump_dir} "
            f"--cyclic "
            f"--data_fraction {data_fraction} "
            f"--dropout 0.01 "
            f"--weight_decay 1e-7 "
        )
        if redshift is not None:
            cmd += f" --redshift {redshift} "

        run_cmd(cmd, debug, seed)


def run_bayesian_hp(dump_dir, debug, seed):
    """Bayesian scale tests
        Default configurations used when not specified
        e.g. data_fraction(1),no redshift
    """

    lu.print_green(f"SEED {seed}: BAYESIAN HP")

    if seed != LIST_SEED[0]:
        return

    list_params = [[-2, -7, 4, 3], [-1, -7, 4, 3], [-2, -1, 20, 5]]
    list_params_output = [[-1, -0.5, 2, 1], [-0.5, -0.1, 2, 1], [-0.5, -0.1, 3, 2]]

    if debug is True:
        list_params = list_params[:1]
        list_params_output = list_params_output[:1]

    for (params, params_output) in product(list_params, list_params_output):
        log_sigma1, log_sigma2, rho_scale_lower, rho_scale_upper = params
        log_sigma1_output, log_sigma2_output, rho_scale_lower_output, rho_scale_upper_output = (
            params_output
        )

        cmd = (
            f"python -W ignore run.py --train_rnn "
            f"--model bayesian "
            f"--dump_dir {dump_dir} "
            f"--data_fraction 0.2 "
            f"--num_inference_samples 10 "
            f"--log_sigma1 {log_sigma1} "
            f"--log_sigma2 {log_sigma2} "
            f"--rho_scale_lower {rho_scale_lower} "
            f"--rho_scale_upper {rho_scale_upper} "
            f"--log_sigma1_output {log_sigma1_output} "
            f"--log_sigma2_output {log_sigma2_output} "
            f"--rho_scale_lower_output {rho_scale_lower_output} "
            f"--rho_scale_upper_output {rho_scale_upper_output} "
        )
        run_cmd(cmd, debug, seed)


def run_bayesian_best(dump_dir, debug, seed):
    """Bayesian scale tests
        Default configurations used when not specified
        e.g. data_fraction(1),no redshift
    """

    lu.print_green(f"SEED {seed}: BAYESIAN BEST")

    list_nb_classes = [2, 3, 7]
    list_redshift = [None, "zpho", "zspe"]
    list_data_fraction = [0.43, 1.0]
    for (nb_classes, redshift, data_fraction) in product(
        list_nb_classes, list_redshift, list_data_fraction
    ):

        # Carry out representativeness only in binary classification
        if data_fraction == 0.43 and nb_classes != 2:
            continue

        cmd = (
            f"python -W ignore run.py --train_rnn "
            f"--model bayesian "
            f"--source_data photometry "
            f"--dump_dir {dump_dir} "
            f"--data_fraction {data_fraction} "
            f"--nb_classes {nb_classes} "
            f"--log_sigma1 -1 "
            f"--log_sigma2 -7 "
            f"--rho_scale_lower 4 "
            f"--rho_scale_upper 3 "
            f"--log_sigma1_output -0.5 "
            f"--log_sigma2_output -0.1 "
            f"--rho_scale_lower_output 3 "
            f"--rho_scale_upper_output 2 "
        )
        if redshift is not None:
            cmd += f" --redshift {redshift} "

        run_cmd(cmd, debug, seed)

    list_data_fraction = [0.5, 1.0]
    for (redshift, data_fraction) in product(list_redshift, list_data_fraction):
        cmd = (
            f"python -W ignore run.py --train_rnn "
            f"--model bayesian "
            f"--source_data saltfit "
            f"--dump_dir {dump_dir} "
            f"--data_fraction {data_fraction} "
            f"--batch_size 1024 "
            f"--monitor_interval 5 "
            f"--log_sigma1 -1 "
            f"--log_sigma2 -7 "
            f"--rho_scale_lower 4 "
            f"--rho_scale_upper 3 "
            f"--log_sigma1_output -0.5 "
            f"--log_sigma2_output -0.1 "
            f"--rho_scale_lower_output 3 "
            f"--rho_scale_upper_output 2 "
        )
        if redshift is not None:
            cmd += f" --redshift {redshift} "

        run_cmd(cmd, debug, seed)


def run_representative(dump_dir, debug, seed):

    lu.print_green(f"SEED {seed}: REPRESENTATIVE")

    # Get all trained models with saltfit + DF == 1
    list_saltfit = (Path(dump_dir) / "models").glob("**/*saltfit_DF_1.0*.pt")

    # Only run representative on models which have a photometry counterpart
    list_saltfit = [
        m
        for m in list_saltfit
        if Path(str(m).replace("saltfit_DF_1.0", "photometry_DF_0.43")).exists()
    ]

    for model_file in list_saltfit:

        # Validate saltfit model on photometry test set
        cmd = f"python -W ignore run.py --validate_rnn "
        cmd += f"--override_source_data photometry " f"--model_files {model_file} "

        run_cmd(cmd, debug, seed)


def run_performance(dump_dir, debug, seed):
    """Performance and plots
    """

    lu.print_green(f"SEED {seed}: PERFORMANCE")

    # Make sure all PRED files have accompanying METRICS file
    list_predictions = (Path(dump_dir) / "models").glob("**/*PRED*.pickle")
    prediction_files_str = " ".join(list(map(str, list_predictions)))

    cmd = f"python run.py --metrics --prediction_files {prediction_files_str} "
    run_cmd(cmd, debug, seed)

    # Aggregate all metrics
    cmd = "python -W ignore run.py --performance " f"--dump_dir {dump_dir} "
    run_cmd(cmd, debug, seed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SNIa classification")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    default_dump_dir = Path(dir_path).parent / "snndump"
    parser.add_argument(
        "--dump_dir",
        type=str,
        default=default_dump_dir,
        help="Default path where data and models are dumped",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Switch to debug mode: will run dummy experiments to quickly check the whole pipeline",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=LIST_SEED,
        nargs="+",
        choices=LIST_SEED,
        help="Seed with which to run the experiments",
    )
    args = parser.parse_args()

    list_seeds = args.seeds[:2] if args.debug else args.seeds

    for seed in list_seeds:

        if seed == list_seeds[0]:
            ############################
            # Data
            ############################
            run_data(args.dump_dir, args.debug, seed)

            ####################
            # Misc. benchmarks
            ####################
            run_speed(args.dump_dir, args.debug, seed)
            run_benchmark_cyclic(args.dump_dir, args.debug, seed)

            ##################
            # Hyperparams
            ##################
            run_baseline_hp(args.dump_dir, args.debug, seed)
            run_variational_hp(args.dump_dir, args.debug, seed)
            run_bayesian_hp(args.dump_dir, args.debug, seed)

        ##################
        # Baseline models
        ##################
        run_baseline(args.dump_dir, args.debug, seed)

        ##################
        # Bayesian models
        ##################
        run_variational_best(args.dump_dir, args.debug, seed)
        run_bayesian_best(args.dump_dir, args.debug, seed)

    ##########################
    # Metrics, science plots
    ##########################
    run_representative(args.dump_dir, args.debug, seed)
    run_performance(args.dump_dir, args.debug, seed)
