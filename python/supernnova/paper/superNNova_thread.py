import os
import pandas as pd
from pathlib import Path
import supernnova.conf as conf
from . import superNNova_plots as sp
from . import superNNova_metrics as sm
from ..utils import logging_utils as lu
from ..visualization import early_prediction

"""
Obtaining metrics and plots for SuperNNova paper

Selection of models is hard coded

Code is far from optimized
"""

"""
Best performing algorithms in SuperNNova
"""
Base = (
    "DES_vanilla_CLF_2_R_None_photometry_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean_C"
)
Var = "DES_variational_CLF_2_R_None_photometry_DF_1.0_N_global_lstm_32x2_0.01_128_True_mean_C_WD_1e-07"
BBB = "DES_bayesian_CLF_2_R_None_photometry_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean_Bayes_0.75_-1.0_-7.0_4.0_3.0_-0.5_-0.1_3.0_2.0"  # _KL_0.1"
RF = "DES_randomforest_CLF_2_R_None_saltfit_DF_1.0_N_global"
list_models = [RF, Base, Var, BBB]
list_models_rnn = [Base, Var, BBB]

# useful formats
Base_salt = Base.replace("photometry", "saltfit")


def SuperNNova_stats_and_plots(settings):
    """ Reproduce stats and plots used for SuperNNova paper.
    BEWARE: Selection is hardcoded

    Args:
        settings (ExperimentSettings): custom class to hold hyperparameters
    """

    # Load summary statistics
    df_stats = pd.read_csv(Path(settings.stats_dir) / "summary_stats.csv")

    # Create latex tables
    # sm.create_accuracy_latex_tables(df_stats, settings)

    # Rest of stats and plots in paper
    # can be ran in debug mode: only printing model names
    # or in no plot mode: only printing stats
    SuperNNova_stats_and_plots_thread(df_stats, settings, plots=True, debug=False)


def SuperNNova_stats_and_plots_thread(df, settings, plots=True, debug=False):
    """Stats quoted in paper which are not in the latex tables and plots

    Args:
        df (pandas.DataFrame) : summary statistics df
        settings (ExperimentSettings): custom class to hold hyperparameters
        plots (Boolean optional): make pltos or only printout stats 
        debug (Boolean optional): only print tasks
    Returns:
        printout: stats as organized in paper
        figures (png) : figures for paper at settings.dump_dir/figures/
        lightcurves (png): lightcurves used on paper at settings.dump_dir/lightcurves/modelname.*png
    """

    """
    Ordered as in paper
    """
    pd.set_option("max_colwidth", 1000)
    print(lu.str_to_greenstr(f"STATISTICS USED IN SUPERNNOVA"))

    # Baseline experiments
    baseline(df, settings, plots, debug)
    # Bayesian experiments
    df_delta, df_delta_ood = sm.get_delta_metrics(df, settings)
    bayesian(df, df_delta, df_delta_ood, settings, plots, debug)
    # Towards statistical analyses/cosmology
    towards_cosmo(df, df_delta, df_delta_ood, settings, plots, debug)


def baseline(df, settings, plots, debug):
    """
    Baseline RNN
    """
    # 0. Figure example
    if plots:
        print(
            lu.str_to_yellowstr("Plotting candidates for Baseline binary (Figure 2.)")
        )
        model_file = f"{settings.models_dir}/{Base.replace('DES_vanilla_','vanilla_S_0_')}/{Base.replace('DES_vanilla_','vanilla_S_0_')}.pt"
        if os.path.exists(model_file):
            if debug:
                print(model_file)
            else:
                model_settings = conf.get_settings_from_dump(model_file)
                early_prediction.make_early_prediction(
                    model_settings, nb_lcs=20, do_gifs=True
                )
        else:
            print(lu.str_to_redstr(f"File not found {model_file}"))

    # 1. Hyper-parameters
    # saltfit, DF 0.2
    sel_criteria = ["DES_vanilla_CLF_2_R_None_saltfit_DF_0.2"]
    print(lu.str_to_bluestr(f"Hyperparameters {sel_criteria}"))
    if not debug:
        sm.get_metric_ranges(df, sel_criteria)

    # 2. Normalization
    # saltfit, DF 0.5
    sel_criteria = Base_salt.replace("DF_1.0", "DF_0.5").split("global")
    print(lu.str_to_bluestr(f"Normalization {sel_criteria}"))
    if not debug:
        df_sel = sm.get_metric_ranges(df, sel_criteria)

    # 3. Comparing with other methods
    print(lu.str_to_bluestr(f"Other methods:"))
    # Figure: accuracy vs. number of SNe
    if plots:
        print(lu.str_to_yellowstr("Plotting accuracy vs. SNe (Figure 3.)"))
        if not debug:
            sp.performance_plots(settings)
    # baseline best, saltfit
    if debug:
        print(Base_salt)
        print(Base_salt.replace("DF_1.0", "DF_0.05"))
        print(Base_salt.replace("DF_1.0", "DF_0.05").replace("None", "zpho"))
    else:
        sm.acc_auc_df(df, [Base_salt], data="saltfit")

        # RF and baseline comparisson with Charnock Moss
        sm.acc_auc_df(
            df,
            [
                RF,
                Base_salt.replace("DF_1.0", "DF_0.05"),
                Base_salt.replace("DF_1.0", "DF_0.05").replace("None", "zpho"),
            ],
        )

    # 4. Redshift, contamination
    # baseline saltfit, 1.0, all redshifts
    sel_criteria = Base_salt.split("None")
    print("salt")
    if debug:
        print(sel_criteria, Base.split("None"))
    if not debug:
        sm.print_contamination(df, sel_criteria, settings, data="saltfit")
    print("photometry")
    if not debug:
        sm.print_contamination(df, Base.split("None"), settings, data="photometry")

    # Multiclass
    # Plotting Confusion Matrix for just one seed
    if plots:
        print(
            lu.str_to_yellowstr(
                "Plotting confusion matrix for multiclass classification (Figure 4.)"
            )
        )
        for target in [3, 7]:
            settings.prediction_files = [
                settings.models_dir
                + "/"
                + model.strip("DES_").replace("CLF_2", f"S_0_CLF_{target}")
                + "/"
                + model.replace("DES_", "PRED_DES_").replace(
                    "CLF_2", f"S_0_CLF_{target}"
                )
                + ".pickle"
                for model in [Base, Var, BBB]
            ]
            if debug:
                print(settings.prediction_files)
            else:
                sp.science_plots(settings, onlycnf=True)
        # Uncomment to see some examples of seven-way classification
        # print(lu.str_to_yellowstr("Plotting candidates for multiclass classification"))
        # model_file = f"{settings.models_dir}/{Base.replace('DES_vanilla_','vanilla_S_0_').replace('CLF_2','CLF_7')}/{Base.replace('DES_vanilla_','vanilla_S_0_').replace('CLF_2','CLF_7')}.pt"
        # if os.path.exists(model_file):
        #     model_settings = conf.get_settings_from_dump(model_file)
        #     early_prediction.make_early_prediction(model_settings, nb_lcs=20)
        # else:
        #     print(lu.str_to_redstr(f"File not found {model_file}"))


def bayesian(df, df_delta, df_delta_ood, settings, plots, debug):
    """
    Bayesian RNNs: BBB and Variational
    """

    # 2. Variational hyper-parameters
    sel_criteria = ["DES_variational_CLF_2_R_None_saltfit_DF_0.2_N_global_lstm_32x2_"]
    print(lu.str_to_bluestr(f"Hyperparameters {sel_criteria}"))
    if not debug:
        sm.get_metric_ranges(df, sel_criteria)

    # 3. BBB hyper-parameters
    sel_criteria = ["DES_bayesian_CLF_2_R_None_saltfit_DF_0.2_N_global_lstm_32x2_"]
    print(lu.str_to_bluestr(f"Hyperparameters {sel_criteria}"))
    if not debug:
        sm.get_metric_ranges(df, sel_criteria)

    # 2 and 3. Best models
    print(lu.str_to_bluestr("Best performing Bayesian accuracies"))
    if debug:
        print(
            [
                Var,
                BBB,
                Var.replace("None", "zpho"),
                BBB.replace("None", "zpho"),
                Var.replace("None", "zspe"),
                BBB.replace("None", "zspe"),
            ]
        )
    else:
        sm.acc_auc_df(
            df,
            [
                Var,
                BBB,
                Var.replace("None", "zpho"),
                BBB.replace("None", "zpho"),
                Var.replace("None", "zspe"),
                BBB.replace("None", "zspe"),
            ],
        )
    # contamination
    # baseline saltfit, 1.0, all redshifts
    for model in [Var, BBB]:
        sel_criteria = model.split("None")
        if debug:
            print("contamination")
            print(sel_criteria)
        else:
            sm.print_contamination(df, sel_criteria, settings, data="photometry")

    # 4. Uncertainties
    print(lu.str_to_bluestr("Best performing Bayesian uncertainties"))
    print("Epistemic behaviour")
    for model in [Var, BBB]:
        m_right = model.replace("photometry", "saltfit")
        m_left = model.replace("photometry", "saltfit").replace("DF_1.0", "DF_0.5")
        print("salt", m_left, m_right)
        df_sel = df_delta[
            (df_delta["model_name_left"] == m_left)
            & (df_delta["model_name_right"] == m_right)
        ]
        if not debug:
            sm.nice_df_print(
                df_sel,
                keys=[
                    "all_accuracy_mean_delta",
                    "mean_all_class0_std_dev_mean_delta",
                    "all_entropy_mean_delta",
                ],
            )
        m_right = model
        m_left = model.replace("DF_1.0", "DF_0.43")
        print("complete", m_left, m_right)
        df_sel = df_delta[
            (df_delta["model_name_left"] == m_left)
            & (df_delta["model_name_right"] == m_right)
        ]
        if not debug:
            sm.nice_df_print(
                df_sel,
                keys=[
                    "all_accuracy_mean_delta",
                    "mean_all_class0_std_dev_mean_delta",
                    "all_entropy_mean_delta",
                ],
            )

    print("Uncertainty size")
    df_sel = df[df["model_name_noseed"].isin([Var, BBB])]
    df_sel = df_sel.round(4)
    if not debug:
        sm.nice_df_print(
            df_sel, keys=["mean_all_class0_std_dev_mean", "mean_all_class0_std_dev_std"]
        )

    if plots:
        print(
            lu.str_to_yellowstr(
                "Plotting candidates for multiclass classification (Fig. 5)"
            )
        )
        for model in [Var, BBB]:
            model_file = (
                f"{settings.models_dir}/"
                + model.replace("DES_", "").replace("CLF_2", "S_0_CLF_7")
                + "/"
                + model.replace("DES_", "").replace("CLF_2", "S_0_CLF_7")
                + ".pt"
            )
            if os.path.exists(model_file):
                if debug:
                    print(model_file)
                else:
                    model_settings = conf.get_settings_from_dump(model_file)
                    early_prediction.make_early_prediction(
                        model_settings, nb_lcs=20, do_gifs=True
                    )
            else:
                print(lu.str_to_redstr(f"File not found {model_file}"))
        print(lu.str_to_yellowstr("Adding gifs for binary classification"))
        for model in [Var, BBB]:
            model_file = (
                f"{settings.models_dir}/"
                + model.strip("DES_").replace("CLF_2", "S_0_CLF_2")
                + "/"
                + model.strip("DES_").replace("CLF_2", "S_0_CLF_2")
                + ".pt"
            )
            if os.path.exists(model_file):
                if debug:
                    print(model_file)
                else:
                    model_settings = conf.get_settings_from_dump(model_file)
                    early_prediction.make_early_prediction(
                        model_settings, nb_lcs=10, do_gifs=True
                    )


def towards_cosmo(df, df_delta, df_delta_ood, settings, plots, debug):
    """
    Towards cosmology
    """
    # 1. Calibration
    print(lu.str_to_bluestr("Calibration"))
    df_sel = df.copy()
    df_sel = df_sel.round(4)
    # rf can't be done with photometry
    df_sel = df_sel[
        (
            df_sel["model_name_noseed"].isin(
                [l.replace("photometry", "saltfit") for l in list_models]
            )
        )
        & (df_sel["source_data"] == "saltfit")
    ]
    print("saltfit")
    if not debug:
        sm.nice_df_print(
            df_sel,
            keys=[
                "model_name_noseed",
                "calibration_dispersion_mean",
                "calibration_dispersion_std",
            ],
        )
    # without rf
    print("photometry")
    df_sel = df.copy()
    df_sel = df_sel[
        (df_sel["model_name_noseed"].isin([Base, Var, BBB]))
        & (df_sel["source_data"] == "photometry")
    ]
    if not debug:
        sm.nice_df_print(
            df_sel,
            keys=[
                "model_name_noseed",
                "calibration_dispersion_mean",
                "calibration_dispersion_std",
            ],
        )
    # Calibration vs. training set size
    # using salt
    print(lu.str_to_bluestr("Calibration vs. data set size"))
    print("Baseline")
    sel_criteria = Base_salt.split("DF_1.0")
    if debug:
        print(sel_criteria)
    else:
        sm.get_metric_ranges(
            df, sel_criteria, metric="calibration_dispersion", round_output=5
        )
    # Calibration vs. dataset nature
    sel_criteria = [
        "DES_vanilla_CLF_2_R_None_saltfit_DF_1.0_N_global_lstm_32x2_0.05_128_True_mean_C"
    ]
    if debug:
        print(sel_criteria)
    else:
        sm.get_metric_ranges(
            df, sel_criteria, metric="calibration_dispersion", round_output=5
        )
    # Calibration figure
    if plots:
        print(lu.str_to_yellowstr("Plotting reliability diagram (Figure 6)"))
        tmp_pred_files = settings.prediction_files
        settings.prediction_files = [
            settings.models_dir
            + "/"
            + model.strip("DES_")
            .replace("CLF_2", "S_0_CLF_2")
            .replace("photometry", "saltfit")
            + "/"
            + model.replace("DES_", "PRED_DES_")
            .replace("CLF_2", "S_0_CLF_2")
            .replace("photometry", "saltfit")
            + ".pickle"
            for model in [RF, Base, Var, BBB]
        ]
        if debug:
            print(settings.prediction_files)
        else:
            sp.plot_calibration(settings)
        settings.prediction_files = tmp_pred_files

    # 2. Representativeness
    print(lu.str_to_bluestr("Representativeness"))
    for model in [Base, Var, BBB]:
        m_left = model.replace("photometry", "saltfit")
        m_right = model.replace("DF_1.0", "DF_0.43")
        print(m_left, m_right)
        df_sel = df_delta[
            (df_delta["model_name_left"] == m_left)
            & (df_delta["model_name_right"] == m_right)
        ]
        if not debug:
            sm.nice_df_print(
                df_sel,
                keys=[
                    "all_accuracy_mean_delta",
                    "mean_all_class0_std_dev_mean_delta",
                    "all_entropy_mean_delta",
                ],
            )

    # 3. OOD
    print(lu.str_to_bluestr("Out-of-distribution light-curves"))
    # OOD type assignement figure
    if plots:
        print(lu.str_to_yellowstr("Plotting OOD classification percentages (Figure 8)"))
        if not debug:
            sp.create_OOD_classification_plots(df, list_models_rnn, settings)
    # Get entropy
    print("binary")
    df_sel = df_delta_ood[df_delta_ood["model_name_noseed"].isin(list_models_rnn)]
    if not debug:
        sm.nice_df_print(df_sel)
    print("ternary")
    list_models_sel = [l.replace("CLF_2", "CLF_3") for l in list_models_rnn]
    df_sel = df_delta_ood[df_delta_ood["model_name_noseed"].isin(list_models_sel)]
    if not debug:
        sm.nice_df_print(df_sel)
    print("seven-way")
    list_models_sel = [l.replace("CLF_2", "CLF_7") for l in list_models_rnn]
    df_sel = df_delta_ood[df_delta_ood["model_name_noseed"].isin(list_models_sel)]
    if not debug:
        sm.nice_df_print(df_sel)
    if plots:
        print(
            lu.str_to_yellowstr(
                "Plotting OOD candidates with seven-way classification (Figure 9)"
            )
        )
        model_file = (
            f"{settings.models_dir}/"
            + Var.replace("DES_variational_", "variational_S_0_").replace(
                "CLF_2", "CLF_7"
            )
            + "/"
            + Var.replace("DES_variational_", "variational_S_0_").replace(
                "CLF_2", "CLF_7"
            )
            + ".pt"
        )
        if os.path.exists(model_file):
            if debug:
                print(model_file)
            else:
                model_settings = conf.get_settings_from_dump(model_file)
                early_prediction.make_early_prediction(model_settings, nb_lcs=20)
        else:
            print(lu.str_to_redstr(f"File not found {model_file}"))

    # 4. Cosmology
    print(lu.str_to_bluestr("SNe Ia for cosmology"))
    # Plotting Hubble residuals adn other science plots for just one seed
    if plots:
        print(lu.str_to_yellowstr("Plotting Hubble residuals (Figures 10 and 11)"))
        tmp_pred_files = settings.prediction_files
        settings.prediction_files = [
            settings.models_dir
            + "/"
            + model.strip("DES_").replace("CLF_2", "S_0_CLF_2")
            + "/"
            + model.replace("DES_", "PRED_DES_").replace("CLF_2", "S_0_CLF_2")
            + ".pickle"
            for model in [Base, Var, BBB]
        ]
        if debug:
            print(settings.prediction_files)
        else:
            sp.science_plots(settings)
        settings.prediction_files = tmp_pred_files
