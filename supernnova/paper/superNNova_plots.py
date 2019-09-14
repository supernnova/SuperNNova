import re
import os
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pylab as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from astropy.cosmology import FlatLambdaCDM
from sklearn.metrics import confusion_matrix

from ..utils import data_utils as du
from ..utils import logging_utils as lu
from ..utils import performance_utils as pu
from ..utils import visualization_utils as vu

# Plotting styles
from ..utils.visualization_utils import (
    ALL_COLORS,
    BI_COLORS,
    CONTRAST_COLORS,
    MARKER_DIC,
    FILL_DIC,
    MARKER_LIST,
    CMAP,
    PATTERNS,
    get_model_visualization_name,
)

cosmo = FlatLambdaCDM(H0=70, Om0=0.295)

#################
# Plot utils
#################


def class_target_decode(target):
    if target == 2:
        return "Binary"
    if target == 3:
        return "Ternary"
    else:
        return "Seven-way"


def dist_mu(redshift):
    mu = cosmo.distmod(redshift)

    return mu.value


def create_sigma_df(df_grouped, class_=0):
    """From grouped prediction df create a df with sigma values
    """
    sigma_all_list = []
    sigma_peak_list = []
    snid_list = []
    pred_class_list = []
    for SNID, SNID_df in df_grouped:
        arr_proba = SNID_df[f"all_class{class_}"]
        perc_16 = np.percentile(arr_proba, 16)
        perc_84 = np.percentile(arr_proba, 84)
        sigma_all_list.append(perc_84 - perc_16)

        arr_proba = SNID_df[f"PEAKMJD_class{class_}"]
        perc_16 = np.percentile(arr_proba, 16)
        perc_84 = np.percentile(arr_proba, 84)
        sigma_peak_list.append(perc_84 - perc_16)
        snid_list.append(SNID)

        # get predicition for this SNID
        k_all_probas = [k for k in SNID_df.keys() if "all_class" in k]
        median_prob_forSNID = SNID_df[k_all_probas].median()
        pred_class = median_prob_forSNID.idxmax()
        arr_proba = SNID_df[pred_class]
        # get sigma for this class
        perc_16 = np.percentile(arr_proba, 16)
        perc_84 = np.percentile(arr_proba, 84)
        pred_class_list.append(perc_84 - perc_16)

    df = pd.DataFrame()
    df["SNID"] = np.array(snid_list)
    df["sigma_all"] = np.array(sigma_all_list)
    df["sigma_peak"] = np.array(sigma_peak_list)
    df["pred_sigma_all"] = np.array(pred_class_list)
    return df


def plot_acc_vs_nsn(df, settings):
    """Plot accuracy vs number of SNe used for training

    Args:
        df (DataFrame): prediction dataframe
        settings (ExperimentSettings): custom class to hold hyperparameters

    Returns:
        figure (png)
    """
    plt.clf()
    fig = plt.figure()
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    ax1.set_axisbelow(True)

    models_list = ["randomforest", "vanilla"]
    redshift_list = df["redshift"].unique()

    label_dic = {"randomforest": "Random Forest", "vanilla": "Baseline RNN"}

    group_cols = ["model_name_noseed", "model_type", "redshift", "data_fraction"]
    keep_cols = group_cols + ["all_accuracy"]

    # Cast to float for groupby operation (all_accuracy is type `O`)
    df.all_accuracy = df.all_accuracy.astype(float)

    df_errorbars = (
        df[keep_cols]
        .groupby(group_cols)
        .mean()
        .rename(columns={"all_accuracy": "all_accuracy_mean"})
        .reset_index()
    )
    df_errorbars["all_accuracy_std"] = (
        df[keep_cols]
        .groupby(group_cols)
        .std()
        .rename(columns={"all_accuracy": "all_accuracy_std"})
        .reset_index()["all_accuracy_std"]
    )

    for i, basemodel in enumerate(models_list):
        for z in redshift_list:
            df_sel = df_errorbars[
                (df_errorbars["model_type"] == basemodel)
                & (df_errorbars["redshift"] == z)
            ]
            # Plot these independently to avoid polluting legend
            ax1.errorbar(
                df_sel["data_fraction"],
                df_sel["all_accuracy_mean"],
                yerr=df_sel["all_accuracy_std"],
                c=CONTRAST_COLORS[i],
                fmt="none",
                zorder=3 if basemodel == "vanilla" else 1,
            )
            ax1.plot(
                df_sel["data_fraction"],
                df_sel["all_accuracy_mean"],
                label=label_dic[basemodel],
                marker=MARKER_DIC[basemodel],
                c=CONTRAST_COLORS[i],
                fillstyle=FILL_DIC[z],
                lw=0,
                markersize=10,
                markeredgewidth=1.5,
            )
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            lw=0,
            color="indigo",
            label="Baseline RNN",
            markerfacecolor="w",
            markersize=12,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            lw=0,
            color="darkorange",
            label="Random Forest",
            markerfacecolor="w",
            markersize=12,
        ),
    ]

    ax1.legend(handles=legend_elements, loc=4)
    ax1.set_ylabel("accuracy", fontsize=18)
    ax1.set_ylim(91, 100)
    ax1.set_xlim(0.025)
    ax1.set_xlabel("# SNe for training", fontsize=18)

    # exchange axis and reformat
    ax2 = ax1.twiny()
    ax1Xs = [round(i, 1) for i in ax1.get_xticks()]
    ax2Xs = []
    for X in ax1Xs:
        # BEWARE: only valid with SALTfitted sample
        ax2Xs.append("{:0.1e}".format(int(X * 881_969 * 0.8)))

    ax1.set_xticklabels(ax2Xs)
    ax2.set_xticks(ax1Xs)
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xticklabels(ax1Xs)

    title = ax1.set_title("data fraction", fontsize=18)
    title.set_y(1.1)
    plt.tight_layout()

    fig.subplots_adjust(top=0.85)
    fig.savefig(f"{settings.figures_dir}/accuracy_vs_nSN.png")
    plt.close()
    plt.clf()


def plot_calibration(settings):
    """Plot reliability diagram

    Args:
        settings (ExperimentSettings): custom class to hold hyperparameters

    Returns:
        figure (png)
    """

    if len(settings.prediction_files) == 0:
        print(
            lu.str_to_yellowstr("Warning: no prediction files provided. Not plotting")
        )
        return
    else:
        metric_files = [
            f.replace("PRED", "METRICS")
            for f in settings.prediction_files
            if os.path.exists(f.replace("PRED", "METRICS"))
        ]
        tmp_not_found = [
            f.replace("PRED", "METRICS")
            for f in settings.prediction_files
            if not os.path.exists(f.replace("PRED", "METRICS"))
        ]
        if len(tmp_not_found) > 0:
            print(lu.str_to_redstr(f"Files not found {tmp_not_found}"))

    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax11 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    plot_path = f"{settings.figures_dir}/calibration"
    Path(plot_path).mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(metric_files):
        df = pd.read_pickle(f)
        mean_bins, TPF = df["calibration_mean_bins"][0], df["calibration_TPF"][0]
        model_name = df["model_name"][0]

        calibration_dispersion = TPF - mean_bins
        ax1.plot(
            mean_bins,
            TPF,
            "s-",
            color=ALL_COLORS[i],
            label=get_model_visualization_name(df["model_name"][0]),
            marker=MARKER_LIST[i],
        )
        ax11.scatter(
            mean_bins,
            calibration_dispersion,
            color=ALL_COLORS[i],
            marker=MARKER_LIST[i],
        )

    ax1.set_ylabel("fraction of positives (fP)", fontsize=18)
    ax1.set_ylim([-0.05, 1.05])
    ax11.set_xlabel(f"mean predicted probability", fontsize=18)
    ax1.legend()
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax11.set_ylabel("residual fP", fontsize=18)
    ax11.set_ylim([-0.2, 0.2])
    ax11.plot([0, 1], np.zeros(len([0, 1])), "k:")
    ax11.plot([0, 1], 0.1 * np.ones(len([0, 1])), ":", color="grey")
    ax11.plot([0, 1], -0.1 * np.ones(len([0, 1])), ":", color="grey")
    plt.setp(ax11.get_xticklabels(), visible=False)

    if len(metric_files) == 1:
        nameout = f"{plot_path}/calib_{model_name}.png"
    else:
        nameout = f"{plot_path}/calib_multimodels.png"

    plt.savefig(nameout)
    plt.close()
    plt.clf()


def plot_confusion_matrix(
    settings, cm, classes, normalize=False, cmap=vu.CMAP, nameout=None
):
    """Plot confusion matrix
    Based on sklearn tutorial

    Args:
        settings (ExperimentSettings): custom class to hold hyperparameters
        cm (np.array): confusion matrix
        classes (list): name of classes in cm
        normalize (optional): Normalization can be applied by setting `normalize=True`
        cmap (colormap)
        nameout (optional): out name for figure

    Returns:
        figure (png)
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    cbar = plt.colorbar()
    cbar.set_clim(0, 1.0)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label", fontsize=18)
    plt.xlabel("Predicted label", fontsize=18)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plot_path = f"{settings.figures_dir}/cnf_matrix"
    os.makedirs(plot_path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{plot_path}/{nameout}.png")


def multiplot_violin_paper(df, fname, settings):
    """Plot data properties as violin plots.
    
    Far from optimized code: seaborn does not make this easy so added
    a lot of formatting using raw matplotlib commands

    Args:
        df (DataFrame): prediction dataframe
        fname (filename):
        settings (ExperimentSettings): custom class to hold hyperparameters

    Returns:
        figure (png)
    """

    # Set up the axes with gridspec
    plt.clf()
    fig = plt.figure()
    grid = plt.GridSpec(2, 4, hspace=0.2, wspace=0.2)
    ax_00 = fig.add_subplot(grid[0, 0])
    ax_01 = fig.add_subplot(grid[0, 1], sharey=ax_00)
    ax_02 = fig.add_subplot(grid[0, 2], sharey=ax_00)
    ax_03 = fig.add_subplot(grid[0, 3], sharey=ax_00)
    ax_1 = fig.add_subplot(grid[1, :])

    axes = [ax_00, ax_01, ax_02, ax_03]

    # Ia vs non Ia
    sns.set_palette(sns.color_palette(BI_COLORS))
    g = sns.violinplot(
        x="target",
        y="SIM_PEAKMAG_g",
        hue="salt",
        data=df,
        split=True,
        ax=axes[0],
        inner="quartile",
    )
    g.set_xlabel("")
    g.legend_.remove()
    g.yaxis.set_tick_params(labelsize=14)
    g.set_title("g", fontsize=14)
    g.set_ylabel("magnitude", fontsize=14)
    g.set_ylim(20, 28)
    g.spines["right"].set_visible(False)
    g.spines["top"].set_visible(False)
    g.set_xticklabels(["Ia", "nonIa"], fontsize=14)

    g = sns.violinplot(
        x="target",
        y="SIM_PEAKMAG_i",
        hue="salt",
        data=df,
        split=True,
        ax=axes[1],
        inner="quartile",
    )
    g.set_xlabel("")
    g.set_ylabel("")
    g.legend_.remove()
    g.yaxis.set_ticks_position("none")
    g.set_title("i", fontsize=14)
    g.set_ylim(20, 28)
    g.spines["right"].set_visible(False)
    g.spines["top"].set_visible(False)
    g.spines["left"].set_visible(False)
    g.set_xticklabels(["Ia", "nonIa"], fontsize=14)
    plt.setp(axes[1].get_yticklabels(), visible=False)

    g = sns.violinplot(
        x="target",
        y="SIM_PEAKMAG_r",
        hue="salt",
        data=df,
        split=True,
        ax=axes[2],
        inner="quartile",
    )
    g.legend_.remove()
    g.yaxis.set_ticks_position("none")
    g.set_xlabel("")
    g.set_ylabel("")
    g.set_title("r", fontsize=18)
    g.xaxis.set_tick_params(labelsize=14)
    g.set_ylim(20, 28)
    g.spines["right"].set_visible(False)
    g.spines["top"].set_visible(False)
    g.spines["left"].set_visible(False)
    g.set_xticklabels(["Ia", "nonIa"], fontsize=14)
    plt.setp(axes[2].get_yticklabels(), visible=False)

    g = sns.violinplot(
        x="target",
        y="SIM_PEAKMAG_z",
        hue="salt",
        data=df,
        split=True,
        ax=axes[3],
        inner="quartile",
    )
    g.legend_.remove()
    g.yaxis.set_ticks_position("none")
    g.set_title("z", fontsize=14)
    g.set_xlabel("")
    g.set_ylabel("")
    g.set_ylim(20, 28)
    g.spines["right"].set_visible(False)
    g.spines["top"].set_visible(False)
    g.spines["left"].set_visible(False)
    g.set_xticklabels(["Ia", "nonIa"], fontsize=14)
    plt.setp(axes[3].get_yticklabels(), visible=False)

    # redshift
    g = sns.violinplot(
        x="SNTYPE",
        y="SIM_REDSHIFT_CMB",
        hue="salt",
        data=df,
        split=True,
        ax=ax_1,
        inner="quartile",
    )
    g.set_ylabel("simulated redshift", fontsize=14)
    g.set_xlabel("")
    g.set_ylim(0, 1.0)
    g.set_xticklabels([a for a in settings.sntypes.values()], fontsize=14)
    g.xaxis.set_tick_params(labelsize=14)
    g.yaxis.set_tick_params(labelsize=14)
    g.legend_.remove()
    g.spines["right"].set_visible(False)
    g.spines["top"].set_visible(False)
    g.spines["bottom"].set_visible(False)

    plt.savefig(f"{settings.figures_dir}/multiviolin_{fname}.png")
    plt.close()
    del fig


def binned_2d(
    bin_centers,
    y_dic,
    xname,
    yname,
    label_list,
    color_sequence,
    MARKER_LIST,
    nameout,
    extra_line=None,
):
    """Plot scatter plot

    Args:
        bin_centers (np.array): x centers
        y_dic (dict): y values
        xname (str): xlabel
        yname (str): ylabel
        label_list (list(str)): for legend
        color_sequence (list(str)): color code
        MARKER_LIST (list(str)): marker code
        nameout (str): out name of figure (.png)
        extra_line (optional): extra line on 0 and 100 (y)

    Returns:
        figure (png)
    """
    plt.clf()
    fig = plt.figure()
    ax = plt.gca()
    for i, k in enumerate(y_dic.keys()):
        ax.scatter(
            bin_centers,
            y_dic[k],
            color=color_sequence[i],
            label=label_list[i],
            marker=MARKER_LIST[i],
        )
    if extra_line:
        ax.plot(
            ax.get_xlim(), np.zeros(len(ax.get_xlim())), color="grey", linestyle="--"
        )
        ax.plot(
            ax.get_xlim(),
            100 * np.ones(len(ax.get_xlim())),
            color="grey",
            linestyle="--",
        )
    ax.set_ylim(0.01, 1)
    ax.set_yscale("log")
    ax.set_xlabel(xname, fontsize=18)
    ax.set_ylabel(yname, fontsize=18)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.tight_layout()
    plt.savefig(nameout)
    plt.clf()
    plt.close()
    del fig


def plot_acc_matrix(acc_dic, n_meas_dic, settings, nameout):
    """Plot accuracy matrix

    Args:
        acc_dic (dict): with np.arrays of accuracy
        n_meas_dic (dict): with n measurements required for this accuracy
        settings (ExperimentSettings): custom class to hold hyperparameters
        nameout (str): outname for plot
    Returns:
        figure (png)
    """

    plt.clf()
    fig = plt.figure(figsize=(4, 6))
    ax = fig.add_subplot(111)

    min_length = min([len(acc_dic[band]) for band in settings.list_filters])
    x_labels = settings.list_filters
    y_labels = [int(round(n)) for n in n_meas_dic["g"]][:min_length]
    CMAP.set_bad(color="white")
    acc_mat = np.vstack([acc_dic[band][:min_length] for band in settings.list_filters])
    acc_mat = acc_mat.transpose()
    cax = ax.matshow(acc_mat, cmap=CMAP, vmin=50, vmax=100.0)
    fig.colorbar(cax)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xticks(np.arange(0, len(x_labels), 1))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(0, len(y_labels), 1))
    ax.set_yticklabels(y_labels)
    plt.tight_layout()
    plt.savefig(nameout)
    plt.close()
    del fig


def plot_HDres_histos_vs_z(
    df,
    nameout,
    threshold_var="class0",
    threshold_list=[0.5, 0.7, 0.9],
    threshold_sign=">",
):
    """Plot Hubble diagram residuals and histograms
    selects class sample and performas HD

    Args:
        df (DataFrame): predictions
        nameout (str): outname figure
        threshold_var (str): which class used for sample selection
        threshold_list (list): list of probability threshold for selecting sample
        threshold_sign (str): sign of the probability threshold (e.g. ">" )
    Returns:
        figure (png)
    """

    P = df[df["class0"] > 0.5]
    Ias = df[df["target"] == 0]

    TP = P[P["target"] == 0]
    FP = P[P["target"] != 0]

    sel_TP_dic = {}
    sel_FP_dic = {}
    for t in threshold_list:
        if threshold_sign == ">":
            sel_TP_dic[t] = TP[TP[threshold_var] > t]
            sel_FP_dic[t] = FP[FP[threshold_var] > t]
        else:
            sel_TP_dic[t] = TP[TP[threshold_var] < t]
            sel_FP_dic[t] = FP[FP[threshold_var] < t]

    plt.clf()
    cm = CMAP
    fig = plt.figure(figsize=(14, 14))
    # gs = gridspec.GridSpec(4, 2, width_ratios=[3, 1], height_ratios=[2, 2, 1, 1])
    # gs.update(wspace=0.1, hspace=0.3)

    # # gridspec init
    # ax00 = plt.subplot(gs[0, 0])  # Hres Ia
    # ax10 = plt.subplot(gs[1, 0], sharex=ax00)  # Hres CC
    # ax20 = plt.subplot(gs[2:, 0], sharex=ax00)  # efficiency
    # ax01 = plt.subplot(gs[0, 1], sharey=ax00)  # histo Ia
    # ax11 = plt.subplot(gs[1, 1], sharey=ax10)  # histo CC
    # ax21 = plt.subplot(gs[2, 1])  # histo x1
    # ax31 = plt.subplot(gs[3, 1])  # histo c
    gs = gridspec.GridSpec(3, 3, height_ratios=[2, 2, 1])
    # gs.update(wspace=0.2, hspace=0.1)

    # gridspec init
    ax00 = plt.subplot(gs[0, 0:2])  # Hres Ia
    ax10 = plt.subplot(gs[1, 0:2], sharex=ax00)  # Hres CC
    ax20 = plt.subplot(gs[2, 0])  # redshift dist
    ax01 = plt.subplot(gs[0, 2], sharey=ax00)  # histo Ia
    ax11 = plt.subplot(gs[1, 2], sharey=ax10)  # histo CC
    ax21 = plt.subplot(gs[2, 1])  # histo x1
    ax31 = plt.subplot(gs[2, 2])  # histo c

    # lines
    ax00.plot([0, 1.2], np.zeros(len([0, 1.2])), "k:")
    ax10.plot([0, 1.2], np.zeros(len([0, 1.2])), "k:")

    mubins = np.arange(-2, 2 + 0.1, 0.1)

    # Hres w. histogram
    def HRwhisto(
        df, sel_dic, ax_left, ax_right, threshold_sign, ylabel="TP", visible=False
    ):
        if ylabel == "TP":
            sntyp = "Ia"
        else:
            sntyp = "CC"
        ax_left.scatter(
            df["SIM_REDSHIFT_CMB"],
            df["delmu"],
            c=df["class0"],
            cmap=CMAP,
            vmin=0.5,
            vmax=1,
            s=8,
        )
        ax_left.errorbar(
            df["SIM_REDSHIFT_CMB"],
            df["delmu"],
            yerr=df["delmu_err"],
            color="gray",
            zorder=0,
            fmt="none",
            marker="none",
        )

        ax_left.set_ylim(-2, 2)
        ax_left.set_xlim(0, 1.2)
        ax_left.set_ylabel(f"{ylabel} residual", fontsize=18)
        ax_left.tick_params(labelsize=14)
        plt.setp(ax_left.get_xticklabels(), visible=visible)
        if visible is True:
            ax_left.set_xlabel("simulated redshift", fontsize=18)
        for t in threshold_list:
            sel = sel_dic[t]
            n_SNe = len(sel)
            ax_right.hist(
                sel["delmu"],
                orientation="horizontal",
                histtype="step",
                color=cm(t),
                bins=mubins,
                density=True,
                label=f"{n_SNe} {sntyp} {threshold_sign} {t}",
                lw=2,
            )
        ax_right.legend(loc="lower center", prop={"size": 13})
        plt.setp(ax_right.get_yticklabels(), visible=False)
        plt.setp(ax_right.get_xticklabels(), visible=False)
        ax_right.plot(
            [ax_right.get_xlim()[0], ax_right.get_xlim()[1]],
            np.zeros(len([ax_right.get_xlim()[0], ax_right.get_xlim()[1]])),
            "k:",
        )

    HRwhisto(TP, sel_TP_dic, ax00, ax01, threshold_sign, ylabel="TP", visible=False)
    HRwhisto(FP, sel_FP_dic, ax10, ax11, threshold_sign, ylabel="FP", visible=True)

    # z histos
    n, bins_to_use, tmp = ax20.hist(
        Ias["SIM_REDSHIFT_CMB"], histtype="step", color="black", bins=15, lw=3
    )

    for t in threshold_list:
        sel_TP = sel_TP_dic[t]
        sel_FP = sel_FP_dic[t]
        ax20.hist(
            sel_TP["SIM_REDSHIFT_CMB"], histtype="step", color=cm(t), bins=bins_to_use
        )
        ax20.hist(
            sel_FP["SIM_REDSHIFT_CMB"],
            histtype="step",
            color=cm(t),
            linestyle="--",
            bins=bins_to_use,
        )
    ax20.set_xlim(0, 1.2)
    ax20.tick_params(labelsize=14)
    ax20.set_xlabel("simulated redshift", fontsize=18)

    # hist stretch
    n, bins_to_use, tmp = ax21.hist(Ias["x1"], color="black", histtype="step", lw=3)
    for t in threshold_list:
        sel_TP = sel_TP_dic[t]
        ax21.hist(
            sel_TP["x1"],
            orientation="vertical",
            histtype="step",
            color=cm(t),
            bins=bins_to_use,
            lw=2,
        )
    ax21.set_xlabel("x1", fontsize=18)
    ax21.yaxis.set_label_position("right")
    ax21.set_xlim(-3, 3)
    ax21.tick_params(labelsize=14)
    # color histo
    n, bins_to_use, tmp = ax31.hist(Ias["c"], color="black", histtype="step", lw=3)
    for t in threshold_list:
        sel_TP = sel_TP_dic[t]
        ax31.hist(
            sel_TP["c"],
            orientation="vertical",
            histtype="step",
            color=cm(t),
            bins=bins_to_use,
            lw=2,
        )
    ax31.set_xlabel("c", fontsize=18)
    ax31.set_xlim(-1, 1)
    ax31.tick_params(labelsize=14)
    ax31.yaxis.set_label_position("right")

    gs.tight_layout(fig)
    plt.savefig(nameout)
    plt.close()
    del fig


#################
# Formatting
#################


def seaborn_formatting_mag(df, settings):
    """Seaborn friendly formatting
    
    Basic formatting and eliminating outliers (to avoid rejection by seaborn of pd.DataFrame)

    Args:
        df (DataFrame): predictions
        settings (ExperimentSettings): custom class to hold hyperparameters
    Returns:
        df (DataFrame): reformatted
    """
    df["salt"] = df["dataset_saltfit_2classes"] != -1
    df = du.tag_type(df, settings, type_column="SNTYPE")
    # because it doesn't like my normal df
    df_skimmed = pd.DataFrame()
    for f in ["g", "r", "i", "z"]:
        var = "SIM_PEAKMAG_" + f
        df_skimmed[var] = np.array([k for k in df[var].values])
    df_skimmed["salt"] = np.array([k for k in df["salt"].values])
    df_skimmed["target"] = np.array([k for k in df["target_2classes"].values])
    df_skimmed["SIM_REDSHIFT_CMB"] = np.array(
        [k for k in df["SIM_REDSHIFT_CMB"].values]
    )
    df_skimmed["SNTYPE"] = np.array([k for k in df["SNTYPE"].values])
    # skimm
    for f in ["g", "r", "i", "z"]:
        var = "SIM_PEAKMAG_" + f
        df_skimmed = df_skimmed[(df_skimmed[var] > 20) & (df_skimmed[var] < 28)]

    return df_skimmed


#################
# Computations
#################


def make_measurements_df(df, settings, group_bayesian=False):
    """Obtain measurements
    Args:
        df (DataFrame): predictions
        settings (ExperimentSettings): custom class to hold hyperparameters
        group_bayesian (Boolean): if BRNNs ar eused need to group predictions
    Returns:
        df (DataFrame): with measurements and necessary keys to compute accuracy
    """
    list_df = []
    for key in ["-2", "-1", "", "+2"]:
        if key != "all":
            format_key = f"PEAKMJD{key}"
        else:
            format_key = key
        tmp_df = pd.DataFrame()
        tmp_df[
            [f"num_{band}" for band in settings.list_filters]
            + [f"{format_key}_class{i}" for i in [0, 1]]
            + ["target", "SNID"]
        ] = df[
            [f"{format_key}_num_{band}" for band in settings.list_filters]
            + [f"{format_key}_class{i}" for i in [0, 1]]
            + ["target", "SNID"]
        ]
        tmp_df = pu.reformat_df(
            tmp_df,
            key,
            keep=[f"num_{band}" for band in settings.list_filters],
            group_bayesian=group_bayesian,
        )
        list_df.append(tmp_df)
    ndf = pd.concat(list_df)
    return ndf


def distance_modulus(df):
    """Add distance modulus
    Args:
        df (DataFrame): with SALT2 fitted features
    Returns:
        df (DataFrame): with distance modulus computed
    """

    # SNIa parameters
    Mb = 19.365
    alpha = 0.144  # from sim
    beta = 3.1
    # Add distance modulus to this Data Frame
    df["mu"] = (
        np.array(df["mB"]) + Mb + np.array(alpha * df["x1"]) - np.array(beta * df["c"])
    )
    df["delmu"] = df["mu"].values - dist_mu(df["SIM_REDSHIFT_CMB"].values.astype(float))
    # assuming theoretical mu nor alpha, beta, abs mag have errors
    df["delmu_err"] = (
        np.array(df["mBERR"])
        + np.array(alpha * df["x1ERR"])
        - np.array(beta * df["cERR"])
    )
    return df


def sel_eff(merged, threshold, settings):
    """Efficiency curve for different probabilities
    
    Args:
        df (DataFrame): with SALT2 fitted features
    Returns:
        mean_bins (np.array)
        efficience (np.array)
    """
    sel = merged[merged["class0"] > threshold].copy()
    if len(sel) > 0:
        mean_bins, efficiency = pu.get_quantity_vs_variable(
            "efficiency", "SIM_REDSHIFT_CMB", sel, settings, nbins=10, mean_bins=True
        )
    else:
        mean_bins, efficiency = np.arange(0, 1.2, 1.2 / 10.0), np.zeros(10)
    return mean_bins, efficiency


#################
# Plot functions
#################


def datasets_plots(df, settings):
    """Dataset violin plots
    peak magnitudes and redshift distributions of representative and non-representative datasets
    
    Args:
        df (DataFrame): predictions
        settings (ExperimentSettings): custom class to hold hyperparameters
    """

    # Reformat into seaborn friendly format
    df = seaborn_formatting_mag(df, settings)
    multiplot_violin_paper(df, "test", settings)


def performance_plots(settings):
    """Performance: accuracy vs. size training set
        Uses only Saltfit data and Baseline and Random Forest models

    Args:
        settings (ExperimentSettings): custom class to hold hyperparameters
    """

    # Read performance summaries (created by supernnova.validate.performance)
    list_files = Path(f"{settings.models_dir}").glob("**/*METRICS*.pickle")
    list_files = map(str, list_files)

    list_files = [
        f
        for f in list_files
        if ("randomforest" in f or "vanilla" in f)
        and "saltfit" in f
        and "global" in f
        and "_CLF_2" in f
        and "0.25" not in f
    ]
    # select only best hp (issue for df 0.2) and only cyclic vanilla
    tmp_list_files = [
        l for l in list_files if "N_global_lstm_32x2_0.05_128_True_mean_C" in l
    ]
    tmp_list_files += [l for l in list_files if "randomforest" in l]
    list_files = tmp_list_files

    if len(list_files) == 0:
        print(
            lu.str_to_brightstr(
                "supernnova/visualization/superNNova_plots:performance_plots.py"
            ),
            lu.str_to_yellowstr(
                "\nNo predictions found. Please train and validate randomforest and vanilla models"
            ),
        )
        return

    # read all performance metrics
    list_df = []
    for f in list_files:
        df = pd.read_pickle(f)
        model_name = df["model_name"][0]
        model_name_noseed = re.sub(r"S\_\d+_", "", model_name)
        df["model_name_noseed"] = model_name_noseed
        # Use regular expressions to look for hyperparameters
        nb_classes = int(re.search(r"(?<=CLF\_)\d+(?=\_)", model_name).group())
        data_fraction = float(re.search(r"(?<=DF\_)\d\.\d+(?=\_)", model_name).group())
        seed = int(re.search(r"(?<=S\_)\d+(?=\_)", model_name).group())
        redshift = re.search(r"(?<=R\_)[A-Za-z]+(?=\_)", model_name).group()
        model_type = model_name.split("_")[0]
        model_source_data = re.search(r"(?<=\_)[A-Za-z]+(?=\_DF\_)", model_name).group()
        # avoiding models trained and validated with different datasets
        if model_source_data != df["source_data"].values.astype(str)[0]:
            continue

        df["model_type"] = model_type
        df["data_fraction"] = data_fraction
        df["nb_classes"] = nb_classes
        df["seed"] = seed
        df["redshift"] = redshift

        list_df.append(df)

    df_all = pd.concat(list_df, axis=0, sort=True)

    # Plot acc vs. n SN in training
    plot_acc_vs_nsn(df_all, settings)


def purity_vs_z(df, model_name, settings):
    """Purity and contamination redshift evolution
    
    Args:
        df (DataFrame):
        modelname (str): name of model to be used
        settings (ExperimentSettings): custom class to hold hyperparameters
    """
    df = pu.reformat_df(df, "all", keep=None, group_bayesian=True)

    bin_centers, purity_arr = pu.get_quantity_vs_variable(
        "purity", "SIM_REDSHIFT_CMB", df, settings
    )
    y_dic = {}
    y_dic["purity"] = purity_arr
    for typ in [t for t in settings.sntypes if t != 101]:
        bin_centers, y_dic[typ] = pu.get_quantity_vs_variable(
            "contamination", "SIM_REDSHIFT_CMB", df, settings, contamination_by=typ
        )

    # now make the plots for each model
    plot_path = f"{settings.figures_dir}/puritycontamination_vs_z"
    os.makedirs(plot_path, exist_ok=True)
    nameout = f"{plot_path}/{model_name}.png"
    color_sequence = [ALL_COLORS[4]] + [
        ALL_COLORS[i] for i in range(len(y_dic)) if i != 4
    ]
    label_list = ["purity"] + [
        f"{settings.sntypes[k]}" for k in y_dic.keys() if k != 101 and k != "purity"
    ]
    binned_2d(
        bin_centers,
        y_dic,
        "redshift",
        "purity/contamination",
        label_list,
        color_sequence,
        MARKER_LIST,
        nameout,
        extra_line=True,
    )


def cadence_acc_matrix(df, model_name, settings):
    """Matrix with accuracy w.r. number of measurements in a band
    
    Correlation between accuracy and a certain number of observations required per filter
    
    Args:
        df (DataFrame):
        modelname (str): name of model to be used
        settings (ExperimentSettings): custom class to hold hyperparameters
    """
    if "bayesian" or "variational" in model_name:
        measurements_df = make_measurements_df(df, settings, group_bayesian=True)
    else:
        measurements_df = make_measurements_df(df, settings)
    n_measurements_dic = {}
    accuracy_dic = {}
    for band in settings.list_filters:
        n_measurements_dic[band], accuracy_dic[band] = pu.get_quantity_vs_variable(
            "accuracy",
            f"num_{band}",
            measurements_df,
            settings,
            nbins=max(measurements_df[f"num_{band}"]),
            intervals=True,
        )
    plot_path = f"{settings.figures_dir}/accuracy_measurements_matrix"
    os.makedirs(plot_path, exist_ok=True)
    nameout = f"{plot_path}/{model_name}.png"
    plot_acc_matrix(accuracy_dic, n_measurements_dic, settings, nameout)


def hubble_residuals(df, model_name, fits, settings):
    """Hubble residuals for classified supernovae as type Ia
    
    Uses SALT2 fits to compute the distance modulus, therefore list is not complete if fit failed.
    
    Args:
        df (DataFrame):
        modelname (str): name of model to be used
        fits (DataFrame): information from SALT2 fit
        settings (ExperimentSettings): custom class to hold hyperparameters
    """

    # Init plot
    plot_path = f"{settings.figures_dir}/HDresiduals"
    os.makedirs(plot_path, exist_ok=True)

    def reformat_df_HR(df, model_name, keep_list=["mB", "x1", "c"]):
        """format and add distance modulus info
        including error bars
        """
        df = pu.reformat_df(df, "all", keep=keep_list, group_bayesian=True)
        df = pd.merge(df, fits, on="SNID")
        df = distance_modulus(df)
        df = df[~np.isnan(df["mu"])]
        return df

    # Hresiduals with probability thresholds
    merged = reformat_df_HR(df, model_name, keep_list=["mB", "x1", "c"])
    nameout = f"{plot_path}/{model_name}.png"
    plot_HDres_histos_vs_z(
        merged,
        nameout,
        threshold_var="class0",
        threshold_list=[0.5, 0.7, 0.9],
        threshold_sign=">",
    )

    # residuals with uncertainty thresholds
    if "bayesian" in model_name or "variational" in model_name:
        # get one std deviation
        df_grouped = df.groupby("SNID", as_index=False)
        sigma_df = create_sigma_df(df_grouped)
        df_tmp = pd.merge(df, sigma_df, on="SNID")
        merged = reformat_df_HR(
            df_tmp, model_name, keep_list=["mB", "x1", "c", "sigma_all"]
        )
        nameout = f"{plot_path}/{model_name}_cut_uncertainty.png"
        plot_HDres_histos_vs_z(
            merged,
            nameout,
            threshold_var="sigma_all",
            threshold_list=[0.05, 0.1, 0.2],
            threshold_sign="<",
        )


def cnf_matrix(df, model_name, settings):
    """Get confusion matrix from predictions
    
    Args:
        df (DataFrame): predictions for a given model
        modelname (str): name of model to be used
        settings (ExperimentSettings): custom class to hold hyperparameters
    """

    df_r = pu.reformat_df(df, key="all")
    cnf_matrix = confusion_matrix(df_r["target"], df_r["predicted_target"])

    plt.figure()
    settings.nb_classes = len(df_r["target"].unique())
    class_names = [du.sntype_decoded(i, settings) for i in df_r["target"].unique()]
    plot_confusion_matrix(
        settings, cnf_matrix, classes=class_names, normalize=True, nameout=model_name
    )


def plot_speed_benchmark(dump_dir):
    """Plot RNN inference speed benchmarks

    **N.B.** You should have run the speed benchmarks with

    .. code: python

        python run.py --speed

    - Plot speed based on device
    - Plot speed based on choice of model (BBB, Variational or Baseline)


    Args:
        dump_dir (str): Root folder where results are stored
    """

    speed_file = os.path.join(dump_dir, "stats/rnn_speed.csv")

    assert os.path.isfile(speed_file), lu.str_to_redstr(
        f"speed_file does not exist. Run ``python run.py --speed`` first."
    )

    df = pd.read_csv(speed_file)

    df_cpu = df[df.device == "cpu"]
    df_gpu = df[df.device == "gpu"]

    cpu_is_available = len(df_cpu) > 0
    gpu_is_available = len(df_gpu) > 0

    # CPU benchmark should always be available
    assert cpu_is_available

    n_models = len(df.model.unique())

    if gpu_is_available:
        # Space bars by 2 units to leave room for gpu
        idxs_cpu = 0.5 + np.arange(3 * n_models)[::3]
        idxs_gpu = idxs_cpu + 1
        xticks = idxs_cpu + 0.5
        xtick_labels = df_cpu.model.values.tolist()

    else:
        # Space bars by 1 unit
        idxs_cpu = 0.5 + np.arange(2 * n_models)[::2]
        xticks = idxs_cpu
        xtick_labels = df_cpu.model.values.tolist()

    plt.figure()
    ax = plt.gca()

    for i in range(len(idxs_cpu)):
        label = "CPU" if i == 0 else None
        plt.bar(
            idxs_cpu[i],
            df_cpu["Supernova_per_s"].values[i],
            width=1,
            color="C0",
            label=label,
        )

    if gpu_is_available:
        for i in range(len(idxs_gpu)):
            label = "GPU" if i == 0 else None
            plt.bar(
                idxs_gpu[i],
                df_gpu["Supernova_per_s"].values[i],
                width=1,
                color="C2",
                label=label,
            )

    ax.set_ylabel("Lightcurves / s", fontsize=16)
    ax.set_title("Inference throughput", fontsize=20)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_yscale("log")
    ax.legend()

    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(dump_dir, "figures/rnn_speed.png"))
    plt.clf()
    plt.close()


def create_OOD_classification_plots(df, list_models, settings):
    """OOD classification for a model (ppercentages of predictions by class and uncertainties)

    Args:
        df (pandas.DataFrame) : summary statistics df
        list: list with models to use
        settings (ExperimentSettings): custom class to hold hyperparameters
    """

    def OOD_classification_percentages(df, model, targets=2):

        df_sel = df.copy()
        df_sel = df_sel[
            df_sel["model_name_noseed"] == model.replace("CLF_2", f"CLF_{targets}")
        ]
        df_sel = df_sel.fillna(0)

        if len(df_sel) != 0:
            # Get the number of objects classified into a certain class
            # compute percentages
            list_keys = [
                k
                for k in df_sel.keys()
                if "all_" in k and "_num_pred_class" in k and "mean" in k
            ]
            # hack to get total number of OOD lcs (they should be all the same)
            df_perc = (
                df_sel[list_keys]
                .div(
                    df_sel[[l for l in list_keys if "random" in l]].sum(axis=1), axis=0
                )
                .multiply(100)
                .round(2)
            )

            # add percentages of non-classifications
            df_perc["all_percentage_non_pred_mean"] = df_sel[
                "all_percentage_non_pred_mean"
            ]
            for prefix in ["random", "shuffle", "reverse", "sin"]:
                df_perc[f"all_{prefix}_percentage_non_pred_mean"] = df_sel[
                    f"all_{prefix}_percentage_non_pred_mean"
                ]

            # add model name
            df_perc["modelname"] = df_sel["model_name_noseed"].apply(
                lambda x: x.split("_")[1]
            )
            df_perc = df_perc.replace("vanilla", "baseline")
            df_perc = df_perc.fillna(0).round(2)

            # Group by target type: SNe or OOD types
            # save in dictionary for plotting
            percentages = {}
            for to_classify in ["SNe", "random", "shuffle", "reverse", "sin"]:
                if to_classify != "SNe":
                    key_list = [
                        f"all_{to_classify}_num_pred_class{i}_mean"
                        for i in range(targets)
                    ] + [f"all_{to_classify}_percentage_non_pred_mean"]
                else:
                    key_list = [
                        f"all_num_pred_class{i}_mean" for i in range(targets)
                    ] + ["all_percentage_non_pred_mean"]
                percentages[to_classify] = df_perc[key_list].values.flatten().tolist()

        else:
            percentages = {}
            for to_classify in ["SNe", "random", "shuffle", "reverse", "sin"]:
                percentages[to_classify] = np.zeros(targets + 1).tolist()
        return percentages

    def autolabel(ax, rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            if height > 90:
                factor_text = 0.8
            else:
                factor_text = 1.05
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                (factor_text * height),
                f"{height}",
                ha="center",
                va="bottom",
                fontsize=32,
            )

    def barplot_single(ax, labels, percentage, color):
        rects = ax.bar(
            np.arange(len(labels)),
            percentage,
            color=color,
            edgecolor="black",
            tick_label=labels,
        )
        ax.yaxis.set_tick_params(labelsize=40)
        ax.xaxis.set_tick_params(labelsize=40)
        ax.set_ylim(0.01, 99.9)
        autolabel(ax, rects)

    for model in list_models:
        # fig, ax = plt.subplots(5, 3, figsize=(25, 30), sharey=True)
        fig = plt.figure(figsize=(25, 30))
        gs = gridspec.GridSpec(
            5, 3, height_ratios=[1, 1, 1, 1, 1], width_ratios=[1, 1, 2]
        )
        ax = {}
        ax["00"] = plt.subplot(gs[0])
        ax["01"] = plt.subplot(gs[1], sharey=ax["00"])
        ax["02"] = plt.subplot(gs[2], sharey=ax["00"])
        ax["10"] = plt.subplot(gs[3])
        ax["11"] = plt.subplot(gs[4], sharey=ax["10"])
        ax["12"] = plt.subplot(gs[5], sharey=ax["10"])
        ax["20"] = plt.subplot(gs[6])
        ax["21"] = plt.subplot(gs[7], sharey=ax["20"])
        ax["22"] = plt.subplot(gs[8], sharey=ax["20"])
        ax["30"] = plt.subplot(gs[9])
        ax["31"] = plt.subplot(gs[10], sharey=ax["30"])
        ax["32"] = plt.subplot(gs[11], sharey=ax["30"])
        ax["40"] = plt.subplot(gs[12])
        ax["41"] = plt.subplot(gs[13], sharey=ax["40"])
        ax["42"] = plt.subplot(gs[14], sharey=ax["40"])

        # Percentages for a predicted class
        for j, target in enumerate([2, 3, len(settings.sntypes)]):
            percentages = OOD_classification_percentages(df, model, targets=target)
            labels = []
            for t in range(target):
                settings.nb_classes = target
                typ = du.sntype_decoded(t, settings)
                labels.append(typ.strip("SN "))
            # add non-classified
            labels.append("None")
            colors = ALL_COLORS[:target] + ["gray"]
            # Row
            for i, to_classify in enumerate(
                ["SNe", "random", "shuffle", "reverse", "sin"]
            ):
                barplot_single(ax[f"{i}{j}"], labels, percentages[to_classify], colors)
                if i == 0:
                    ax[f"{i}{j}"].set_title(class_target_decode(target), fontsize=50)
                if i < 4:
                    plt.setp(ax[f"{i}{j}"].get_xticklabels(), visible=False)
                ax[f"{i}0"].set_ylabel(to_classify, fontsize=50)
                # dont show axis
                if i < 4:
                    plt.setp(ax[f"{i}{j}"].get_xticklabels(), visible=False)
                if j > 0:
                    plt.setp(ax[f"{i}{j}"].get_yticklabels(), visible=False)
        plt.subplots_adjust(
            left=0.08, right=0.99, bottom=0.03, top=0.95, wspace=0.0, hspace=0.02
        )
        # fig.tight_layout()
        plot_path = f"{settings.figures_dir}/OOD_percentages"
        Path(plot_path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{plot_path}/OOD_{model}_classification_percentages.png")
        plt.close()
        plt.clf()


def science_plots(settings, onlycnf=False):
    """Plots for SuperNNova paper
    
    Saved in settings.figures_dir. Need to provide prediction files and linked settings

    Args:
        settings (ExperimentSettings): custom class to hold hyperparameters

    """

    if len(settings.prediction_files) == 0:
        print(
            lu.str_to_yellowstr("Warning: no prediction files provided. Not plotting")
        )
        return

    # Load data
    df_SNinfo = du.load_HDF5_SNinfo(settings)

    # Get extra info from fits (for distance modulus)
    fits = du.load_fitfile(settings)
    if len(fits) !=0:
        fits = fits[["SNID", "cERR", "mBERR", "x1ERR"]]

        # check if files are there
        tmp_not_found = [m for m in settings.prediction_files if not os.path.exists(m)]
        if len(tmp_not_found) > 0:
            print(lu.str_to_redstr(f"Files not found {tmp_not_found}"))
            tmp_prediction_files = [
                m for m in settings.prediction_files if os.path.exists(m)
            ]
            settings.prediction_files = tmp_prediction_files

        for f in settings.prediction_files:
            df = pd.read_pickle(f)
            model_name = Path(f).stem

            cols_to_merge = ["SNID", "SIM_REDSHIFT_CMB", "SNTYPE", "mB", "x1", "c"]
            cols_to_merge += [c for c in df_SNinfo.columns if "unique_nights" in c]
            cols_to_merge += [c for c in df_SNinfo.columns if "_num_" in c]

            df = df.merge(df_SNinfo.reset_index()[cols_to_merge], how="left", on="SNID")

            if onlycnf:
                cnf_matrix(df, model_name, settings)
            else:
                # Science plots
                purity_vs_z(df, model_name, settings)
                # cadence_acc_matrix(df, model_name, settings)
                hubble_residuals(df, model_name, fits, settings)
                cnf_matrix(df, model_name, settings)
