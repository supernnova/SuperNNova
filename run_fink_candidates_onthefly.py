"""
Run on-the-fly SuperNNova classification for Fink broker candidate objects.

Compares all models found in a directory and produces per-object plots with
one P(Ia) curve per model, plus a multi-model accuracy summary table.

Workflow:
    1. Read diaObjectId list from a CSV of interesting candidates.
    2. Download light-curves for each object from the Fink alert-broker API.
    3. Transform photometry to the SuperNNova FLUXCAL convention (using ZP 27.5).
    4. For each model × object: run early predictions (one classify per MJD day).
    5. Save per-object PNG plots: light-curve (top) + P(Ia) per model (bottom).
    6. Print accuracy table: full-LC and at-peak, per model.

Usage:
    Local run:
    .venv/bin/python run_fink_candidates_onthefly.py \\
        --csv_file interesting_objects_20260218.csv \\
        --models_dir /Users/amoller/Science/RubinLSST/LSST_obsv4_sims/ICML_2025/BNN_SWAG_Rubin/models \\
        --outdir plots_fink_candidates \\
        --device cpu

Notes:
    - Requires network access to https://api.lsst.fink-portal.org/api/v1/sources
    - Each model sub-directory must contain a .pt file + cli_args.json + norm.json.
    - Host-galaxy features (HOSTGAL_*) are filled with zeros (redshift = "none").
"""

import io
import os
import re
import time
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from supernnova.validation.validate_onthefly import classify_lcs, get_settings

matplotlib.use("Agg")  # non-interactive backend for saving figures


# ── Minimal coloured logging (no colorama dependency) ─────────────────────────
_RESET = "\033[0m"


def _log(colour_code: str, *args) -> None:
    print(f"{colour_code}{' '.join(str(a) for a in args)}{_RESET}", flush=True)


def log_info(*args) -> None:   _log("\033[34m", *args)   # blue
def log_ok(*args) -> None:     _log("\033[32m", *args)   # green
def log_warn(*args) -> None:   _log("\033[33m", *args)   # yellow
def log_err(*args) -> None:    _log("\033[31m", *args)   # red


# ── Constants ─────────────────────────────────────────────────────────────────

FINK_API_URL = "https://api.lsst.fink-portal.org/api/v1/sources"

_FLUXCAL_FACTOR = 10 ** (-(31.4 - 27.5) / 2.5)

BAND_TO_FLT = {
    "u": "LSST-u", "g": "LSST-g", "r": "LSST-r",
    "i": "LSST-i", "z": "LSST-z", "y": "LSST-Y",
}

FILTER_COLORS = {
    "u": "purple", "g": "indigo", "r": "royalblue",
    "i": "darkorange", "z": "maroon", "Y": "red",
}

# Color cycle for models — enough for ~10 models
MODEL_COLOR_CYCLE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#66c2a5", "#fc8d62",
]
MODEL_LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"]


# ── Model discovery ───────────────────────────────────────────────────────────


def _short_name(pt_path: Path) -> str:
    """Extract a compact model label from its filename.

    ``vanilla_S_0_CLF_2_R_none_photometry_..._mean.pt`` → ``vanilla_S0``
    ``vanilla_S_0_CLF_2_R_none_photometry_..._swag.pt`` → ``vanilla_S0_swag``
    """
    stem = pt_path.stem
    is_swag = stem.endswith("_swag")
    m = re.match(r"(vanilla|variational|bayesian|swag)_S_(\d+)_", stem,
                 re.IGNORECASE)
    if m:
        name = f"{m.group(1)}_S{m.group(2)}"
        if is_swag:
            name += "_swag"
        return name
    # Fallback: first 25 chars of stem
    return stem[:25]


def discover_models(models_dir: str) -> dict[str, str]:
    """Find all SuperNNova .pt model files inside models_dir sub-directories.

    Expected layout::

        models_dir/
            <model_name>/
                <model_name>.pt
                cli_args.json
                norm.json

    Parameters
    ----------
    models_dir : str
        Root directory containing one sub-directory per model.

    Returns
    -------
    dict[str, str]
        Mapping short_name → absolute path to .pt file, sorted by short name.
    """
    root = Path(models_dir)
    # os.walk with followlinks=True is used instead of Path.glob so that
    # symbolic links to directories are traversed correctly.
    pts = sorted(
        Path(dirpath) / fname
        for dirpath, _, filenames in os.walk(root, followlinks=True)
        for fname in filenames
        if fname.endswith(".pt")
    )
    if not pts:
        raise FileNotFoundError(f"No .pt files found under {models_dir}")

    models: dict[str, str] = {}
    for pt in pts:
        name = _short_name(pt)
        # Resolve duplicates by appending an index
        if name in models:
            name = f"{name}_{sum(1 for k in models if k.startswith(name))}"
        models[name] = str(pt)

    log_ok(f"Discovered {len(models)} model(s) in {models_dir}:")
    for name, path in models.items():
        log_info(f"    {name:25s}  {path}")
    return models


# ── Steps 1-3: unchanged ──────────────────────────────────────────────────────


def load_candidate_ids(csv_file: str) -> tuple[list[str], dict[str, str] | None]:
    """Load diaObjectId values from the interesting-objects CSV.

    If the CSV contains a ``Type`` column it is returned as a mapping
    ``{diaObjectId: type_string}`` so that true labels can be used for
    accuracy / completeness / efficiency metrics.  Otherwise ``None`` is
    returned for the type map.

    Returns
    -------
    ids : list[str]
    type_map : dict[str, str] | None
        Maps each diaObjectId (string) to its spectroscopic type label, or
        ``None`` when the column is absent.
    """
    df = pd.read_csv(csv_file, skipinitialspace=True)
    df["diaObjectId"] = df["diaObjectId"].astype(str).str.strip()
    ids = df["diaObjectId"].unique().tolist()
    log_ok(f"Loaded {len(ids)} candidate IDs from {csv_file}")

    # Accept either "Type" (generic) or "Obj. Type" (TNS export format)
    type_col = next((c for c in ("Type", "Obj. Type") if c in df.columns), None)

    type_map: dict[str, str] | None = None
    if type_col is not None:
        deduped = df.drop_duplicates("diaObjectId").set_index("diaObjectId")
        type_map = (
            deduped[type_col]
            .fillna("")
            .astype(str)
            .to_dict()
        )
        n_typed = sum(1 for v in type_map.values() if v)
        log_ok(f"  '{type_col}' column found — true labels available for {n_typed}/{len(ids)} objects.")
        log_ok("  SN Ia (any subtype containing 'Ia') treated as the positive class.")

    return ids, type_map


def fetch_lc_from_fink(dia_object_id: str, timeout: int = 30) -> pd.DataFrame | None:
    """Fetch a single light-curve from the Fink alert broker API."""
    try:
        r = requests.post(
            FINK_API_URL,
            json={"diaObjectId": dia_object_id, "output-format": "json"},
            timeout=timeout,
        )
        r.raise_for_status()
        df = pd.read_json(io.BytesIO(r.content))
        if df.empty:
            log_warn(f"  No data returned for {dia_object_id}")
            return None
        return df
    except Exception as exc:
        log_err(f"  Failed to fetch {dia_object_id}: {exc}")
        return None


def download_all_lcs(ids: list[str]) -> pd.DataFrame | None:
    """Download light-curves for all candidate IDs."""
    frames = []
    for idx, obj_id in enumerate(ids):
        log_info(f"  [{idx + 1}/{len(ids)}] Fetching {obj_id} …")
        df = fetch_lc_from_fink(obj_id)
        if df is not None:
            frames.append(df)
    if not frames:
        log_err("No light-curves downloaded – aborting.")
        return None
    return pd.concat(frames, ignore_index=True)


def transform_to_snn_format(df_fink: pd.DataFrame) -> pd.DataFrame:
    """Reformat raw Fink photometry to SuperNNova FLUXCAL convention."""
    df = df_fink.copy()
    df["SNID"] = df["r:diaObjectId"].astype(str)
    df["MJD"] = df["r:midpointMjdTai"]
    df["FLUXCAL"] = df["r:psfFlux"] * _FLUXCAL_FACTOR
    df["FLUXCALERR"] = df["r:psfFluxErr"] * _FLUXCAL_FACTOR
    df["FLT"] = df["r:band"].map(BAND_TO_FLT)
    for col in ["HOSTGAL_SPECZ", "HOSTGAL_SPECZ_ERR",
                "HOSTGAL_PHOTOZ", "HOSTGAL_PHOTOZ_ERR", "MWEBV"]:
        df[col] = 0.0
    keep = ["SNID", "MJD", "FLUXCAL", "FLUXCALERR", "FLT",
            "HOSTGAL_SPECZ", "HOSTGAL_SPECZ_ERR",
            "HOSTGAL_PHOTOZ", "HOSTGAL_PHOTOZ_ERR", "MWEBV"]
    n_raw = len(df)
    df = df[keep].dropna(subset=["MJD", "FLUXCAL", "FLUXCALERR", "FLT"]).sort_values("MJD").reset_index(drop=True)
    n_dropped = n_raw - len(df)
    if n_dropped:
        log_warn(f"  Dropped {n_dropped} rows with NaN flux/MJD/FLT values from Fink.")
    float_cols = ["MJD", "FLUXCAL", "FLUXCALERR",
                  "HOSTGAL_SPECZ", "HOSTGAL_SPECZ_ERR",
                  "HOSTGAL_PHOTOZ", "HOSTGAL_PHOTOZ_ERR", "MWEBV"]
    df[float_cols] = df[float_cols].astype(np.float64)
    n_obj = df["SNID"].nunique()
    log_ok(f"Transformed {len(df)} observations for {n_obj} objects.")
    return df


def average_same_night_obs(df: pd.DataFrame) -> pd.DataFrame:
    """Average multiple observations of the same object, band, and night.

    "Same night" is defined by the integer part of MJD (floor), so all
    exposures within a single UTC day are grouped together.

    Error propagation for the mean of N measurements:
        FLUXCAL_avg    = mean(FLUXCAL_i)
        FLUXCALERR_avg = sqrt( sum(FLUXCALERR_i²) ) / N

    All other columns (HOSTGAL_*, MWEBV) are constant (zero) so their
    first value is kept unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        SuperNNova-format dataframe (output of transform_to_snn_format).

    Returns
    -------
    pd.DataFrame
        Dataframe with intra-night, same-band duplicates averaged out.
    """
    df = df.copy()
    df["_night"] = np.floor(df["MJD"]).astype(int)

    group_cols = ["SNID", "_night", "FLT"]
    n_before = len(df)

    # Count observations per (SNID, night, FLT) group using transform so we
    # can identify duplicates without a separate groupby call.
    grp_sizes = df.groupby(group_cols)["FLUXCAL"].transform("count")
    n_multi_groups = int((grp_sizes > 1).sum() > 0)   # 1 if any duplicates exist

    if n_multi_groups == 0:
        log_info("  avg_same_night: no intra-night duplicates found — nothing to average.")
        return df.drop(columns=["_night"])

    # How many groups have >1 obs (for the log message)?
    n_dup_groups = int((df.groupby(group_cols)["FLUXCAL"].count() > 1).sum())

    const_cols = ["HOSTGAL_SPECZ", "HOSTGAL_SPECZ_ERR",
                  "HOSTGAL_PHOTOZ", "HOSTGAL_PHOTOZ_ERR", "MWEBV"]

    grp = df.groupby(group_cols, sort=False)

    # Use explicit separate operations to avoid the dict-with-callable form of
    # groupby().agg(), which can silently misbehave in some pandas 2.x builds.
    df_mjd_flux = grp[["MJD", "FLUXCAL"]].mean()

    # Propagated error for the mean: sqrt(sum(σ²)) / N
    df_err = grp["FLUXCALERR"].apply(
        lambda x: float(np.sqrt((x.to_numpy() ** 2).sum()) / len(x))
    ).rename("FLUXCALERR")

    df_const = grp[const_cols].first()

    df_avg = (
        pd.concat([df_mjd_flux, df_err, df_const], axis=1)
        .reset_index()
        .drop(columns=["_night"])
        .sort_values(["SNID", "MJD"])
        .reset_index(drop=True)
    )

    # Sanity-check: verify row count actually decreased
    n_after = len(df_avg)
    if n_after >= n_before:
        log_warn(
            f"  avg_same_night: row count did not decrease ({n_before} → {n_after}). "
            "Check that MJD values are correctly parsed as floats."
        )

    log_ok(
        f"  avg_same_night: {n_before} → {n_after} observations "
        f"({n_dup_groups} intra-night duplicate groups merged)."
    )

    # Per-SNID diagnostic to confirm which objects were affected
    for snid, g in df.groupby("SNID"):
        g_avg = df_avg[df_avg["SNID"] == snid]
        if len(g) != len(g_avg):
            log_info(
                f"    SNID {snid}: {len(g)} obs → {len(g_avg)} obs after averaging"
            )

    return df_avg


# ── Step 4: early predictions (per model) ────────────────────────────────────


def _remap_flt_for_model(df: pd.DataFrame, model_file: str) -> pd.DataFrame:
    """Remap FLT values to the filter names expected by a specific model.

    Some model families (e.g. ELAsTiCC) store filters as bare letters
    ("u","g","r","i","z","y") while our pipeline produces "LSST-u",
    "LSST-g", etc.  If the model's ``list_filters`` differs from the
    FLT values already in the dataframe, this function builds a remap by
    matching on the suffix after the last dash (case-insensitive).

    Returns a copy of *df* with FLT remapped, or the original if no
    remap is needed.
    """
    settings = get_settings(model_file)
    model_filters = set(settings.list_filters)
    data_filters = set(df["FLT"].unique())

    if data_filters <= model_filters:
        return df  # names already match

    remap: dict[str, str] = {}
    for dflt in data_filters:
        if dflt in model_filters:
            remap[dflt] = dflt
            continue
        # Try matching on the character(s) after the last "-"
        suffix = dflt.rsplit("-", 1)[-1].lower()
        for mflt in model_filters:
            if mflt.lower() == suffix:
                remap[dflt] = mflt
                break

    if set(remap.values()) != model_filters & set(remap.values()):
        pass  # partial remap is still better than none

    unmapped = data_filters - set(remap)
    if unmapped:
        log_warn(
            f"  FLT remap: could not map {unmapped} to any filter in "
            f"{model_filters} — those observations will be dropped."
        )

    df = df.copy()
    df["FLT"] = df["FLT"].map(remap)
    df = df.dropna(subset=["FLT"])  # drop rows whose filter had no match

    if remap and set(remap.keys()) != set(remap.values()):
        log_info(
            f"  FLT remap applied: "
            + ", ".join(f"{k}→{v}" for k, v in sorted(remap.items()))
        )

    return df


def _run_early_predictions_one_model(
    df_obj: pd.DataFrame, model_file: str, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Classify one object × one model at each successive MJD day.

    Returns
    -------
    mjd_days : np.ndarray, shape (T,)
    pia      : np.ndarray, shape (T,)   — P(Ia) at each timestep
    """
    df_obj = _remap_flt_for_model(df_obj, model_file)
    df_obj = df_obj.copy()
    df_obj["MJD_int"] = df_obj["MJD"].astype(int)
    mjd_days = sorted(df_obj["MJD_int"].unique())

    mjd_list: list[int] = []
    pia_list: list[float] = []

    first_error_shown = False
    for day in mjd_days:
        df_day = df_obj[df_obj["MJD_int"] <= day].copy()
        try:
            _, preds = classify_lcs(df_day, model_file, device)
            # preds: (1, num_inference_samples, nb_classes) → mean over samples
            mean_prob = preds[0].mean(axis=0)
            mjd_list.append(day)
            pia_list.append(float(mean_prob[0]))   # class 0 = Ia
        except Exception as exc:
            if not first_error_shown:
                log_err(f"      classify_lcs failed at MJD {day}:")
                traceback.print_exc()
                first_error_shown = True
            else:
                log_warn(f"      Skipping MJD {day}: {exc}")

    if not pia_list:
        return np.array([]), np.array([])
    return np.array(mjd_list), np.array(pia_list)


def run_early_predictions_all_models(
    df_obj: pd.DataFrame, models: dict[str, str], device: str
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Run early predictions for one object across all models.

    Parameters
    ----------
    df_obj : pd.DataFrame
        SuperNNova-format dataframe for a single SNID.
    models : dict[str, str]
        short_name → .pt path.
    device : str

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]]
        short_name → (mjd_days, pia_over_time)
    """
    results = {}
    for name, path in models.items():
        log_info(f"    Early preds — {name} …")
        mjd_days, pia = _run_early_predictions_one_model(df_obj, path, device)
        results[name] = (mjd_days, pia)
    return results


# ── Step 4b: metrics at full LC and at peak ───────────────────────────────────


def get_metrics_all_models(
    df_obj: pd.DataFrame, models: dict[str, str], device: str
) -> list[dict]:
    """Classify at full-LC and at-peak for every model.

    Parameters
    ----------
    df_obj : pd.DataFrame
    models : dict[str, str]
    device : str

    Returns
    -------
    list[dict]
        One dict per model with keys: model, prob_ia_full, pred_class_full,
        prob_ia_peak, pred_class_peak, peak_mjd.
    """
    peak_mjd = float(df_obj.loc[df_obj["FLUXCAL"].idxmax(), "MJD"])

    rows = []
    for name, path in models.items():
        row: dict = dict(
            model=name,
            prob_ia_full=np.nan, pred_class_full=-1,
            prob_ia_peak=np.nan, pred_class_peak=-1,
            peak_mjd=peak_mjd,
        )
        # Remap FLT once per model, then slice at-peak from the remapped df
        df_model = _remap_flt_for_model(df_obj, path)
        df_peak = df_model[df_model["MJD"] <= peak_mjd].copy()

        # Full LC
        try:
            _, preds = classify_lcs(df_model, path, device)
            samples = preds[0]                      # (n_samples, n_classes)
            mean_prob = samples.mean(axis=0)
            row["prob_ia_full"] = float(mean_prob[0])
            row["std_ia_full"]  = float(samples[:, 0].std())
            row["pred_class_full"] = int(np.argmax(mean_prob))
        except Exception as exc:
            log_warn(f"    Metrics full-LC failed [{name}]: {exc}")

        # At peak
        try:
            _, preds = classify_lcs(df_peak, path, device)
            samples = preds[0]
            mean_prob = samples.mean(axis=0)
            row["prob_ia_peak"] = float(mean_prob[0])
            row["std_ia_peak"]  = float(samples[:, 0].std())
            row["pred_class_peak"] = int(np.argmax(mean_prob))
        except Exception as exc:
            log_warn(f"    Metrics at-peak failed [{name}]: {exc}")

        rows.append(row)
    return rows


# ── Step 5: plotting ──────────────────────────────────────────────────────────


def plot_lc_and_predictions(
    df_obj: pd.DataFrame,
    snid: str,
    model_preds: dict[str, tuple[np.ndarray, np.ndarray]],
    model_metrics: list[dict],
    outdir: str,
    true_type: str | None = None,
) -> None:
    """Two-panel figure: light-curve (top) + P(Ia) per model (bottom).

    Parameters
    ----------
    df_obj : pd.DataFrame
    snid : str
    model_preds : dict[str, tuple[np.ndarray, np.ndarray]]
        short_name → (mjd_days, pia_over_time)
    model_metrics : list[dict]
        Output of get_metrics_all_models(); used for peak MJD and title.
    outdir : str
    true_type : str | None
        Spectroscopic type label (e.g. ``"SN Ia"``) to display in the plot
        title.  Omitted when ``None`` or empty.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Peak MJD is the same for all models (it's a data property)
    peak_mjd = model_metrics[0]["peak_mjd"] if model_metrics else np.nan

    fig = plt.figure(figsize=(11, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.38)

    # ── Top panel: light-curve ────────────────────────────────────────────────
    ax_lc = fig.add_subplot(gs[0])
    for flt in sorted(df_obj["FLT"].unique()):
        bare = flt.replace("LSST-", "")
        color = FILTER_COLORS.get(bare, "black")
        sel = df_obj[df_obj["FLT"] == flt]
        ax_lc.errorbar(sel["MJD"], sel["FLUXCAL"], yerr=sel["FLUXCALERR"],
                       fmt="o", label=flt, color=color,
                       elinewidth=1, capsize=2, markersize=4)

    if not np.isnan(peak_mjd):
        ax_lc.axvline(peak_mjd, color="black", linestyle="--",
                      linewidth=1, alpha=0.6, label="Peak flux")

    ax_lc.set_ylabel("FLUXCAL")
    title = f"Object {snid}"
    if true_type:
        title += f"  [{true_type}]"
    ax_lc.set_title(title, fontsize=10)
    ax_lc.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax_lc.grid(alpha=0.3)

    # ── Bottom panel: P(Ia) over time, one line per model ────────────────────
    ax_pred = fig.add_subplot(gs[1])
    any_plotted = False
    first_mjd = None

    for i, (name, (mjd_days, pia)) in enumerate(model_preds.items()):
        if len(mjd_days) == 0:
            continue
        color = MODEL_COLOR_CYCLE[i % len(MODEL_COLOR_CYCLE)]
        ls = MODEL_LINESTYLES[i % len(MODEL_LINESTYLES)]
        ax_pred.plot(mjd_days, pia, color=color, linestyle=ls,
                     linewidth=1.5, alpha=0.85, label=name)
        if first_mjd is None:
            first_mjd = mjd_days[0]
        any_plotted = True

    if any_plotted and first_mjd is not None:
        ax_pred.axvline(first_mjd, color="grey", linestyle=":",
                        linewidth=1, alpha=0.6)
        if not np.isnan(peak_mjd):
            ax_pred.axvline(peak_mjd, color="black", linestyle="--",
                            linewidth=1, alpha=0.6, label="Peak flux")
    else:
        ax_pred.text(0.5, 0.5, "Insufficient data for early predictions",
                     ha="center", va="center", transform=ax_pred.transAxes)

    ax_pred.axhline(0.5, color="grey", linestyle=":", linewidth=1, alpha=0.6)
    ax_pred.set_ylim(-0.05, 1.05)
    ax_pred.set_xlabel("MJD")
    ax_pred.set_ylabel("P(Ia)")
    ax_pred.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    ax_pred.grid(alpha=0.3)

    fig_name = Path(outdir) / f"early_pred_{snid}.png"
    plt.tight_layout()
    plt.savefig(fig_name, dpi=150, bbox_inches="tight")
    plt.clf()
    plt.close(fig)
    log_ok(f"  Saved: {fig_name}")


# ── Extra evaluation plots ────────────────────────────────────────────────────


def plot_pr_curves(all_metrics: list[dict], models: dict[str, str], outdir: str) -> None:
    """Precision-Recall curves + Average Precision per model.

    Requires true labels (``true_label`` key).  Produces one figure with two
    panels: full light-curve and at-peak.

    Parameters
    ----------
    all_metrics : list[dict]
        Flat metrics list — must contain ``true_label``, ``prob_ia_full``,
        ``prob_ia_peak``.
    models : dict[str, str]
    outdir : str
    """
    try:
        from sklearn.metrics import precision_recall_curve, average_precision_score
    except ImportError:
        log_warn("PR curves: scikit-learn not found — skipping.")
        return

    df = pd.DataFrame(all_metrics)
    labeled = df[df["true_label"].notna()].copy()
    if labeled.empty:
        log_warn("PR curves: no true labels — skipping.")
        return

    model_names = list(models.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, prob_col, title in zip(
        axes,
        ["prob_ia_full", "prob_ia_peak"],
        ["Full light-curve", "At peak"],
    ):
        for i, m in enumerate(model_names):
            mdf = labeled[labeled["model"] == m].dropna(subset=[prob_col])
            if mdf.empty or mdf["true_label"].nunique() < 2:
                continue
            y_true  = mdf["true_label"].astype(int).values
            y_score = mdf[prob_col].values
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            color = MODEL_COLOR_CYCLE[i % len(MODEL_COLOR_CYCLE)]
            ls    = MODEL_LINESTYLES[i % len(MODEL_LINESTYLES)]
            ax.step(rec, prec, where="post", color=color, linestyle=ls,
                    linewidth=1.5, label=f"{m}  AP={ap:.2f}")

        # Random-classifier baseline = fraction of positives
        frac_ia = labeled["true_label"].mean()
        ax.axhline(frac_ia, color="grey", linestyle=":", linewidth=1,
                   label=f"Random ({frac_ia:.2f})")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Completeness (Recall)")
        ax.set_ylabel("Efficiency (Precision)")
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Precision-Recall curves  (SN Ia = positive class)", fontsize=11)
    plt.tight_layout()
    fig_name = Path(outdir) / "pr_curves.png"
    plt.savefig(fig_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log_ok(f"  Saved: {fig_name}")


def plot_efficiency_vs_phase(
    all_early_series: list[dict],
    models: dict[str, str],
    outdir: str,
) -> None:
    """Completeness and efficiency as a function of days since first detection.

    For each day *d* after first detection, each object contributes its most
    recent P(Ia) prediction (using all observations up to and including day
    *d*).  Completeness and efficiency are then computed across all labeled
    objects that have at least one observation by day *d*.

    Parameters
    ----------
    all_early_series : list[dict]
        One entry per (SNID, model) with keys: ``model``, ``true_label``
        (int|None), ``mjd_days`` (np.ndarray), ``pia`` (np.ndarray),
        ``first_mjd`` (float).
    models : dict[str, str]
    outdir : str
    """
    labeled = [s for s in all_early_series if s.get("true_label") is not None]
    if not labeled:
        log_warn("Efficiency-vs-phase: no true labels — skipping.")
        return

    model_names = list(models.keys())

    # Phase grid: 0 … max relative day across all objects and models
    max_phase = max(
        float(s["mjd_days"][-1] - s["first_mjd"])
        for s in labeled if len(s["mjd_days"]) > 0
    )
    phase_grid = np.arange(0, int(max_phase) + 1, 1)

    fig, (ax_c, ax_e) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                      gridspec_kw={"hspace": 0.1})

    for i, m in enumerate(model_names):
        series_m = [s for s in labeled if s["model"] == m and len(s["mjd_days"]) > 0]
        if not series_m:
            continue

        compl_list, effic_list, n_obs_list = [], [], []
        for d in phase_grid:
            preds_d, labels_d = [], []
            for s in series_m:
                rel = s["mjd_days"] - s["first_mjd"]
                mask = rel <= d
                if not mask.any():
                    continue
                last_pia = float(s["pia"][mask][-1])
                preds_d.append(1 if last_pia >= 0.5 else 0)
                labels_d.append(int(s["true_label"]))

            n_obs_list.append(len(preds_d))
            if len(preds_d) < 1:
                compl_list.append(np.nan)
                effic_list.append(np.nan)
                continue

            pred = np.array(preds_d)
            true = np.array(labels_d)
            tp = int(((pred == 1) & (true == 1)).sum())
            fp = int(((pred == 1) & (true == 0)).sum())
            fn = int(((pred == 0) & (true == 1)).sum())
            compl_list.append(tp / (tp + fn) if (tp + fn) > 0 else np.nan)
            effic_list.append(tp / (tp + fp) if (tp + fp) > 0 else np.nan)

        color = MODEL_COLOR_CYCLE[i % len(MODEL_COLOR_CYCLE)]
        ls    = MODEL_LINESTYLES[i % len(MODEL_LINESTYLES)]
        ax_c.plot(phase_grid, compl_list, color=color, linestyle=ls,
                  linewidth=1.5, label=m)
        ax_e.plot(phase_grid, effic_list, color=color, linestyle=ls,
                  linewidth=1.5, label=m)

    for ax, ylabel in [(ax_c, "Completeness (Recall)"),
                       (ax_e, "Efficiency (Precision)")]:
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
        ax.grid(alpha=0.3)

    ax_e.set_xlabel("Days since first detection")
    ax_c.set_title("Completeness and Efficiency vs. Phase  (threshold P(Ia) ≥ 0.5)",
                   fontsize=10)
    plt.tight_layout()
    fig_name = Path(outdir) / "efficiency_vs_phase.png"
    plt.savefig(fig_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log_ok(f"  Saved: {fig_name}")


def plot_uncertainty_vs_correctness(
    all_metrics: list[dict],
    models: dict[str, str],
    outdir: str,
) -> None:
    """Scatter of P(Ia) std (uncertainty) vs. P(Ia) mean, coloured by outcome.

    For each model two panels are shown (full-LC and at-peak).  Each point is
    one object.  When true labels are available points are coloured
    green (correct) / red (incorrect) / grey (unlabelled).  Without true
    labels all points are grey.

    Parameters
    ----------
    all_metrics : list[dict]
        Must contain ``std_ia_full``, ``std_ia_peak``, ``prob_ia_full``,
        ``prob_ia_peak``, optionally ``true_label`` and ``pred_class_full``.
    models : dict[str, str]
    outdir : str
    """
    df = pd.DataFrame(all_metrics)
    if "std_ia_full" not in df.columns:
        log_warn("Uncertainty plot: std fields missing — skipping.")
        return

    model_names = list(models.keys())
    has_true = "true_label" in df.columns and df["true_label"].notna().any()
    n_models  = len(model_names)

    fig, axes = plt.subplots(
        n_models, 2,
        figsize=(10, 3.5 * max(n_models, 1)),
        squeeze=False,
    )

    for i, m in enumerate(model_names):
        mdf = df[df["model"] == m].copy()

        for j, (prob_col, std_col, cls_col, title_sfx) in enumerate([
            ("prob_ia_full", "std_ia_full", "pred_class_full", "Full LC"),
            ("prob_ia_peak", "std_ia_peak", "pred_class_peak", "At peak"),
        ]):
            ax = axes[i][j]
            valid = mdf.dropna(subset=[prob_col, std_col])

            if has_true:
                labeled   = valid[valid["true_label"].notna()]
                unlabeled = valid[valid["true_label"].isna()]

                # Correct = true_label matches prediction (class 0 ↔ Ia)
                correct   = labeled[labeled["true_label"].astype(int) == (labeled[cls_col] == 0).astype(int)]
                incorrect = labeled[labeled["true_label"].astype(int) != (labeled[cls_col] == 0).astype(int)]

                ax.scatter(correct[prob_col],   correct[std_col],
                           c="green",  marker="o", s=50, alpha=0.8,
                           label="Correct", zorder=3)
                ax.scatter(incorrect[prob_col], incorrect[std_col],
                           c="red",    marker="x", s=60, alpha=0.8,
                           label="Incorrect", zorder=3)
                if not unlabeled.empty:
                    ax.scatter(unlabeled[prob_col], unlabeled[std_col],
                               c="grey", marker="s", s=40, alpha=0.5,
                               label="No label", zorder=2)
            else:
                ax.scatter(valid[prob_col], valid[std_col],
                           c="steelblue", marker="o", s=50, alpha=0.8)

            ax.axvline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_xlabel("Mean P(Ia)")
            ax.set_ylabel("Std P(Ia)  [uncertainty]")
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(bottom=-0.01)
            ax.set_title(f"{m} — {title_sfx}", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

    fig.suptitle("Uncertainty vs. P(Ia)  (std across inference samples)", fontsize=11)
    plt.tight_layout()
    fig_name = Path(outdir) / "uncertainty_vs_correctness.png"
    plt.savefig(fig_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log_ok(f"  Saved: {fig_name}")


# ── Step 6: multi-model accuracy summary ─────────────────────────────────────


def _is_ia_type(type_str: str | None) -> bool | None:
    """Return True if *type_str* represents an SN Ia (any subtype), else False.

    Returns ``None`` when the label is missing / empty.
    """
    if not type_str:
        return None
    return "ia" in type_str.lower()


def print_metrics_summary(all_metrics: list[dict]) -> None:
    """Print per-model accuracy table across all objects.

    When true labels are present (``true_label`` key in the dicts), the
    aggregate section reports accuracy, completeness (recall), and
    efficiency (precision) instead of just the fraction classified as Ia.

    Parameters
    ----------
    all_metrics : list[dict]
        Flat list of dicts, each with keys: SNID, model, prob_ia_full,
        pred_class_full, prob_ia_peak, pred_class_peak, and optionally
        true_type (str) and true_label (int, 1=Ia / 0=non-Ia).
    """
    df = pd.DataFrame(all_metrics)
    model_names = df["model"].unique().tolist()
    has_true = "true_label" in df.columns and df["true_label"].notna().any()

    log_ok("\n── Per-object metrics ───────────────────────────────────────────")

    # Header
    col_w = 12
    header = f"{'SNID':>22}"
    if has_true:
        header += f"  {'True type':>12}"
    for m in model_names:
        label = m[:col_w]
        header += f"  {label+'|full':>{col_w}}  {label+'|peak':>{col_w}}"
    log_info(header)
    log_info("─" * len(header))

    for snid, grp in df.groupby("SNID"):
        row_str = f"{snid:>22}"
        if has_true:
            tt = grp["true_type"].iloc[0] if "true_type" in grp.columns else ""
            row_str += f"  {str(tt)[:12]:>12}"
        for m in model_names:
            mrow = grp[grp["model"] == m]
            if mrow.empty:
                row_str += f"  {'N/A':>{col_w}}  {'N/A':>{col_w}}"
                continue
            mrow = mrow.iloc[0]
            sym_full = "+" if mrow["pred_class_full"] == 0 else "-"
            sym_peak = "+" if mrow["pred_class_peak"] == 0 else "-"
            p_full = mrow["prob_ia_full"]
            p_peak = mrow["prob_ia_peak"]
            row_str += (
                f"  {f'{p_full:.2f} {sym_full}':>{col_w}}"
                f"  {f'{p_peak:.2f} {sym_peak}':>{col_w}}"
            )
        log_info(row_str)

    # Per-model aggregate
    if has_true:
        log_ok("\n── Aggregate per model (true labels) ────────────────────────────")
        agg_header = (
            f"  {'Model':25s}"
            f"  {'Full acc':>8}  {'Full compl':>10}  {'Full effic':>10}  {'Full P(Ia)':>10}"
            f"  {'Peak acc':>8}  {'Peak compl':>10}  {'Peak effic':>10}  {'Peak P(Ia)':>10}"
        )
        log_info(agg_header)
        log_info("─" * len(agg_header))
        for m in model_names:
            mdf = df[df["model"] == m].copy()
            mdf = mdf[mdf["true_label"].notna()]
            if mdf.empty:
                log_warn(f"  {m:25s}  (no labeled objects)")
                continue
            tl = mdf["true_label"].astype(int)

            def _metrics(pred_col: str, prob_col: str):
                valid = mdf[pred_col] != -1
                if not valid.any():
                    return "N/A", "N/A", "N/A", "N/A"
                pred = (mdf.loc[valid, pred_col] == 0).astype(int)
                true = tl[valid]
                tp = int(((pred == 1) & (true == 1)).sum())
                fp = int(((pred == 1) & (true == 0)).sum())
                fn = int(((pred == 0) & (true == 1)).sum())
                tn = int(((pred == 0) & (true == 0)).sum())
                acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else float("nan")
                compl = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
                effic = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
                mp = mdf.loc[valid, prob_col].mean()
                return (
                    f"{acc:.0%}" if not np.isnan(acc) else "N/A",
                    f"{compl:.0%}" if not np.isnan(compl) else "N/A",
                    f"{effic:.0%}" if not np.isnan(effic) else "N/A",
                    f"{mp:.3f}" if not np.isnan(mp) else "N/A",
                )

            acc_f, c_f, e_f, mp_f = _metrics("pred_class_full", "prob_ia_full")
            acc_p, c_p, e_p, mp_p = _metrics("pred_class_peak", "prob_ia_peak")
            log_ok(
                f"  {m:25s}"
                f"  {acc_f:>8}  {c_f:>10}  {e_f:>10}  {mp_f:>10}"
                f"  {acc_p:>8}  {c_p:>10}  {e_p:>10}  {mp_p:>10}"
            )
    else:
        log_ok("\n── Aggregate accuracy per model (frac classified as Ia) ─────────")
        agg_header = f"  {'Model':25s}  {'Full-LC acc':>11}  {'Full-LC mean P(Ia)':>18}  {'Peak acc':>8}  {'Peak mean P(Ia)':>15}"
        log_info(agg_header)
        log_info("─" * len(agg_header))
        for m in model_names:
            mdf = df[df["model"] == m]
            valid_full = mdf["pred_class_full"] != -1
            valid_peak = mdf["pred_class_peak"] != -1
            acc_full = (mdf.loc[valid_full, "pred_class_full"] == 0).mean()
            acc_peak = (mdf.loc[valid_peak, "pred_class_peak"] == 0).mean()
            mp_full = mdf.loc[valid_full, "prob_ia_full"].mean()
            mp_peak = mdf.loc[valid_peak, "prob_ia_peak"].mean()
            log_ok(
                f"  {m:25s}  {acc_full:>11.0%}  {mp_full:>18.3f}"
                f"  {acc_peak:>8.0%}  {mp_peak:>15.3f}"
            )


# ── README ────────────────────────────────────────────────────────────────────


def write_model_readme(models: dict[str, str], outdir: str) -> None:
    """Write a README.md to *outdir* mapping each legend entry to its model path.

    The index shown here matches the colour/line ordering in the plots.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    readme = Path(outdir) / "README.md"

    import datetime
    lines = [
        "# Model legend",
        "",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Each plot shows one P(Ia) curve per model.  "
        "The table below maps the legend label and plot colour to the full model path.",
        "",
        "| # | Label | Colour | Line | Path |",
        "|---|-------|--------|------|------|",
    ]
    for i, (name, path) in enumerate(models.items()):
        colour = MODEL_COLOR_CYCLE[i % len(MODEL_COLOR_CYCLE)]
        ls = MODEL_LINESTYLES[i % len(MODEL_LINESTYLES)]
        ls_name = {"-": "solid", "--": "dashed", "-.": "dashdot", ":": "dotted"}.get(ls, ls)
        lines.append(f"| {i} | `{name}` | `{colour}` | {ls_name} | `{path}` |")

    lines += [""]
    readme.write_text("\n".join(lines))
    log_ok(f"  Model legend written: {readme}")


def append_accuracy_to_readme(all_metrics: list[dict], models: dict[str, str], outdir: str) -> None:
    """Append accuracy results to the README.md created by write_model_readme.

    Adds two sections:
    - **Aggregate accuracy per model** (Full-LC and peak, fraction classified as Ia
      and mean P(Ia)).
    - **Per-object breakdown** (P(Ia) and classification symbol for every SNID × model).

    Parameters
    ----------
    all_metrics : list[dict]
        Flat list of per-object × per-model metric dicts produced by
        ``get_metrics_all_models`` (keys: SNID, model, prob_ia_full,
        pred_class_full, prob_ia_peak, pred_class_peak).
    models : dict[str, str]
        short_name → .pt path (used only for ordering).
    outdir : str
        Directory where the README.md was written.
    """
    readme = Path(outdir) / "README.md"
    df = pd.DataFrame(all_metrics)
    model_names = list(models.keys())  # preserve discovery order

    import datetime
    has_true = "true_label" in df.columns and df["true_label"].notna().any()

    lines = [
        "",
        "---",
        "",
        "## Accuracy summary",
        "",
        f"_(computed {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})_",
        "",
    ]

    # ── Section 1: fraction classified as Ia (always written) ────────────────
    lines += [
        "> `+` = classified as Ia (P(Ia) > 0.5), `-` = non-Ia.  "
        "Fractions and mean P(Ia) are computed over all processed objects.",
        "",
        "### Aggregate per model — fraction classified as Ia",
        "",
        "| Model | Full-LC frac Ia | Full-LC mean P(Ia) | Peak frac Ia | Peak mean P(Ia) |",
        "|-------|----------------|-------------------|-------------|----------------|",
    ]
    for m in model_names:
        mdf = df[df["model"] == m]
        if mdf.empty:
            lines.append(f"| `{m}` | N/A | N/A | N/A | N/A |")
            continue
        valid_full = mdf["pred_class_full"] != -1
        valid_peak = mdf["pred_class_peak"] != -1
        frac_full = (mdf.loc[valid_full, "pred_class_full"] == 0).mean() if valid_full.any() else float("nan")
        frac_peak = (mdf.loc[valid_peak, "pred_class_peak"] == 0).mean() if valid_peak.any() else float("nan")
        mp_full   = mdf.loc[valid_full, "prob_ia_full"].mean() if valid_full.any() else float("nan")
        mp_peak   = mdf.loc[valid_peak, "prob_ia_peak"].mean() if valid_peak.any() else float("nan")
        lines.append(
            f"| `{m}` "
            f"| {frac_full:.0%} | {mp_full:.3f} "
            f"| {frac_peak:.0%} | {mp_peak:.3f} |"
        )

    # ── Section 2: true-label metrics (only when Type column is present) ──────
    if has_true:
        lines += [
            "",
            "### Aggregate per model — with true labels",
            "",
            "> True labels from the `Type` / `Obj. Type` column of the input CSV.  "
            "**SN Ia (any subtype)** is the positive class.",
            "",
            "Definitions: "
            "**Accuracy** = (TP+TN)/total · "
            "**Completeness** (recall) = TP/(TP+FN) · "
            "**Efficiency** (precision) = TP/(TP+FP)",
            "",
            "| Model | Full acc | Full compl | Full effic | Full mean P(Ia) "
            "| Peak acc | Peak compl | Peak effic | Peak mean P(Ia) |",
            "|-------|---------|-----------|-----------|---------------|"
            "---------|-----------|-----------|--------------|",
        ]

        def _readme_metrics(mdf_m: pd.DataFrame, pred_col: str, prob_col: str):
            labeled = mdf_m[mdf_m["true_label"].notna()]
            if labeled.empty:
                return "N/A", "N/A", "N/A", "N/A"
            valid = labeled[pred_col] != -1
            if not valid.any():
                return "N/A", "N/A", "N/A", "N/A"
            pred = (labeled.loc[valid, pred_col] == 0).astype(int)
            true = labeled.loc[valid, "true_label"].astype(int)
            tp = int(((pred == 1) & (true == 1)).sum())
            fp = int(((pred == 1) & (true == 0)).sum())
            fn = int(((pred == 0) & (true == 1)).sum())
            tn = int(((pred == 0) & (true == 0)).sum())
            total = tp + tn + fp + fn
            acc   = f"{(tp + tn) / total:.0%}" if total > 0 else "N/A"
            compl = f"{tp / (tp + fn):.0%}" if (tp + fn) > 0 else "N/A"
            effic = f"{tp / (tp + fp):.0%}" if (tp + fp) > 0 else "N/A"
            mp = labeled.loc[valid, prob_col].mean()
            mp_str = f"{mp:.3f}" if not np.isnan(mp) else "N/A"
            return acc, compl, effic, mp_str

        for m in model_names:
            mdf_m = df[df["model"] == m]
            if mdf_m.empty:
                lines.append(f"| `{m}` | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
                continue
            acc_f, c_f, e_f, mp_f = _readme_metrics(mdf_m, "pred_class_full", "prob_ia_full")
            acc_p, c_p, e_p, mp_p = _readme_metrics(mdf_m, "pred_class_peak", "prob_ia_peak")
            lines.append(
                f"| `{m}` | {acc_f} | {c_f} | {e_f} | {mp_f} "
                f"| {acc_p} | {c_p} | {e_p} | {mp_p} |"
            )

    # Per-object breakdown
    lines += [
        "",
        "### Per-object breakdown",
        "",
        "Symbol: `+` = classified as Ia, `-` = classified as non-Ia, `?` = failed",
        "",
    ]

    # Build header with one column per model × {full, peak}
    header_cols = "| SNID |"
    sep_cols = "|------|"
    if has_true:
        header_cols += " True type |"
        sep_cols += "-----------|"
    for m in model_names:
        short = m[:14]
        header_cols += f" {short} full | {short} peak |"
        sep_cols += "--------|--------|"
    lines += [header_cols, sep_cols]

    for snid, grp in df.groupby("SNID"):
        row_str = f"| `{snid}` |"
        if has_true:
            tt = grp["true_type"].iloc[0] if "true_type" in grp.columns else ""
            row_str += f" {tt} |"
        for m in model_names:
            mrow = grp[grp["model"] == m]
            if mrow.empty:
                row_str += " N/A | N/A |"
                continue
            r = mrow.iloc[0]
            sym_full = "+" if r["pred_class_full"] == 0 else ("-" if r["pred_class_full"] != -1 else "?")
            sym_peak = "+" if r["pred_class_peak"] == 0 else ("-" if r["pred_class_peak"] != -1 else "?")
            p_full = f"{r['prob_ia_full']:.3f}" if not np.isnan(r["prob_ia_full"]) else "N/A"
            p_peak = f"{r['prob_ia_peak']:.3f}" if not np.isnan(r["prob_ia_peak"]) else "N/A"
            row_str += f" {p_full} {sym_full} | {p_peak} {sym_peak} |"
        lines.append(row_str)

    lines += [""]
    with readme.open("a") as fh:
        fh.write("\n".join(lines))
    log_ok(f"  Accuracy results appended to: {readme}")


# ── Main orchestration ────────────────────────────────────────────────────────


def main(
    csv_file: str,
    models_dir: str,
    outdir: str,
    device: str,
    lc_outfile: str | None,
    max_objects: int | None,
    force_download: bool = False,
    avg_same_night: bool = False,
) -> None:
    """End-to-end pipeline: discover models → download LCs → classify → plot."""
    t0 = time.time()

    # ── 0. Discover models ────────────────────────────────────────────────────
    models = discover_models(models_dir)
    write_model_readme(models, outdir)

    # ── 1. Read candidate IDs ─────────────────────────────────────────────────
    log_info("Step 1: loading candidate IDs …")
    ids, type_map = load_candidate_ids(csv_file)
    if max_objects is not None:
        ids = ids[:max_objects]
        log_warn(f"  (capped at {max_objects} objects for this run)")

    # ── 2 & 3. Download + transform (or load from cache) ─────────────────────
    if lc_outfile and Path(lc_outfile).exists() and not force_download:
        log_warn(f"Step 2-3: cached file found → {lc_outfile}  (use --force_download to repull)")
        df_snn = pd.read_csv(lc_outfile, dtype={"SNID": str})
        float_cols = ["MJD", "FLUXCAL", "FLUXCALERR",
                      "HOSTGAL_SPECZ", "HOSTGAL_SPECZ_ERR",
                      "HOSTGAL_PHOTOZ", "HOSTGAL_PHOTOZ_ERR", "MWEBV"]
        df_snn[float_cols] = df_snn[float_cols].astype(np.float64)
    else:
        log_info("Step 2: downloading light-curves from Fink …")
        df_fink = download_all_lcs(ids)
        if df_fink is None:
            return
        log_info("Step 3: transforming to SuperNNova format …")
        df_snn = transform_to_snn_format(df_fink)
        if lc_outfile:
            df_snn.to_csv(lc_outfile, index=False)
            log_ok(f"  Light-curve CSV saved: {lc_outfile}")

    # ── 3b. Average intra-night same-band observations (optional) ─────────────
    if avg_same_night:
        log_info("Step 3b: averaging intra-night same-band observations …")
        df_snn = average_same_night_obs(df_snn)

    # ── 4, 4b & 5. Per-object: early preds + metrics + plot ──────────────────
    log_info("Step 4-5: early predictions, metrics and plots …")
    available_ids = df_snn["SNID"].unique()
    if max_objects is not None:
        available_ids = available_ids[:max_objects]
        log_warn(f"  (predictions capped at {max_objects} objects)")
    log_info(f"  {len(available_ids)} objects × {len(models)} models")

    all_metrics: list[dict] = []
    all_early_series: list[dict] = []   # for efficiency-vs-phase plot

    for obj_snid in available_ids:
        log_info(f"\n  ── Object {obj_snid} ──")
        df_obj = df_snn[df_snn["SNID"] == obj_snid].copy()

        # Resolve true type and label for this object (if available)
        true_type: str | None = (type_map.get(obj_snid, "") or "") if type_map else None
        true_label: int | None = None
        if true_type is not None:
            is_ia = _is_ia_type(true_type)
            true_label = (1 if is_ia else 0) if is_ia is not None else None
            if true_type:
                log_info(f"    True type: {true_type}  (label={'Ia' if true_label == 1 else 'non-Ia' if true_label == 0 else 'unknown'})")

        model_preds = run_early_predictions_all_models(df_obj, models, device)
        model_metrics = get_metrics_all_models(df_obj, models, device)

        # ── per-object accuracy snapshot ──────────────────────────────────────
        log_info(f"    {'Model':<28}  {'Full P(Ia)':>10}  {'Full cls':>8}  {'Peak P(Ia)':>10}  {'Peak cls':>8}")
        for row in model_metrics:
            p_full = f"{row['prob_ia_full']:.3f}" if not np.isnan(row["prob_ia_full"]) else "   N/A"
            p_peak = f"{row['prob_ia_peak']:.3f}" if not np.isnan(row["prob_ia_peak"]) else "   N/A"
            s_full = "Ia +" if row["pred_class_full"] == 0 else ("??" if row["pred_class_full"] == -1 else "non-Ia -")
            s_peak = "Ia +" if row["pred_class_peak"] == 0 else ("??" if row["pred_class_peak"] == -1 else "non-Ia -")
            log_info(f"    {row['model']:<28}  {p_full:>10}  {s_full:>8}  {p_peak:>10}  {s_peak:>8}")
            log_info(f"    {'':28}  {models.get(row['model'], 'unknown')}")

        for row in model_metrics:
            all_metrics.append({
                "SNID": obj_snid,
                "true_type": true_type or "",
                "true_label": true_label,
                **row,
            })

        # Collect time-series for efficiency-vs-phase plot
        peak_mjd = model_metrics[0]["peak_mjd"] if model_metrics else np.nan
        for model_name, (mjd_days, pia) in model_preds.items():
            if len(mjd_days) > 0:
                all_early_series.append({
                    "SNID": obj_snid,
                    "true_label": true_label,
                    "model": model_name,
                    "mjd_days": mjd_days,
                    "pia": pia,
                    "first_mjd": float(mjd_days[0]),
                    "peak_mjd": peak_mjd,
                })

        plot_lc_and_predictions(
            df_obj=df_obj,
            snid=obj_snid,
            model_preds=model_preds,
            model_metrics=model_metrics,
            outdir=outdir,
            true_type=true_type,
        )

    # ── 6. Summary + extra evaluation plots ──────────────────────────────────
    print_metrics_summary(all_metrics)
    append_accuracy_to_readme(all_metrics, models, outdir)

    # Save full per-object × per-model metrics (including std) as a CSV so that
    # downstream comparison scripts can access uncertainty estimates directly.
    metrics_csv = Path(outdir) / "metrics_all.csv"
    pd.DataFrame(all_metrics).to_csv(metrics_csv, index=False)
    log_ok(f"  Metrics CSV saved: {metrics_csv}")

    log_info("\nStep 7: extra evaluation plots …")
    plot_pr_curves(all_metrics, models, outdir)
    plot_efficiency_vs_phase(all_early_series, models, outdir)
    plot_uncertainty_vs_correctness(all_metrics, models, outdir)

    log_ok(f"\nDone! Total time: {time.time() - t0:.1f}s — plots in '{outdir}/'")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Download Fink light-curves and compare all SuperNNova models "
            "in a directory with per-object P(Ia) plots and accuracy summary."
        )
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="Fink_Rubin_alerts/interesting_objects_20260218.csv",
        help="CSV with diaObjectId column (default: interesting_objects_20260218.csv)",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default=(
            "/Users/amoller/Science/RubinLSST/LSST_obsv4_sims/ICML_2025/"
            "BNN_SWAG_Rubin/models"
        ),
        help="Directory containing one sub-folder per model (each with a .pt file)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="PyTorch device (default: cpu)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="Fink_Rubin_alerts/plots_fink_candidates",
        help="Directory to save per-object PNG plots (default: plots_fink_candidates/)",
    )
    parser.add_argument(
        "--max_objects",
        type=int,
        default=None,
        help="Cap the number of objects processed (useful for quick tests)",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        default=False,
        help="Re-download from Fink even if the cached CSV already exists",
    )
    parser.add_argument(
        "--avg_same_night",
        action="store_true",
        default=False,
        help=(
            "Average multiple observations of the same object, band, and night "
            "(floor(MJD) as night ID). Flux errors are propagated as "
            "sqrt(sum(σ²))/N for the mean."
        ),
    )

    args = parser.parse_args()

    main(
        csv_file=args.input_file,
        models_dir=args.models_dir,
        outdir=args.outdir,
        device=args.device,
        lc_outfile=f"{args.outdir}/sn_candidates_lcs_fink_SNANA.csv",
        max_objects=args.max_objects,
        force_download=args.force_download,
        avg_same_night=args.avg_same_night,
    )
