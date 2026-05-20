"""
compare_fink_runs.py

Compare SuperNNova classification metrics between two output directories:
  - raw light-curves    (default: Fink_Rubin_alerts/plots_tns_SN)
  - co-added photometry (default: Fink_Rubin_alerts/plots_tns_SN_coadd)

Reads the README.md files written by run_fink_candidates_onthefly.py and
produces three comparison figures saved to --outdir:

  comparison_heatmap.png   — delta (coadd − raw) heatmap: models × metrics
  comparison_bars.png      — raw vs coadd grouped bar chart per model
  comparison_pia_scatter.png — per-object mean P(Ia) scatter, raw vs coadd

Usage:
    .venv/bin/python compare_fink_runs.py \\
        --raw_dir   Fink_Rubin_alerts/plots_tns_SN \\
        --coadd_dir Fink_Rubin_alerts/plots_tns_SN_coadd \\
        --outdir    Fink_Rubin_alerts/comparison
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

matplotlib.use("Agg")

# ── Colours ───────────────────────────────────────────────────────────────────

TYPE_COLORS = {
    "SN Ia": "#e41a1c",
    "SN II": "#377eb8",
    "SN I":  "#ff7f00",
    "SN Ib": "#984ea3",
    "SN IIn": "#4daf4a",
}
DEFAULT_COLOR = "#999999"

# ── Markdown parsing ──────────────────────────────────────────────────────────


def _parse_md_table_after(text: str, section_marker: str) -> pd.DataFrame | None:
    """Return the first markdown table found after `section_marker` as a raw
    string DataFrame (no type conversions yet).  Returns None if not found."""
    idx = text.find(section_marker)
    if idx == -1:
        return None

    lines, in_table = [], False
    for line in text[idx:].splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            lines.append(stripped)
            in_table = True
        elif in_table:
            break  # first non-table line ends the block

    if len(lines) < 3:
        return None

    # Parse header (position 0) and data rows (positions 2+, skip separator)
    def _split_row(row: str) -> list[str]:
        return [c.strip() for c in row.split("|") if c.strip()]

    headers = _split_row(lines[0])
    rows = [_split_row(l) for l in lines[2:] if _split_row(l)]
    if not rows:
        return None

    # Deduplicate column names (pandas behaviour for duplicate headers)
    seen: dict[str, int] = {}
    deduped = []
    for h in headers:
        if h in seen:
            seen[h] += 1
            deduped.append(f"{h}.{seen[h]}")
        else:
            seen[h] = 0
            deduped.append(h)

    n = len(deduped)
    df = pd.DataFrame([r[:n] for r in rows], columns=deduped)
    return df


def _to_float(s: str) -> float:
    """'88%' → 0.88 · '0.438' → 0.438 · 'N/A'/'' → NaN · strip backticks."""
    s = str(s).strip().strip("`").strip()
    if not s or s == "N/A":
        return np.nan
    if s.endswith("%"):
        return float(s[:-1]) / 100.0
    try:
        return float(s)
    except ValueError:
        return np.nan


def _extract_pia(cell: str) -> float:
    """Extract the leading float from a cell like '0.987 +' or '0.023 -'."""
    m = re.match(r"([\d.]+)", str(cell).strip())
    return float(m.group(1)) if m else np.nan


# ── Loaders ───────────────────────────────────────────────────────────────────


def load_true_label_metrics(readme: Path) -> pd.DataFrame | None:
    """Parse the 'Aggregate per model — with true labels' table."""
    text = readme.read_text()
    df = _parse_md_table_after(text, "Aggregate per model — with true labels")
    if df is None:
        print(f"  [warn] True-label table not found in {readme}", file=sys.stderr)
        return None
    model_col = df.columns[0]
    df[model_col] = df[model_col].str.strip("` ")
    for col in df.columns[1:]:
        df[col] = df[col].apply(_to_float)
    return df.set_index(model_col)


def load_frac_ia_metrics(readme: Path) -> pd.DataFrame | None:
    """Parse the 'Aggregate per model — fraction classified as Ia' table."""
    text = readme.read_text()
    df = _parse_md_table_after(text, "Aggregate per model — fraction classified as Ia")
    if df is None:
        return None
    model_col = df.columns[0]
    df[model_col] = df[model_col].str.strip("` ")
    for col in df.columns[1:]:
        df[col] = df[col].apply(_to_float)
    return df.set_index(model_col)


def load_per_object(readme: Path) -> pd.DataFrame | None:
    """Parse the per-object breakdown table.

    Returns a DataFrame with columns:
        SNID, true_type, mean_pia_full, mean_pia_peak
    where the mean is taken across all models.
    """
    text = readme.read_text()
    df = _parse_md_table_after(text, "Per-object breakdown")
    if df is None:
        return None

    snid_col = df.columns[0]
    type_col = df.columns[1]

    result = pd.DataFrame()
    result["SNID"] = df[snid_col].str.strip("` ")
    result["true_type"] = df[type_col].str.strip()

    # Remaining columns alternate full / peak for each model
    metric_cols = df.columns[2:]
    n_models = len(metric_cols) // 2

    full_vals = np.full((len(df), n_models), np.nan)
    peak_vals = np.full((len(df), n_models), np.nan)

    for j in range(n_models):
        full_col = metric_cols[2 * j]
        peak_col = metric_cols[2 * j + 1]
        full_vals[:, j] = df[full_col].apply(_extract_pia).values
        peak_vals[:, j] = df[peak_col].apply(_extract_pia).values

    result["mean_pia_full"] = np.nanmean(full_vals, axis=1)
    result["mean_pia_peak"] = np.nanmean(peak_vals, axis=1)
    return result


# ── Plots ─────────────────────────────────────────────────────────────────────


def plot_delta_heatmap(
    raw: pd.DataFrame, coadd: pd.DataFrame, outdir: Path, label: str = "true_label"
) -> None:
    """Delta heatmap: coadd − raw for every (model, metric) cell.

    Green = coadd is better, red = coadd is worse, white = no change.
    """
    # Align on shared models and columns
    models = raw.index.intersection(coadd.index).tolist()
    cols   = raw.columns.intersection(coadd.columns).tolist()
    if not models or not cols:
        print("  [warn] No shared models/metrics for heatmap — skipping.", file=sys.stderr)
        return

    delta = coadd.loc[models, cols].values - raw.loc[models, cols].values  # (n_models, n_metrics)

    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 1.1), max(4, len(models) * 0.55)))

    vmax = np.nanmax(np.abs(delta)) or 0.1
    im = ax.imshow(delta, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(cols)):
            val = delta[i, j]
            if np.isnan(val):
                txt = "N/A"
            else:
                txt = f"{val:+.0%}" if abs(val) < 1 else f"{val:+.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black" if abs(val) < vmax * 0.6 else "white")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.set_title("Δ metrics: co-added − raw  (green = improvement)", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Δ (coadd − raw)")

    plt.tight_layout()
    out = outdir / f"comparison_heatmap_{label}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_comparison_bars(
    raw: pd.DataFrame, coadd: pd.DataFrame, outdir: Path
) -> None:
    """Grouped bar chart: raw vs coadd side-by-side for key true-label metrics."""
    key_metrics = [c for c in [
        "Full acc", "Full compl", "Full effic",
        "Peak acc", "Peak compl", "Peak effic",
    ] if c in raw.columns and c in coadd.columns]

    if not key_metrics:
        print("  [warn] No key metrics found for bar chart — skipping.", file=sys.stderr)
        return

    models = raw.index.intersection(coadd.index).tolist()
    n_models  = len(models)
    n_metrics = len(key_metrics)

    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(2.5 * n_metrics, max(4, 0.35 * n_models + 2)),
        sharey=False,
    )
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_models)
    w = 0.35

    for ax, metric in zip(axes, key_metrics):
        r_vals = raw.loc[models, metric].values.astype(float)
        c_vals = coadd.loc[models, metric].values.astype(float)

        bars_r = ax.barh(x + w / 2, r_vals,   height=w, color="#e41a1c", alpha=0.8, label="Raw")
        bars_c = ax.barh(x - w / 2, c_vals,   height=w, color="#377eb8", alpha=0.8, label="Co-add")

        ax.set_yticks(x)
        ax.set_yticklabels(models, fontsize=7)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel(metric, fontsize=8)
        ax.axvline(0.5, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.grid(axis="x", alpha=0.3)
        if ax is axes[0]:
            ax.legend(fontsize=7)

    fig.suptitle("Raw vs Co-added photometry — key metrics per model", fontsize=10)
    plt.tight_layout()
    out = outdir / "comparison_bars.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_pia_scatter(raw_obj: pd.DataFrame, coadd_obj: pd.DataFrame, outdir: Path) -> None:
    """Scatter raw vs coadd mean P(Ia) per object, coloured by true type."""
    merged = raw_obj.merge(coadd_obj, on=["SNID", "true_type"],
                           suffixes=("_raw", "_coadd"))
    if merged.empty:
        print("  [warn] No shared objects for scatter — skipping.", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, xcol, ycol, title in [
        (axes[0], "mean_pia_full_raw",  "mean_pia_full_coadd",  "Full light-curve"),
        (axes[1], "mean_pia_peak_raw",  "mean_pia_peak_coadd",  "At peak"),
    ]:
        # Diagonal reference
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.7, alpha=0.5)
        ax.axvline(0.5, color="grey", linestyle=":", linewidth=0.7, alpha=0.5)

        for _, row in merged.iterrows():
            tt = str(row["true_type"]).strip()
            # Normalise subtype to broad class for colour lookup
            color_key = next((k for k in TYPE_COLORS if tt.startswith(k)), None)
            color = TYPE_COLORS.get(color_key, DEFAULT_COLOR)
            ax.scatter(row[xcol], row[ycol], color=color, s=60, alpha=0.85, zorder=3)
            ax.annotate(
                str(row["SNID"])[-6:],           # last 6 digits to keep it compact
                (row[xcol], row[ycol]),
                fontsize=5, xytext=(3, 3), textcoords="offset points",
            )

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Mean P(Ia) — raw", fontsize=9)
        ax.set_ylabel("Mean P(Ia) — co-added", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)

    # Legend for types
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markersize=8, label=t)
        for t, c in TYPE_COLORS.items()
    ]
    axes[1].legend(handles=handles, fontsize=7, loc="lower right")

    fig.suptitle("Per-object mean P(Ia): raw vs co-added  (above diagonal = coadd higher)",
                 fontsize=10)
    plt.tight_layout()
    out = outdir / "comparison_pia_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Uncertainty quantification comparison ────────────────────────────────────


def load_metrics_csv(run_dir: Path) -> pd.DataFrame | None:
    """Load the metrics_all.csv saved by run_fink_candidates_onthefly.py.

    Returns None (with a warning) when the file is absent — the caller must
    re-run the main script with the updated version to generate it.
    """
    csv_path = run_dir / "metrics_all.csv"
    if not csv_path.exists():
        print(
            f"  [warn] {csv_path} not found.\n"
            "         Re-run run_fink_candidates_onthefly.py to generate it.",
            file=sys.stderr,
        )
        return None
    df = pd.read_csv(csv_path, dtype={"SNID": str})
    if "std_ia_full" not in df.columns:
        print(
            f"  [warn] {csv_path} has no 'std_ia_full' column — "
            "re-run the main script with the latest version.",
            file=sys.stderr,
        )
        return None
    return df


def plot_uncertainty_comparison(
    raw_df: pd.DataFrame,
    coadd_df: pd.DataFrame,
    outdir: Path,
) -> None:
    """Three-panel uncertainty comparison between raw and co-added runs.

    Panel 1 — Mean std per model (bar chart): how much uncertainty each model
    produces on average, for full-LC and at-peak.

    Panel 2 — Per-object std scatter (raw vs coadd): each point is one object
    averaged across models. Points below the diagonal = co-adding reduces
    uncertainty. Coloured by true type.

    Panel 3 — Uncertainty vs |ΔP(Ia)|: does higher uncertainty correlate with
    larger P(Ia) change between raw and coadd? One point per object × model.
    """

    # ── Per-model aggregate uncertainty ──────────────────────────────────────
    def _model_std(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.groupby("model")[["std_ia_full", "std_ia_peak"]]
            .mean()
            .rename(columns={"std_ia_full": "Full-LC std", "std_ia_peak": "Peak std"})
        )

    raw_std   = _model_std(raw_df)
    coadd_std = _model_std(coadd_df)
    models = raw_std.index.intersection(coadd_std.index).tolist()

    if not models:
        print("  [warn] No shared models for UQ comparison.", file=sys.stderr)
        return

    # ── Print console delta ───────────────────────────────────────────────────
    _RESET = "\033[0m"; _GREEN = "\033[32m"; _RED = "\033[31m"; _GREY = "\033[90m"
    print("\n── Uncertainty (std P(Ia)) — Δ coadd − raw ──")
    print("  (green = co-adding reduces uncertainty)\n")
    for col in ["Full-LC std", "Peak std"]:
        print(f"  {'Model':25s}  {'Raw':>8}  {'Co-add':>8}  {'Δ':>8}")
        print("  " + "─" * 56)
        for m in models:
            r = raw_std.loc[m, col]
            c = coadd_std.loc[m, col]
            d = c - r
            # Lower std = less uncertainty = improvement → green when delta < 0
            colour = _GREEN if d < -0.001 else (_RED if d > 0.001 else _GREY)
            print(f"  {m:25s}  {r:8.4f}  {c:8.4f}  {colour}{d:+8.4f}{_RESET}")
        print()

    # ── Figure 1: mean std per model ─────────────────────────────────────────
    n_models = len(models)
    x = np.arange(n_models)
    w = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, n_models * 0.45 + 2)))

    for ax, col, title in [
        (axes[0], "Full-LC std", "Full light-curve"),
        (axes[1], "Peak std",    "At peak"),
    ]:
        r_vals = raw_std.loc[models, col].values
        c_vals = coadd_std.loc[models, col].values

        ax.barh(x + w / 2, r_vals, height=w, color="#e41a1c", alpha=0.8, label="Raw")
        ax.barh(x - w / 2, c_vals, height=w, color="#377eb8", alpha=0.8, label="Co-add")
        ax.set_yticks(x)
        ax.set_yticklabels(models, fontsize=8)
        ax.set_xlabel("Mean std P(Ia)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Mean uncertainty per model: raw vs co-added  (lower = more confident)",
                 fontsize=10)
    plt.tight_layout()
    out1 = outdir / "uncertainty_mean_per_model.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out1}")

    # ── Figure 2: per-object std scatter ─────────────────────────────────────
    raw_obj = (
        raw_df.groupby("SNID")[["std_ia_full", "std_ia_peak", "true_type"]]
        .agg({"std_ia_full": "mean", "std_ia_peak": "mean", "true_type": "first"})
        .reset_index()
    )
    coadd_obj = (
        coadd_df.groupby("SNID")[["std_ia_full", "std_ia_peak"]]
        .mean()
        .reset_index()
    )
    merged = raw_obj.merge(coadd_obj, on="SNID", suffixes=("_raw", "_coadd"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, rc, cc, title in [
        (axes[0], "std_ia_full_raw", "std_ia_full_coadd", "Full light-curve"),
        (axes[1], "std_ia_peak_raw", "std_ia_peak_coadd", "At peak"),
    ]:
        lim_max = max(merged[rc].max(), merged[cc].max()) * 1.1
        ax.plot([0, lim_max], [0, lim_max], color="grey", linestyle="--",
                linewidth=0.8, alpha=0.6)
        ax.axhline(0, color="grey", linewidth=0.5)

        for _, row in merged.iterrows():
            tt = str(row["true_type"]).strip()
            ck = next((k for k in TYPE_COLORS if tt.startswith(k)), None)
            color = TYPE_COLORS.get(ck, DEFAULT_COLOR)
            ax.scatter(row[rc], row[cc], color=color, s=55, alpha=0.85, zorder=3)
            ax.annotate(str(row["SNID"])[-6:], (row[rc], row[cc]),
                        fontsize=5, xytext=(3, 3), textcoords="offset points")

        ax.set_xlim(left=-0.002)
        ax.set_ylim(bottom=-0.002)
        ax.set_xlabel("Std P(Ia) — raw", fontsize=9)
        ax.set_ylabel("Std P(Ia) — co-added", fontsize=9)
        ax.set_title(f"{title}  (below diagonal = co-add more confident)", fontsize=9)
        ax.grid(alpha=0.3)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markersize=8, label=t)
        for t, c in TYPE_COLORS.items()
    ]
    axes[1].legend(handles=handles, fontsize=7, loc="lower right")
    fig.suptitle("Per-object uncertainty: raw vs co-added  (mean std across models)",
                 fontsize=10)
    plt.tight_layout()
    out2 = outdir / "uncertainty_per_object_scatter.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out2}")

    # ── Figure 3: uncertainty vs |ΔP(Ia)| ────────────────────────────────────
    shared_cols = ["SNID", "model", "prob_ia_full", "prob_ia_peak",
                   "std_ia_full", "std_ia_peak", "true_label"]
    shared_cols = [c for c in shared_cols
                   if c in raw_df.columns and c in coadd_df.columns]
    merged_m = raw_df[shared_cols].merge(
        coadd_df[shared_cols], on=["SNID", "model"], suffixes=("_raw", "_coadd")
    )
    if merged_m.empty:
        return

    merged_m["delta_pia_full"] = (
        merged_m["prob_ia_full_coadd"] - merged_m["prob_ia_full_raw"]
    ).abs()
    merged_m["delta_pia_peak"] = (
        merged_m["prob_ia_peak_coadd"] - merged_m["prob_ia_peak_raw"]
    ).abs()
    # Average std across raw and coadd as a proxy for overall uncertainty
    merged_m["mean_std_full"] = (
        merged_m["std_ia_full_raw"] + merged_m["std_ia_full_coadd"]
    ) / 2
    merged_m["mean_std_peak"] = (
        merged_m["std_ia_peak_raw"] + merged_m["std_ia_peak_coadd"]
    ) / 2

    has_labels = "true_label_raw" in merged_m.columns and \
                 merged_m["true_label_raw"].notna().any()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, xcol, ycol, title in [
        (axes[0], "mean_std_full", "delta_pia_full", "Full light-curve"),
        (axes[1], "mean_std_peak", "delta_pia_peak", "At peak"),
    ]:
        if has_labels:
            true_lbl = merged_m["true_label_raw"]
            colors = np.where(true_lbl == 1, "#e41a1c", "#377eb8")
        else:
            colors = ["steelblue"] * len(merged_m)

        ax.scatter(merged_m[xcol], merged_m[ycol], c=colors,
                   alpha=0.5, s=30, zorder=2)

        # Correlation annotation
        valid = merged_m[[xcol, ycol]].dropna()
        if len(valid) > 2:
            corr = valid.corr().iloc[0, 1]
            ax.text(0.05, 0.93, f"r = {corr:.2f}", transform=ax.transAxes,
                    fontsize=9, color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

        ax.set_xlabel("Mean std P(Ia)  [uncertainty]", fontsize=9)
        ax.set_ylabel("|ΔP(Ia)| between raw and co-added", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)

    if has_labels:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="#e41a1c", markersize=8, label="SN Ia"),
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="#377eb8", markersize=8, label="non-Ia"),
        ]
        axes[1].legend(handles=handles, fontsize=8)

    fig.suptitle("Does higher uncertainty → larger P(Ia) change between raw and co-added?",
                 fontsize=10)
    plt.tight_layout()
    out3 = outdir / "uncertainty_vs_delta_pia.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out3}")


# ── Console summary ───────────────────────────────────────────────────────────


def print_delta_table(raw: pd.DataFrame, coadd: pd.DataFrame, title: str) -> None:
    """Print a coloured delta table to the terminal."""
    _RESET = "\033[0m"
    _GREEN = "\033[32m"
    _RED   = "\033[31m"
    _GREY  = "\033[90m"

    models = raw.index.intersection(coadd.index).tolist()
    cols   = raw.columns.intersection(coadd.columns).tolist()
    if not models or not cols:
        return

    delta = coadd.loc[models, cols] - raw.loc[models, cols]

    col_w = 10
    print(f"\n{title}")
    print("  (Δ = co-added − raw; green = improvement, red = regression)\n")
    header = f"  {'Model':25s}" + "".join(f"  {c[:col_w]:>{col_w}}" for c in cols)
    print(header)
    print("  " + "─" * (len(header) - 2))
    for m in models:
        row_str = f"  {m:25s}"
        for c in cols:
            val = delta.loc[m, c]
            if np.isnan(val):
                row_str += f"  {'N/A':>{col_w}}"
            else:
                fmt = f"{val:+.0%}" if abs(val) <= 1.0 else f"{val:+.3f}"
                colour = _GREEN if val > 0.005 else (_RED if val < -0.005 else _GREY)
                row_str += f"  {colour}{fmt:>{col_w}}{_RESET}"
        print(row_str)


# ── Main ──────────────────────────────────────────────────────────────────────


def main(raw_dir: str, coadd_dir: str, outdir: str) -> None:
    raw_readme   = Path(raw_dir)   / "README.md"
    coadd_readme = Path(coadd_dir) / "README.md"
    out          = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    for p in [raw_readme, coadd_readme]:
        if not p.exists():
            sys.exit(f"ERROR: README not found: {p}")

    print(f"Raw   : {raw_readme}")
    print(f"Co-add: {coadd_readme}")
    print(f"Output: {out}\n")

    # ── True-label metrics ────────────────────────────────────────────────────
    raw_true   = load_true_label_metrics(raw_readme)
    coadd_true = load_true_label_metrics(coadd_readme)

    if raw_true is not None and coadd_true is not None:
        print_delta_table(raw_true, coadd_true, "── True-label metrics (Δ coadd − raw) ──")
        plot_delta_heatmap(raw_true, coadd_true, out, label="true_label")
        plot_comparison_bars(raw_true, coadd_true, out)
    else:
        print("[warn] True-label metrics missing from one or both READMEs.")

    # ── Frac-Ia metrics ───────────────────────────────────────────────────────
    raw_frac   = load_frac_ia_metrics(raw_readme)
    coadd_frac = load_frac_ia_metrics(coadd_readme)

    if raw_frac is not None and coadd_frac is not None:
        print_delta_table(raw_frac, coadd_frac, "\n── Frac-Ia metrics (Δ coadd − raw) ──")
        plot_delta_heatmap(raw_frac, coadd_frac, out, label="frac_ia")

    # ── Per-object P(Ia) scatter ──────────────────────────────────────────────
    raw_obj   = load_per_object(raw_readme)
    coadd_obj = load_per_object(coadd_readme)

    if raw_obj is not None and coadd_obj is not None:
        plot_pia_scatter(raw_obj, coadd_obj, out)
    else:
        print("[warn] Per-object table missing — scatter plot skipped.")

    # ── Uncertainty quantification comparison ─────────────────────────────────
    raw_mcsv   = load_metrics_csv(Path(raw_dir))
    coadd_mcsv = load_metrics_csv(Path(coadd_dir))

    if raw_mcsv is not None and coadd_mcsv is not None:
        plot_uncertainty_comparison(raw_mcsv, coadd_mcsv, out)
    else:
        print("[info] UQ comparison skipped — re-run the main script to generate "
              "metrics_all.csv files.")

    print("\nDone.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare SuperNNova metrics between raw and co-added Fink runs."
    )
    parser.add_argument(
        "--raw_dir",
        default="Fink_Rubin_alerts/plots_tns_SN",
        help="Output directory from the raw-photometry run",
    )
    parser.add_argument(
        "--coadd_dir",
        default="Fink_Rubin_alerts/plots_tns_SN_coadd",
        help="Output directory from the co-added-photometry run",
    )
    parser.add_argument(
        "--outdir",
        default="Fink_Rubin_alerts/comparison",
        help="Where to save the comparison figures (default: Fink_Rubin_alerts/comparison)",
    )
    args = parser.parse_args()
    main(args.raw_dir, args.coadd_dir, args.outdir)
