#!/usr/bin/env python3
"""Regenerate Chapter 4 penalty-map figure with embedded TrueType fonts."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "assets" / "metrics_export" / "table_expected_cost.csv"
OUT = ROOT / "assets" / "evidence" / "fig_cost_penalty_map.pdf"

SELECTED_MODELS = ["gpt-5-nano", "gpt-5-mini", "gpt-5.1", "gpt-5"]
LINEAR_MAX = 10.0
LOG_MIN = 0.01
LOG_MAX = 10.0
N_GRID = 801

PALETTE = ["#dbeafe", "#93c5fd", "#60a5fa", "#2563eb"]
LINEAR_LABELS = {
    "gpt-5-nano": dict(xy=(0.13, 7.2), rotation=90),
    "gpt-5-mini": dict(xy=(1.15, 6.4), rotation=0),
    "gpt-5.1": dict(xy=(3.15, 5.4), rotation=0),
    "gpt-5": dict(xy=(7.2, 3.0), rotation=0),
}
LOG_LABELS = {
    "gpt-5-nano": dict(xy=(0.028, 2.4), rotation=90),
    "gpt-5-mini": dict(xy=(0.22, 1.8), rotation=0),
    "gpt-5.1": dict(xy=(1.05, 2.2), rotation=0),
    "gpt-5": dict(xy=(4.2, 0.22), rotation=0),
}

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
    }
)


def load_rows() -> list[dict]:
    with CSV_PATH.open(newline="") as f:
        return list(csv.DictReader(f))


def winner_grid(mean_cost, fp, fn, n, c_fn, c_fp):
    X, Y = np.meshgrid(c_fn, c_fp)
    losses = (
        mean_cost[:, None, None]
        + (Y[None, :, :] * fp[:, None, None] + X[None, :, :] * fn[:, None, None])
        / n[:, None, None]
    )
    winner_idx = np.argmin(losses, axis=0)
    present = sorted(int(i) for i in np.unique(winner_idx))
    remap = {old: new for new, old in enumerate(present)}
    display_idx = np.vectorize(remap.get)(winner_idx).astype(float)
    return X, Y, display_idx, present


def draw_panel(ax, X, Y, display_idx, present, models, cmap, labels, title):
    ax.pcolormesh(
        X,
        Y,
        display_idx,
        shading="nearest",
        cmap=cmap,
        vmin=-0.5,
        vmax=len(present) - 0.5,
        rasterized=True,
    )
    ax.contour(
        X,
        Y,
        display_idx,
        levels=np.arange(len(present) - 1) + 0.5,
        colors="white",
        linewidths=0.7,
    )
    for old_idx in present:
        model = models[old_idx]
        spec = labels[model]
        ax.text(
            spec["xy"][0],
            spec["xy"][1],
            model,
            ha="center",
            va="center",
            rotation=spec["rotation"],
            fontsize=7.5,
            color="black",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
        )
    ax.set_xlabel(r"$C_{\mathrm{FN}}$ [USD]")
    ax.set_ylabel(r"$C_{\mathrm{FP}}$ [USD]")
    ax.set_title(title)
    ax.grid(color="white", lw=0.5, alpha=0.35)


def main() -> None:
    rows_by_model = {row["model"]: row for row in load_rows()}
    rows = [rows_by_model[model] for model in SELECTED_MODELS]
    models = [r["model"] for r in rows]
    mean_cost = np.array([float(r["mean_cost"]) for r in rows])
    fp = np.array([float(r["FP"]) for r in rows])
    fn = np.array([float(r["FN"]) for r in rows])
    n = np.array([float(r["n"]) for r in rows])

    cmap = ListedColormap(PALETTE[: len(SELECTED_MODELS)])

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.55))

    c_fn_lin = np.linspace(0.0, LINEAR_MAX, N_GRID)
    c_fp_lin = np.linspace(0.0, LINEAR_MAX, N_GRID)
    X_lin, Y_lin, disp_lin, present_lin = winner_grid(
        mean_cost, fp, fn, n, c_fn_lin, c_fp_lin
    )
    draw_panel(
        axes[0],
        X_lin,
        Y_lin,
        disp_lin,
        present_lin,
        models,
        cmap,
        LINEAR_LABELS,
        r"(a) Linear, $0$–$10$ USD",
    )
    axes[0].set_xlim(0.0, LINEAR_MAX)
    axes[0].set_ylim(0.0, LINEAR_MAX)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].xaxis.set_major_locator(MultipleLocator(2))
    axes[0].yaxis.set_major_locator(MultipleLocator(2))

    c_fn_log = np.logspace(np.log10(LOG_MIN), np.log10(LOG_MAX), N_GRID)
    c_fp_log = np.logspace(np.log10(LOG_MIN), np.log10(LOG_MAX), N_GRID)
    X_log, Y_log, disp_log, present_log = winner_grid(
        mean_cost, fp, fn, n, c_fn_log, c_fp_log
    )
    draw_panel(
        axes[1],
        X_log,
        Y_log,
        disp_log,
        present_log,
        models,
        cmap,
        LOG_LABELS,
        r"(b) Logarithmic, $10^{-2}$–$10$ USD",
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlim(LOG_MIN, LOG_MAX)
    axes[1].set_ylim(LOG_MIN, LOG_MAX)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].xaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    axes[1].yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].yaxis.set_minor_formatter(NullFormatter())

    fig.tight_layout(w_pad=1.6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=220)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
