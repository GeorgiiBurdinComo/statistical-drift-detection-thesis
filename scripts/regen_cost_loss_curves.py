#!/usr/bin/env python3
"""Regenerate Chapter 4 loss-curve figure with embedded TrueType fonts."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "assets" / "metrics_export" / "table_expected_cost.csv"
OUT = ROOT / "assets" / "evidence" / "fig_cost_loss_curves.pdf"

W_VALUES = [0.2, 1.0, 3.0]
LAMBDA_MAX = 0.35
LAMBDA_GRID = np.linspace(0.0, LAMBDA_MAX, 400)

HIGHLIGHT = {
    "gpt-5-mini": ("C3", "gpt-5-mini"),
    "gpt-5-nano": ("C0", "gpt-5-nano"),
    "gpt-5": ("C2", "gpt-5"),
    "gpt-5.2": ("C1", "gpt-5.2"),
}

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
    }
)


def load_rows() -> list[dict]:
    with CSV_PATH.open(newline="") as f:
        return list(csv.DictReader(f))


def loss_curve(mean_cost: float, fp: float, fn: float, n: float, w: float) -> np.ndarray:
    weighted_error = (fp + w * fn) / ((1.0 + w) * n)
    return mean_cost + LAMBDA_GRID * weighted_error


def format_w(w: float) -> str:
    if float(w).is_integer():
        return str(int(w))
    return str(w)


def main() -> None:
    rows = load_rows()
    fig, axes = plt.subplots(len(W_VALUES), 1, figsize=(6.8, 8.8), sharex=True, sharey=True)

    handles = {}
    ymin = float("inf")
    ymax = float("-inf")

    for ax, w in zip(np.atleast_1d(axes), W_VALUES):
        for row in rows:
            model = row["model"]
            mean_cost = float(row["mean_cost"])
            fp = float(row["FP"])
            fn = float(row["FN"])
            n = float(row["n"])
            curve = loss_curve(mean_cost, fp, fn, n, w)
            ymin = min(ymin, float(curve.min()))
            ymax = max(ymax, float(curve.max()))

            if model in HIGHLIGHT:
                color, label = HIGHLIGHT[model]
                (line,) = ax.plot(LAMBDA_GRID, curve, color=color, lw=2.0, label=label, zorder=3)
                handles[label] = line

        ax.set_title(rf"$w={format_w(w)}$")
        ax.grid(True, color="0.90", lw=0.8)
        ax.set_xlabel(r"Total error severity $\lambda$")
        ax.axvline(0.186, color="0.35", lw=0.9, ls="--", zorder=2)
        ax.text(
            0.186,
            0.98,
            r"$\lambda=0.186$",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=7.5,
            color="0.30",
            rotation=90,
        )

    for ax in np.atleast_1d(axes):
        ax.set_ylabel(r"Assigned loss per request $L_t(m;w,\lambda)$ [USD]")

    pad = 0.08 * (ymax - ymin)
    axes[0].set_ylim(max(0.0, ymin - pad), ymax + pad)
    axes[0].set_xlim(0.0, LAMBDA_MAX)

    legend_labels = [label for _, label in HIGHLIGHT.values()]
    fig.legend(
        [handles[label] for label in legend_labels],
        legend_labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=220)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
