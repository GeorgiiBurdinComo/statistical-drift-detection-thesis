#!/usr/bin/env python3
"""Regenerate fig_cost_recall_pareto.pdf with embedded TrueType fonts."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "assets" / "metrics_export" / "table_expected_cost.csv"
OUT = ROOT / "assets" / "evidence" / "fig_cost_recall_pareto.pdf"

TAU = 0.90
BUDGET = 0.002

# Short labels for crowded scatter
SHORT = {
    "gpt-5": "gpt-5",
    "gpt-5.2": "5.2",
    "claude-sonnet-4-6": "sonnet-4.6",
    "gpt-5.4": "5.4",
    "gpt-5.1": "5.1",
    "gpt-5-mini": "mini",
    "gpt-5-nano": "nano",
    "claude-sonnet-4-5": "sonnet-4.5",
    "claude-haiku-4-5": "haiku",
    "gpt-4.1-mini": "4.1-mini",
    "gpt-4.1-nano": "4.1-nano",
}

# Nudge labels away from overlaps (dx, dy) in data units-ish via offset points
LABEL_OFFSET = {
    "gpt-5": (6, 6),
    "gpt-5.2": (6, -10),
    "claude-sonnet-4-6": (6, 4),
    "gpt-5.4": (6, -8),
    "gpt-5.1": (6, 6),
    "gpt-5-mini": (6, 8),
    "gpt-5-nano": (6, -10),
    "claude-sonnet-4-5": (6, 4),
    "claude-haiku-4-5": (6, -8),
    "gpt-4.1-mini": (6, 6),
    "gpt-4.1-nano": (6, -8),
}

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    }
)


def load_rows() -> list[dict]:
    with CSV_PATH.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    rows = load_rows()
    costs = np.array([float(r["inference_cost"]) for r in rows])
    recalls = np.array([float(r["recall"]) for r in rows])
    jstd = np.array([float(r["J_production"]) for r in rows])
    models = [r["model"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.8, 4.8))

    # Feasible region: cost <= B, recall >= tau
    x_min, x_max = 1e-4, max(costs.max() * 1.15, BUDGET * 3)
    y_min, y_max = 0.65, 1.02
    ax.add_patch(
        Rectangle(
            (x_min, TAU),
            BUDGET - x_min,
            y_max - TAU,
            facecolor="0.85",
            edgecolor="none",
            alpha=0.55,
            zorder=0,
            label=rf"Feasible ($\tau={TAU:.2f}$, $B={BUDGET}$)",
        )
    )

    ax.axhline(TAU, color="0.35", ls="--", lw=1.0, zorder=1)
    ax.axvline(BUDGET, color="0.35", ls="--", lw=1.0, zorder=1)
    ax.text(
        x_max * 0.98,
        TAU + 0.008,
        rf"$\tau_{{\mathrm{{rec}}}}={TAU:.2f}$",
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.25",
    )
    ax.text(
        BUDGET * 1.05,
        y_min + 0.015,
        rf"$B={BUDGET}$",
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.25",
        rotation=90,
    )

    sc = ax.scatter(
        costs,
        recalls,
        c=jstd,
        cmap="viridis_r",
        s=55,
        zorder=3,
        edgecolors="0.2",
        linewidths=0.4,
    )

    for m, x, y in zip(models, costs, recalls):
        dx, dy = LABEL_OFFSET.get(m, (6, 4))
        ax.annotate(
            SHORT.get(m, m),
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=7,
            color="0.15",
            zorder=4,
        )

    # Highlight illustrative winners
    by_model = {r["model"]: r for r in rows}
    mini = by_model["gpt-5-mini"]
    gpt5 = by_model["gpt-5"]
    ax.scatter(
        [float(mini["inference_cost"])],
        [float(mini["recall"])],
        marker="*",
        s=220,
        facecolors="none",
        edgecolors="C3",
        linewidths=1.4,
        zorder=5,
        label=r"Cap winner (known $J_{\mathrm{std}}$): gpt-5-mini",
    )
    ax.scatter(
        [float(gpt5["inference_cost"])],
        [float(gpt5["recall"])],
        marker="s",
        s=90,
        facecolors="none",
        edgecolors="C1",
        linewidths=1.4,
        zorder=5,
        label=r"No-cap winner (known $J_{\mathrm{std}}$): gpt-5",
    )

    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Inference cost per request [USD]")
    ax.set_ylabel("Recall")
    ax.set_title("Candidates in the cost–recall plane")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$J_{\mathrm{std}}$")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.92)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=220)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
