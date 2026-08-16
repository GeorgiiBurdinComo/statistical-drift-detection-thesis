#!/usr/bin/env python3
"""Regenerate the disagreement matrix with a per-row vote-share strip."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets" / "metrics_export" / "canonical_disagreement_matrix_source.csv"
SUMMARY = ROOT / "assets" / "metrics_export" / "canonical_disagreement_matrix_summary.json"
PANEL_IDS = ROOT / "assets" / "metrics_export" / "panel_ids.txt"
OUT = ROOT / "assets" / "evidence" / "fig_canonical_disagreement_matrix.pdf"

BLUE = "#4e79a7"
RED = "#f28e2b"
MISSING = "#d9d9d9"
INK = "#333333"
def main() -> None:
    source = pd.read_csv(SOURCE)
    summary = json.loads(SUMMARY.read_text())
    panel_ids = {line.strip() for line in PANEL_IDS.read_text().splitlines() if line.strip()}

    model_cols = [entry["model"] for entry in summary["selected_models"]]
    preds = source[model_cols].astype(float).to_numpy()
    truth = source["label"].astype(float).to_numpy()[:, None]
    coverage = source["coverage"].to_numpy(dtype=int)
    vote_share = source["vote_share"].to_numpy(dtype=float)
    vote_share_strip = vote_share.copy()
    vote_share_strip[coverage == 0] = np.nan
    vote_share_strip = vote_share_strip[:, None]
    panel_selected = source["custom_id"].isin(panel_ids).to_numpy(dtype=float)[:, None]

    pred_cmap = ListedColormap([BLUE, RED])
    pred_cmap.set_bad(MISSING)
    pred_norm = BoundaryNorm([-0.5, 0.5, 1.5], pred_cmap.N)

    share_cmap = LinearSegmentedColormap.from_list("vote_share", [BLUE, "#f7f7f7", RED])
    share_cmap.set_bad(MISSING)
    share_norm = Normalize(vmin=0.0, vmax=1.0)

    panel_cmap = ListedColormap(["#efefef", "#59a14f"])
    panel_norm = BoundaryNorm([-0.5, 0.5, 1.5], panel_cmap.N)

    fig = plt.figure(figsize=(8.0, 9.5))
    grid = fig.add_gridspec(1, 4, width_ratios=[0.34, 0.34, 0.34, 5.0], wspace=0.06)
    ax_truth = fig.add_subplot(grid[0, 0])
    ax_share = fig.add_subplot(grid[0, 1], sharey=ax_truth)
    ax_panel = fig.add_subplot(grid[0, 2], sharey=ax_truth)
    ax = fig.add_subplot(grid[0, 3], sharey=ax_truth)

    ax_truth.imshow(truth, aspect="auto", interpolation="nearest", cmap=pred_cmap, norm=pred_norm)
    share_im = ax_share.imshow(vote_share_strip, aspect="auto", interpolation="nearest", cmap=share_cmap, norm=share_norm)
    ax_panel.imshow(panel_selected, aspect="auto", interpolation="nearest", cmap=panel_cmap, norm=panel_norm)
    ax.imshow(preds, aspect="auto", interpolation="nearest", cmap=pred_cmap, norm=pred_norm)

    ax_truth.set_xticks([0], ["truth"])
    ax_truth.tick_params(axis="x", labelrotation=90)
    ax_share.set_xticks([0], [r"$\hat p_i$"])
    ax_share.tick_params(axis="x", labelrotation=90)
    ax_panel.set_xticks([0], ["panel"])
    ax_panel.tick_params(axis="x", labelrotation=90)
    ax.set_xticks(np.arange(len(model_cols)), model_cols, rotation=35, ha="right")

    ticks = [0, 299, 599, 899, 1198]
    ax_truth.set_yticks(ticks, [str(value + 1) for value in ticks])
    ax.set_yticks(ticks)
    ax.set_yticklabels([])
    ax_share.set_yticks(ticks)
    ax_share.set_yticklabels([])
    ax_panel.set_yticks(ticks)
    ax_panel.set_yticklabels([])
    ax_truth.set_ylabel("Benchmark posts (ordered)")
    ax.set_title("Real-runs disagreement matrix on the frozen 1199-item benchmark", fontsize=10, pad=16)
    ax.tick_params(axis="x", labelsize=8)

    for axis in (ax_truth, ax_share, ax_panel, ax):
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=0)

    band_start = summary["band_start_row"]
    band_end = summary["band_end_row"]
    if band_start is not None and band_end is not None:
        y_center = (band_start + band_end) / 2
        box = Rectangle(
            xy=(-0.48, band_start - 0.48),
            width=len(model_cols) - 0.04,
            height=band_end - band_start + 0.96,
            fill=False,
            edgecolor=INK,
            linewidth=1.4,
        )
        ax.add_patch(box)
        ax.annotate(
            "disagreement band",
            xy=(len(model_cols) - 0.55, y_center),
            xytext=(18, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK,
            arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.8},
            annotation_clip=False,
        )

    handles = [
        Patch(facecolor=BLUE, label="predicted not relevant"),
        Patch(facecolor=RED, label="predicted relevant"),
        Patch(facecolor=MISSING, label="missing prediction"),
        Patch(facecolor="#59a14f", label="selected panel post"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.985))
    cbar = fig.colorbar(share_im, ax=[ax_share, ax_panel, ax], fraction=0.03, pad=0.02)
    cbar.set_label(r"$\hat p_i$")
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    fig.text(
        0.5,
        0.03,
        (
            f"{summary['selected_model_count']} models; "
            f"{summary['fully_observed_rows']} complete rows; "
            f"{summary['partially_observed_rows']} partial rows; "
            f"{summary['fully_missing_rows']} fully missing rows; "
            f"{int(panel_selected.sum())} selected panel rows"
        ),
        ha="center",
        fontsize=8,
        color=INK,
    )
    fig.subplots_adjust(top=0.9, bottom=0.17, left=0.11, right=0.90)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
