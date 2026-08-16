#!/usr/bin/env python3
"""Regenerate the realised distribution of relevant-vote shares."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets" / "metrics_export" / "canonical_disagreement_matrix_source.csv"
OUT = ROOT / "assets" / "evidence" / "fig_vote_share_distribution.pdf"

BLUE = "#4e79a7"
RED = "#f28e2b"
INK = "#333333"


def format_share(value: float) -> str:
    frac = Fraction(value).limit_denominator(8)
    if frac.denominator == 1:
        return str(frac.numerator)
    return rf"$\frac{{{frac.numerator}}}{{{frac.denominator}}}$"


def main() -> None:
    df = pd.read_csv(SOURCE)
    observed = df.loc[df["coverage"].gt(0)].copy()
    counts = observed["vote_share"].value_counts().sort_index()

    x = range(len(counts))
    colors = [BLUE if value < 0.5 else RED if value > 0.5 else "#9c755f" for value in counts.index]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.bar(x, counts.to_numpy(), color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(list(x), [format_share(value) for value in counts.index], rotation=0)
    ax.set_xlabel(r"Relevant-vote share $\hat p_i$")
    ax.set_ylabel("Benchmark posts")
    ax.set_title(r"Realised distribution of benchmark posts by $\hat p_i$")
    ax.grid(axis="y", color="0.9", lw=0.8)

    for xpos, count in zip(x, counts.to_numpy()):
        ax.text(xpos, count + 6, str(int(count)), ha="center", va="bottom", fontsize=7, color=INK)

    fig.text(
        0.5,
        0.01,
        "Blue bars are below 0.5, red bars are above 0.5, and the central bars correspond to the most even vote splits.",
        ha="center",
        fontsize=8,
        color=INK,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
