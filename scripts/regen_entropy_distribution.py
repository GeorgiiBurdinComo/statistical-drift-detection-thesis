#!/usr/bin/env python3
"""Regenerate the realised binary-entropy distribution over benchmark vote shares."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets" / "metrics_export" / "canonical_disagreement_matrix_source.csv"
OUT = ROOT / "assets" / "evidence" / "fig_entropy_distribution.pdf"

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    }
)


def binary_entropy(p: np.ndarray) -> np.ndarray:
    out = np.zeros_like(p, dtype=float)
    mask = (p > 0.0) & (p < 1.0)
    out[mask] = -p[mask] * np.log(p[mask]) - (1.0 - p[mask]) * np.log(1.0 - p[mask])
    return out


def main() -> None:
    df = pd.read_csv(SOURCE)
    observed = df.loc[df["coverage"].gt(0)].copy()
    observed["entropy"] = binary_entropy(observed["vote_share"].to_numpy(dtype=float))

    entropy = observed["entropy"].to_numpy()
    zero_count = int(np.isclose(entropy, 0.0).sum())
    mixed_count = int((entropy > 0.0).sum())
    max_entropy = float(np.log(2.0))

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    bins = np.linspace(0.0, max_entropy, 17)
    ax.hist(entropy, bins=bins, color="#4e79a7", edgecolor="white", linewidth=0.8)
    ax.axvline(max_entropy, color="#f28e2b", lw=1.2, ls="--")
    ax.set_xlabel(r"Binary entropy $H_b(\hat p_i)$")
    ax.set_ylabel("Benchmark posts")
    ax.set_title("Realised distribution of the entropy term")
    ax.text(
        max_entropy - 0.01,
        ax.get_ylim()[1] * 0.94,
        r"$H_b(0.5)=\log 2$",
        ha="right",
        va="top",
        fontsize=8,
        color="#f28e2b",
    )
    ax.grid(axis="y", color="0.9", lw=0.8)
    fig.text(
        0.5,
        0.01,
        f"{len(observed)} observed rows; {zero_count} zero-entropy rows; {mixed_count} mixed-vote rows",
        ha="center",
        fontsize=8,
        color="#333333",
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=220)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
