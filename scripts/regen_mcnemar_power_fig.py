#!/usr/bin/env python3
"""Regenerate fig_mcnemar_power.pdf with embedded TrueType fonts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "assets" / "metrics_export" / "mcnemar_power.json"
OUT = ROOT / "assets" / "evidence" / "fig_mcnemar_power.pdf"

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    }
)


def main() -> None:
    payload = json.loads(JSON_PATH.read_text())
    grid = payload["power_grid_n300"]
    # Expect list of {rho, delta, power, feasible}
    rhos = sorted({row["rho"] for row in grid})
    deltas = sorted({row["delta"] for row in grid})
    power = np.full((len(rhos), len(deltas)), np.nan)
    rho_i = {r: i for i, r in enumerate(rhos)}
    delta_j = {d: j for j, d in enumerate(deltas)}
    for row in grid:
        i, j = rho_i[row["rho"]], delta_j[row["delta"]]
        if row.get("feasible", True) and row["delta"] <= row["rho"] + 1e-12:
            power[i, j] = row["power"]

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    mesh = ax.imshow(
        power,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        extent=[
            deltas[0] - 0.5 * (deltas[1] - deltas[0] if len(deltas) > 1 else 0.01),
            deltas[-1] + 0.5 * (deltas[1] - deltas[0] if len(deltas) > 1 else 0.01),
            rhos[0] - 0.5 * (rhos[1] - rhos[0] if len(rhos) > 1 else 0.01),
            rhos[-1] + 0.5 * (rhos[1] - rhos[0] if len(rhos) > 1 else 0.01),
        ],
    )
    # annotate sparse readable cells
    for i, rho in enumerate(rhos):
        for j, delta in enumerate(deltas):
            val = power[i, j]
            if np.isnan(val):
                ax.text(delta, rho, "—", ha="center", va="center", fontsize=7, color="0.45")
            else:
                ax.text(
                    delta,
                    rho,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=("white" if val < 0.45 else "black"),
                )
    ax.set_xticks(deltas)
    ax.set_yticks(rhos)
    ax.set_xlabel(r"Accuracy drop $\delta$")
    ax.set_ylabel(r"Discordance $\rho$")
    ax.set_title(r"Exact McNemar power ($n=300$, $\alpha=0.05$; blank if $\delta>\rho$)")
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Power")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=220)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
