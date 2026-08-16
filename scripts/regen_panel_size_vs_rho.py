#!/usr/bin/env python3
"""Regenerate the exact minimum panel-size curve as a function of discordance rho."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mcnemar_power import _critical_table, power

ROOT = Path(__file__).resolve().parents[1]
OUT_FIG = ROOT / "assets" / "evidence" / "fig_panel_size_vs_rho.pdf"
OUT_CSV = ROOT / "assets" / "metrics_export" / "panel_size_vs_rho.csv"

G_VALUES = [0.03, 0.05, 0.08]
ALPHA = 0.05
TARGET_POWER = 0.80
PILOT_RHO = 137 / 1199

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    }
)


@lru_cache(maxsize=None)
def crit_table(n: int) -> np.ndarray:
    return _critical_table(n, ALPHA)


def minimum_n(
    rho: float,
    g: float,
    target_power: float = TARGET_POWER,
    start_n: int = 1,
) -> tuple[int, float]:
    if rho < g:
        raise ValueError("Need rho >= g for an admissible design scenario.")

    for n in range(start_n, 3001):
        p = power(n, rho, g, ALPHA, crit_table(n))
        if p >= target_power:
            return n, p
    raise RuntimeError(f"minimum n not found up to 3000 for rho={rho}")


def main() -> None:
    rhos = np.round(np.arange(0.03, 0.201, 0.005), 3)
    rows = []
    curves: dict[float, list[tuple[float, int]]] = {}
    for g in G_VALUES:
        curve = []
        start_n = 1
        for rho in rhos:
            rho_f = float(rho)
            if rho_f < g:
                continue
            n_min, p_at_n_min = minimum_n(rho_f, g, start_n=start_n)
            start_n = n_min
            curve.append((rho_f, n_min))
            rows.append(
                {
                    "rho": rho_f,
                    "g": g,
                    "n_min": n_min,
                    "power_at_n_min": p_at_n_min,
                    "power_at_300": power(300, rho_f, g, ALPHA, crit_table(300)),
                }
            )
        curves[g] = curve

    pilot_n_min, pilot_power = minimum_n(PILOT_RHO, 0.05)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(6.1, 3.9))
    palette = {0.03: "#52514E", 0.05: "#2A78D6", 0.08: "#EB6834"}
    for g in G_VALUES:
        curve = curves[g]
        x = [rho for rho, _ in curve]
        y = [n_min for _, n_min in curve]
        ax.plot(x, y, color=palette[g], lw=2.0, label=rf"$g={int(round(100*g))}$pp")

    ax.axhline(300, color="0.45", lw=1.0, ls="--")
    ax.axvline(PILOT_RHO, color="#2A78D6", lw=1.0, ls="--")
    ax.scatter([PILOT_RHO], [pilot_n_min], color="#2A78D6", s=28, zorder=3)
    ax.scatter([PILOT_RHO], [300], facecolors="white", edgecolors="#2A78D6", s=28, zorder=3)

    ax.annotate(
        rf"pilot $\hat\rho={PILOT_RHO:.3f}$" "\n" rf"exact $n_{{\min}}={pilot_n_min}$",
        xy=(PILOT_RHO, pilot_n_min),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=8,
        color="#52514E",
    )
    ax.set_xlabel(r"Discordance $\rho$")
    ax.set_ylabel(r"Minimum panel size $n_{\min}$")
    ax.set_title(r"Exact minimum panel size for 80\% power at nominal $\alpha=0.05$")
    ax.set_xlim(0.03, 0.20)
    ax.set_ylim(0, 1200)
    ax.grid(axis="y", color="0.88", lw=0.8)
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    fig.savefig(OUT_FIG.with_suffix(".png"), bbox_inches="tight", dpi=220)
    print("Wrote", OUT_FIG)
    print("Wrote", OUT_CSV)
    print(
        f"pilot rho={PILOT_RHO:.6f}, exact n_min={pilot_n_min}, "
        f"power(n=300)={power(300, PILOT_RHO, 0.05, ALPHA, crit_table(300)):.6f}, "
        f"power(n_min)={pilot_power:.6f}"
    )


if __name__ == "__main__":
    main()
