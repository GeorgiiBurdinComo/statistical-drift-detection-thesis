#!/usr/bin/env python3
"""Exact/unconditional McNemar power and attainable MDE under 0 <= delta <= rho."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import binom

OUT = Path(__file__).resolve().parents[1] / "assets" / "metrics_export"


@lru_cache(maxsize=None)
def critical_b(m: int, alpha: float = 0.05) -> int | None:
    if m <= 0:
        return None
    # P(B >= k) = sf(k-1); find minimal k with sf(k-1) <= alpha
    cdf = binom.cdf(np.arange(-1, m), m, 0.5)  # cdf(-1)=0 via empty
    # Use survival from k=0..m
    for k in range(0, m + 1):
        p = binom.sf(k - 1, m, 0.5) if k > 0 else 1.0
        if p <= alpha + 1e-15:
            return k
    return None


def _critical_table(n: int, alpha: float = 0.05) -> np.ndarray:
    """crit[m] = k or -1 if impossible."""
    crit = np.full(n + 1, -1, dtype=int)
    for m in range(n + 1):
        k = critical_b(m, alpha)
        if k is not None:
            crit[m] = k
    return crit


def power(n: int, rho: float, delta: float, alpha: float = 0.05, crit: np.ndarray | None = None) -> float:
    if rho <= 0 or delta < 0:
        return 0.0
    if delta > rho + 1e-12:
        raise ValueError(f"delta={delta} > rho={rho}")
    eta = 0.5 if rho == 0 else min(max(0.5 + delta / (2.0 * rho), 0.0), 1.0)
    if crit is None:
        crit = _critical_table(n, alpha)
    m = np.arange(0, n + 1)
    pm = binom.pmf(m, n, rho)
    pow_m = np.zeros(n + 1)
    for mi in m:
        k = crit[mi]
        if k < 0:
            pow_m[mi] = 0.0
        else:
            pow_m[mi] = binom.sf(k - 1, mi, eta) if mi > 0 else 0.0
    return float(np.dot(pm, pow_m))


def attainable_mde(n: int, rho: float, target_power: float = 0.80, alpha: float = 0.05) -> dict:
    crit = _critical_table(n, alpha)
    if rho <= 0:
        return {"rho": rho, "mde": None, "power_at_rho": 0.0, "attainable": False}
    power_at_rho = power(n, rho, rho, alpha, crit)
    if power_at_rho < target_power - 1e-9:
        return {
            "rho": rho,
            "mde": None,
            "power_at_mde": None,
            "power_at_rho": power_at_rho,
            "attainable": False,
        }
    lo, hi = 0.0, rho
    best = rho
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        p = power(n, rho, mid, alpha, crit)
        if p >= target_power:
            best = mid
            hi = mid
        else:
            lo = mid
    return {
        "rho": rho,
        "mde": best,
        "power_at_mde": power(n, rho, best, alpha, crit),
        "power_at_rho": power_at_rho,
        "attainable": True,
    }


# Median discordance on baseline-vs-later pairs (per-model onset t0 vs each later week).
MODEL_PSI = [
    ("gpt-4.1-nano", 0.152, 16),
    ("gpt-5-nano", 0.124, 15),
    ("gpt-5-mini", 0.080, 15),
    ("gpt-4.1-mini", 0.062, 16),
    ("claude-haiku-4-5", 0.034, 14),
    ("claude-sonnet-4-5", 0.030, 14),
    ("gpt-5.1", 0.020, 15),
    ("gemini-2.5-flash", 0.017, 12),
    ("claude-sonnet-4-6", 0.015, 14),
    ("gpt-5.4", 0.013, 13),
    ("gpt-5.2", 0.010, 14),
    ("gpt-5", 0.007, 15),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n, alpha = 300, 0.05
    crit = _critical_table(n, alpha)
    model_rows = []
    print(f"{'model':20s} {'psi':>6} {'MDE80':>8} {'pow@rho':>8} ok")
    for name, psi, comps in MODEL_PSI:
        r = attainable_mde(n, psi, alpha=alpha)
        model_rows.append(
            {
                "model": name,
                "comparisons": comps,
                "median_psi": psi,
                "mde_80": r["mde"],
                "power_at_rho": r["power_at_rho"],
                "attainable": r["attainable"],
            }
        )
        mde_s = f"{r['mde']:.4f}" if r["mde"] is not None else "n/a"
        print(f"{name:20s} {psi:6.3f} {mde_s:>8} {r['power_at_rho']:8.3f} {r['attainable']}")

    # Compact power grid for n=300
    rhos = [0.01, 0.03, 0.05, 0.07, 0.10, 0.15]
    deltas = [0.0, 0.01, 0.03, 0.05, 0.07, 0.10]
    heat = []
    for rho in rhos:
        for delta in deltas:
            if delta > rho:
                heat.append({"n": n, "rho": rho, "delta": delta, "power": None, "feasible": False})
            else:
                heat.append(
                    {
                        "n": n,
                        "rho": rho,
                        "delta": delta,
                        "power": power(n, rho, delta, alpha, crit),
                        "feasible": True,
                    }
                )

    # Type I at delta=0 should be ~<= alpha (unconditional slightly conservative)
    type_i = {rho: power(n, rho, 0.0, alpha, crit) for rho in rhos}

    crit173 = _critical_table(n, alpha / 173)
    payload = {
        "n": n,
        "alpha": alpha,
        "target_power": 0.80,
        "note": "MDE requires 0<=delta<=rho; unattainable when max power at delta=rho < 0.80",
        "models": model_rows,
        "power_grid_n300": heat,
        "empirical_type_I_delta0": type_i,
        "planning_check": {
            "rho": 0.1143,
            "delta": 0.05,
            "power_single_test": power(n, 0.1143, 0.05, alpha, crit),
            "power_bonferroni_173": power(n, 0.1143, 0.05, alpha / 173, crit173),
        },
    }
    out_json = OUT / "mcnemar_power.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print("\nWrote", out_json)
    print("Planning power:", payload["planning_check"])
    print("Type I (delta=0):", type_i)


if __name__ == "__main__":
    main()
