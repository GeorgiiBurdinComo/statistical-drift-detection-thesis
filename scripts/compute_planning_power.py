#!/usr/bin/env python3
"""Exact one-sided McNemar planning power for fixed n, rho, g, and alpha."""

from math import comb

from scipy.stats import binom


def critical_k(d: int, alpha: float) -> int:
    for k in range(d + 1):
        if binom.sf(k - 1, d, 0.5) <= alpha:
            return k
    return d + 1


def planning_power(n: int, rho: float, g: float, alpha: float = 0.05) -> float:
    if not (0.0 < rho <= 1.0):
        raise ValueError("rho must satisfy 0 < rho <= 1")
    if not (0.0 <= g <= rho):
        raise ValueError("g must satisfy 0 <= g <= rho")

    eta = (rho + g) / (2.0 * rho)  # P(regression | discordant)
    total = 0.0

    for d in range(n + 1):
        p_d = comb(n, d) * (rho ** d) * ((1.0 - rho) ** (n - d))
        k = critical_k(d, alpha)
        reject_given_d = 0.0 if k == d + 1 else binom.sf(k - 1, d, eta)
        total += p_d * reject_given_d

    return total


if __name__ == "__main__":
    n = 300
    rho = 137 / 1199
    g = 0.05
    alpha = 0.05

    value = planning_power(n, rho, g, alpha)
    print(f"Power(n={n}, rho={rho:.6f}, g={g:.4f}, alpha={alpha:.4f}) = {value:.6f}")
