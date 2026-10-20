"""Sanity checks for PPI++ on synthetic data with a known answer."""
import numpy as np
from effect_fidelity.ppi import ppi_mean


def _gain(rho, n=100, N=20000, reps=400, seed=0):
    rng = np.random.default_rng(seed); vr, vp, cover = [], [], 0
    for _ in range(reps):
        z = rng.normal(size=n + N); y = z + 0.0
        f = rho * z + np.sqrt(1 - rho ** 2) * rng.normal(size=n + N)  # corr(y, f) = rho
        th, v, lam = ppi_mean(y[:n], f[:n], f[n:])
        vr.append(np.var(y[:n], ddof=1) / n); vp.append(v); cover += abs(th) <= 1.96 * np.sqrt(v)
    return np.mean(vr) / np.mean(vp), cover / reps


def test_useless_simulator_is_worth_about_one():
    g, cov = _gain(0.0)
    assert 0.9 < g < 1.1 and cov > 0.9


def test_perfect_simulator_is_worth_a_lot():
    g, cov = _gain(0.999)
    assert g > 20 and cov > 0.9


def test_gain_matches_one_over_one_minus_r2():
    g, _ = _gain(0.7)
    assert abs(g - 1 / (1 - 0.49)) < 0.25  # ~1.96x


def test_drop_capped_removes_rows_beyond_the_real_pool():
    import pandas as pd
    from effect_fidelity.ppi import drop_capped
    df = pd.DataFrame(dict(simulator=["s"] * 3 + ["t"] * 3, n=[25, 50, 100] * 2, n_used=[20, 20, 20, 25, 50, 100]))
    out = drop_capped(df)
    assert set(map(tuple, out[["simulator", "n"]].to_numpy())) == {("s", 25), ("t", 25), ("t", 50), ("t", 100)}
