import numpy as np
import pandas as pd
from effect_fidelity.metrics import boot_diff, call, summarize, loeo_correction, summarize_corrected


def synth(scale, noise=0.0, n=17, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(-1, 1, n)
    s = scale * t + rng.normal(0, noise, n)
    se = np.full(n, 0.05)
    df = pd.DataFrame(dict(exp=[f"e{i}" for i in range(n)], d_true=t, d_sim=s, d_true_se=se, d_sim_se=se, sd=1.0,
                           true=t, true_lo=t - 0.1, true_hi=t + 0.1, sim=s, sim_lo=s - 0.1, sim_hi=s + 0.1))
    df["true_call"] = [call(*r) for r in df[["true", "true_lo", "true_hi"]].to_numpy()]
    df["sim_call"] = [call(*r) for r in df[["sim", "sim_lo", "sim_hi"]].to_numpy()]
    df["covered"] = (df.sim_lo <= df.true) & (df.true <= df.sim_hi)
    df["ratio"] = np.where(df.true_call != 0, df.sim / df.true, np.nan)
    df["tvd_A"] = df["tvd_B"] = df["tvd_max"] = 0.05
    return df


def test_boot_diff_basic():
    a, b = np.zeros(400), np.ones(400) * 0.5
    est, lo, hi, se = boot_diff(a, b)
    assert est == 0.5 and lo <= 0.5 <= hi


def test_call():
    assert call(1, 0.2, 2) == 1 and call(-1, -2, -0.1) == -1 and call(0.1, -0.1, 0.3) == 0


def test_perfect_simulator():
    s = summarize(synth(1.0))
    assert s["calib_slope"] == 1.0 or abs(s["calib_slope"] - 1) < 1e-9
    assert s["ci_coverage"] == 1.0 and s["wrong_call_rate"] == 0 and s["sign_agreement"] == 1


def test_inflated_simulator_detected_and_corrected():
    df = synth(2.0, noise=0.0)
    s = summarize(df)
    assert abs(s["calib_slope"] - 2) < 1e-9 and s["median_effect_ratio"] == 2.0
    assert s["ci_coverage"] < 0.3
    c = summarize_corrected(loeo_correction(df))
    assert abs(c["median_scale_c"] - 0.5) < 1e-6
    assert c["coverage_corrected"] > 0.9 and c["mae_d_corrected"] < 1e-6


def test_correction_never_sees_heldout():
    df = synth(2.0, noise=0.3, seed=3)
    dc = loeo_correction(df)
    # change one experiment's truth: only the *other* folds' scale factors may move
    df2 = df.copy(); df2.loc[0, "d_true"] += 5
    dc2 = loeo_correction(df2)
    assert dc.c.iloc[0] == dc2.c.iloc[0]
    assert (dc.c.iloc[1:] != dc2.c.iloc[1:]).all()


def test_tvd():
    from effect_fidelity.metrics import tvd
    assert tvd([0, 1, 1, 1], [0, 0, 1, 1]) == 0.25 and tvd([1, 2], [1, 2]) == 0
