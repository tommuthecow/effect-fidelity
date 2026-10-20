import numpy as np
from effect_fidelity.ppi import ppi_mean, run_curves, call


def test_constant_prediction_reduces_to_real_only():
    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    th, var, lam = ppi_mean(y, np.ones(50), np.ones(500))
    assert lam == 0 and np.isclose(th, y.mean()) and np.isclose(var, y.var(ddof=1) / 50)


def test_matches_ppi_py_point_estimate():
    ppi_py = __import__("ppi_py")
    rng = np.random.default_rng(1)
    x = rng.normal(size=2000); y = x + rng.normal(size=2000)
    th, _, lam = ppi_mean(y[:100], x[:100], x[100:], clip=False)
    ref = ppi_py.ppi_mean_pointestimate(y[:100], x[:100], x[100:], lam=lam)
    assert np.isclose(th, float(np.ravel(ref)[0]))


def test_ppi_coverage_and_gain_on_synthetic():
    rng = np.random.default_rng(2)
    cover, wr, wp = [], [], []
    for _ in range(400):
        x = rng.normal(size=1000); y = 0.3 + 2 * x + rng.normal(size=1000)
        f = x + 5  # biased but correlated predictor: PPI must stay valid
        th, var, _ = ppi_mean(y[:50], f[:50], f[50:])
        cover.append(abs(th - 0.3) <= 1.96 * np.sqrt(var) + 1e-12)
        wp.append(np.sqrt(var)); wr.append(y[:50].std(ddof=1) / np.sqrt(50))
    assert 0.91 <= np.mean(cover) <= 0.99
    assert np.mean(wp) < 0.7 * np.mean(wr)


def test_curves_shape_on_fake_experiment():
    class E:  # minimal experiment stand-in
        id, name, family, scale = "fake", "fake", "fake", 1.0
        rng = np.random.default_rng(3)
        real = {"A": rng.normal(0, 1, 800), "B": rng.normal(0.3, 1, 800)}
    preds = {"sim": {"fake": {a: {"y": E.real[a].tolist(), "f": (E.real[a] + np.random.default_rng(4).normal(0, 1, 800)).tolist()} for a in "AB"}}}
    df = run_curves([E], preds, {}, ns=(25, 100), reps=50)
    assert len(df) == 2 and (df.ppi_width < df.real_width).all() and (df.n_eff > df.n_used).all()
    assert call(1.0, 0.5, 1.5) == 1 and call(-1.0, -2, 0.1) == 0


def test_real_covariates_align_with_outcomes():
    from effect_fidelity.experiments import load_all
    for e in load_all():
        for a in "AB":
            assert len(e.real_cov[a]) == len(e.real[a])


def test_ci_matches_ppi_py_with_tuned_lambda():
    ppi_py = __import__("ppi_py")
    rng = np.random.default_rng(5)
    x = rng.normal(size=3000); y = 0.5 * x + rng.normal(size=3000)
    th, var, lam = ppi_mean(y[:200], x[:200], x[200:], clip=False)
    lo, hi = ppi_py.ppi_mean_ci(y[:200], x[:200], x[200:], alpha=0.05)
    assert np.isclose(th, (float(np.ravel(lo)[0]) + float(np.ravel(hi)[0])) / 2, atol=0.02)
    assert np.isclose(2 * 1.96 * np.sqrt(var), float(np.ravel(hi)[0]) - float(np.ravel(lo)[0]), rtol=0.05)


def test_finetune_overlap_screen():
    from effect_fidelity.finetune import is_excluded
    assert is_excluded("Imagine the US is preparing for an unusual disease expected to kill 600 people")
    assert is_excluded("...spending too little on welfare?")
    assert not is_excluded("How much did the video make you feel angry?")


def test_apd_scoring_on_fake_table():
    import pandas as pd
    from effect_fidelity.apd import score
    rng = np.random.default_rng(0)
    n = 10
    d_true = rng.normal(0, .3, n); d_sim = 0.5 * d_true
    df = pd.DataFrame(dict(exp=[f"e{i}" for i in range(n)], d_true=d_true, d_sim=d_sim, d_sim_se=.05, d_true_se=.02,
                           true_call=np.sign(d_true).astype(int), sim_call=np.sign(d_sim).astype(int), covered=False))
    s = score(df); c = score(df, corrected=True)
    assert s["wrong_rate"] == 0 and c["mae_d"] < s["mae_d"]
