"""Effect-fidelity metrics: does a simulator reproduce the *size* of real treatment effects?"""
from __future__ import annotations
import numpy as np
import pandas as pd

Z = 1.959964


def boot_diff(a: np.ndarray, b: np.ndarray, B: int = 2000, seed: int = 0) -> tuple[float, float, float, float]:
    """Difference in means b - a with a percentile bootstrap 95% CI and bootstrap SE."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    ma = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(B)]) if len(a) < 5000 else \
        rng.normal(a.mean(), a.std(ddof=1) / np.sqrt(len(a)), B)  # large-n: normal bootstrap of the mean
    mb = np.array([b[rng.integers(0, len(b), len(b))].mean() for _ in range(B)]) if len(b) < 5000 else \
        rng.normal(b.mean(), b.std(ddof=1) / np.sqrt(len(b)), B)
    d = mb - ma
    return float(b.mean() - a.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(d.std(ddof=1))


def tvd(real, sim) -> float:
    """Total variation distance between answer distributions (values rounded to the nearest scale point)."""
    r, m = np.round(np.asarray(real, float)), np.round(np.asarray(sim, float))
    cats = np.union1d(r, m)
    pr = np.array([(r == c).mean() for c in cats]); pm = np.array([(m == c).mean() for c in cats])
    return float(0.5 * np.abs(pr - pm).sum())


def pooled_sd(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)))


def call(est, lo, hi) -> int:
    """Decision a team would make: +1 ship B, -1 keep A, 0 no detectable difference."""
    return 1 if lo > 0 else (-1 if hi < 0 else 0)


def experiment_table(exps, sims: dict) -> pd.DataFrame:
    """One row per (simulator, experiment). Effects in native units and in real-SD units (d)."""
    rows = []
    truth = {}
    for e in exps:
        a, b = e.real["A"] * e.scale, e.real["B"] * e.scale
        t, tlo, thi, tse = boot_diff(a, b, seed=1)
        truth[e.id] = (t, tlo, thi, tse, pooled_sd(a, b))
    for sname, res in sims.items():
        for e in exps:
            if e.id not in res:
                continue
            t, tlo, thi, tse, sd = truth[e.id]
            a = np.asarray(res[e.id]["A"]["draws"]) * e.scale
            b = np.asarray(res[e.id]["B"]["draws"]) * e.scale
            s, slo, shi, sse = boot_diff(a, b, seed=2)
            tA = tvd(e.real["A"], res[e.id]["A"]["draws"]); tB = tvd(e.real["B"], res[e.id]["B"]["draws"])
            rows.append(dict(simulator=sname, exp=e.id, name=e.name, family=e.family, unit=e.unit,
                             true=t, true_lo=tlo, true_hi=thi, true_se=tse, sd=sd,
                             sim=s, sim_lo=slo, sim_hi=shi, sim_se=sse, sim_A=a.mean(), sim_B=b.mean(),
                             true_A=e.real["A"].mean() * e.scale, true_B=e.real["B"].mean() * e.scale,
                             n_sim=len(a), tvd_A=tA, tvd_B=tB, tvd_max=max(tA, tB)))
    df = pd.DataFrame(rows)
    df["d_true"], df["d_sim"] = df.true / df.sd, df.sim / df.sd
    df["d_true_se"], df["d_sim_se"] = df.true_se / df.sd, df.sim_se / df.sd
    df["true_call"] = [call(*r) for r in df[["true", "true_lo", "true_hi"]].to_numpy()]
    df["sim_call"] = [call(*r) for r in df[["sim", "sim_lo", "sim_hi"]].to_numpy()]
    df["covered"] = (df.sim_lo <= df.true) & (df.true <= df.sim_hi)
    df["ratio"] = np.where(df.true_call != 0, df.sim / df.true, np.nan)
    return df


def summarize(df: pd.DataFrame) -> dict:
    sig = df[df.true_call != 0]
    nul = df[df.true_call == 0]
    x, y = df.d_true.to_numpy(), df.d_sim.to_numpy()
    slope = float(np.polyfit(x, y, 1)[0]) if len(df) > 2 else np.nan
    r = float(np.corrcoef(x, y)[0, 1]) if len(df) > 2 and np.std(y) > 0 else np.nan
    rho = float(pd.Series(x).corr(pd.Series(y), method="spearman")) if len(df) > 2 else np.nan
    absratio = np.abs(sig.ratio.to_numpy())
    good = df[df.tvd_max < 0.15]
    return dict(
        mean_arm_tvd=float(df[["tvd_A", "tvd_B"]].to_numpy().mean()),
        share_arms_tvd_below_015=float((df[["tvd_A", "tvd_B"]].to_numpy() < 0.15).mean()),
        n_exp_both_arms_tvd_below_015=int(len(good)),
        wrong_call_rate_when_both_arms_good=float((good.sim_call != good.true_call).mean()) if len(good) else np.nan,
        abs_effect_error_d_when_both_arms_good=float(np.mean(np.abs(good.d_sim - good.d_true))) if len(good) else np.nan,
        corr_arm_tvd_vs_effect_error=float(np.corrcoef(df.tvd_max, np.abs(df.d_sim - df.d_true))[0, 1]) if len(df) > 2 else np.nan,
        n_experiments=int(len(df)), n_true_nonnull=int(len(sig)), n_true_null=int(len(nul)),
        sign_agreement=float((np.sign(sig.sim) == np.sign(sig.true)).mean()) if len(sig) else np.nan,
        median_effect_ratio=float(np.median(sig.ratio)) if len(sig) else np.nan,
        median_abs_ratio=float(np.median(absratio)) if len(sig) else np.nan,
        share_inflated_2x=float((sig.ratio > 2).mean()) if len(sig) else np.nan,
        share_shrunk_half=float((sig.ratio.abs() < 0.5).mean()) if len(sig) else np.nan,
        calib_slope=slope, pearson_r=r, spearman_rho=rho,
        mae_d=float(np.mean(np.abs(y - x))),
        ci_coverage=float(df.covered.mean()),
        wrong_call_rate=float((df.sim_call != df.true_call).mean()),
        wrong_winner_rate=float(((sig.sim_call != 0) & (sig.sim_call != sig.true_call)).mean()) if len(sig) else np.nan,
        false_alarm_rate_on_nulls=float((nul.sim_call != 0).mean()) if len(nul) else np.nan,
    )


def loeo_correction(df: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-experiment-out recalibration, in real-SD (d) units.

    For each held-out experiment j, using only the other experiments:
      * scale c  = argmin sum (d_true - c * d_sim)^2   (no intercept: a zero simulated effect maps to zero)
      * tau^2    = excess residual variance beyond the two sampling errors (floored at 0)
    Corrected estimate: c * d_sim_j, 95% CI: +/- 1.96 * sqrt(c^2 se_sim^2 + tau^2).
    tau is the "model error" term that a simulator's own bootstrap CI omits.
    """
    out = []
    for j in range(len(df)):
        tr = df.drop(df.index[j])
        xs, yt = tr.d_sim.to_numpy(), tr.d_true.to_numpy()
        c = float((xs * yt).sum() / max((xs ** 2).sum(), 1e-12))
        resid = yt - c * xs
        tau2 = max(float(np.mean(resid ** 2 - (c * tr.d_sim_se) ** 2 - tr.d_true_se ** 2)), 0.0)
        row = df.iloc[j]
        est = c * row.d_sim
        half = Z * np.sqrt((c * row.d_sim_se) ** 2 + tau2)
        lo, hi = est - half, est + half
        out.append(dict(exp=row.exp, c=c, tau=np.sqrt(tau2), d_corr=est, d_corr_lo=lo, d_corr_hi=hi,
                        covered_corr=bool(lo <= row.d_true <= hi), corr_call=call(est, lo, hi)))
    o = pd.DataFrame(out, index=df.index)
    return pd.concat([df, o.drop(columns="exp")], axis=1)


def summarize_corrected(dc: pd.DataFrame) -> dict:
    return dict(
        mae_d_raw=float(np.mean(np.abs(dc.d_sim - dc.d_true))),
        mae_d_corrected=float(np.mean(np.abs(dc.d_corr - dc.d_true))),
        coverage_raw=float(dc.covered.mean()),
        coverage_corrected=float(dc.covered_corr.mean()),
        wrong_call_raw=float((dc.sim_call != dc.true_call).mean()),
        wrong_call_corrected=float((dc.corr_call != dc.true_call).mean()),
        median_scale_c=float(dc.c.median()),
        median_tau_d=float(dc.tau.median()),
    )
