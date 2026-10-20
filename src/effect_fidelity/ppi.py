"""'How many real people do you still need?' — prediction-powered inference (PPI++) decision curves.

For each experiment, simulator and real sample size n per arm, we repeatedly draw n real respondents per arm and compare
  (a) real-only: difference in means, normal 95% CI;
  (b) sim-only: the simulator's own A/B estimate (150 simulated respondents per arm; does not use real data);
  (c) PPI++: real data plus simulator predictions on the remaining (unlabeled) respondents in the pool.
Scored against the full-sample real effect: CI coverage, wrong-decision rate, CI width, effective sample size.

PPI++ for one arm (Angelopoulos, Duchi, Zrnic 2023), labeled (y_i, f_i) i=1..n, unlabeled f_j j=1..N:
  theta = mean(y) + lam * (mean(f_unlab) - mean(f_lab)),
  var   = Var(y - lam f)/n + lam^2 Var(f)/N,  lam* = Cov(y, f) / ((1 + n/N) Var(f)), clipped to [0, 1].
Arms are independent randomized samples, so the effect is theta_B - theta_A with summed variances.
"""
from __future__ import annotations
import json, zlib
import numpy as np
import pandas as pd
from .simulate import ROOT

Z = 1.959963984540054
NS = (25, 50, 100, 200, 500)
PRED_DIR = ROOT / "results" / "ppi_preds"


def ppi_mean(y, f_lab, f_unlab, clip=True):
    y, f_lab, f_unlab = map(np.asarray, (y, f_lab, f_unlab))
    n, N = len(y), len(f_unlab)
    vf = np.var(np.concatenate([f_lab, f_unlab]), ddof=1)
    if vf <= 1e-12:
        lam = 0.0
    else:
        lam = np.cov(y, f_lab, ddof=1)[0, 1] / ((1 + n / N) * vf)
        if clip:
            lam = float(np.clip(lam, 0.0, 1.0))
    theta = y.mean() + lam * (f_unlab.mean() - f_lab.mean())
    var = np.var(y - lam * f_lab, ddof=1) / n + lam ** 2 * np.var(f_unlab, ddof=1) / N
    return float(theta), float(var), float(lam)


def diff_ci(est, var):
    se = np.sqrt(var)
    return est, est - Z * se, est + Z * se


def call(est, lo, hi):
    return int(np.sign(est)) if (lo > 0 or hi < 0) else 0


def full_truth(exp):
    a, b = exp.real["A"] * exp.scale, exp.real["B"] * exp.scale
    est = b.mean() - a.mean()
    var = a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)
    return diff_ci(est, var)


def sim_only(sim_res):
    a, b = np.asarray(sim_res["A"]["draws"]), np.asarray(sim_res["B"]["draws"])
    return diff_ci(b.mean() - a.mean(), a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


def run_curves(exps, preds: dict, sims: dict, ns=NS, reps=500, seed=0) -> pd.DataFrame:
    """preds: {sim_name: {exp_id: {arm: {y, f}}}}; sims: {sim_name: sim results json} (for sim-only).
    Returns one row per (simulator, experiment, n) with metrics averaged over `reps` draws."""
    rows = []
    for e in exps:
        t, tlo, thi = full_truth(e)
        tcall = call(t, tlo, thi)
        for s, P in preds.items():
            if e.id not in P:
                continue
            d = P[e.id]
            YA, FA, YB, FB = (np.asarray(d[a][k]) for a in "AB" for k in "yf")
            rho = float(np.nanmean([np.corrcoef(Y, F)[0, 1] if np.std(F) > 0 else 0.0 for Y, F in ((YA, FA), (YB, FB))]))
            if s in sims and e.id in sims[s]:
                so, slo, shi = sim_only(sims[s][e.id])
            else:
                so = slo = shi = np.nan
            for n in ns:
                nn = min(n, len(YA) // 2, len(YB) // 2)  # keep >= half the pool unlabeled
                rng = np.random.default_rng([seed, n, zlib.crc32(e.id.encode()), zlib.crc32(s.encode())])
                acc = {k: [] for k in ("r_cov", "r_wrong", "r_w", "p_cov", "p_wrong", "p_w", "lamA", "lamB")}
                for _ in range(reps):
                    out = {}
                    for arm, Y, F in (("A", YA, FA), ("B", YB, FB)):
                        perm = rng.permutation(len(Y))
                        L, U = perm[:nn], perm[nn:]
                        out[arm] = (Y[L].mean(), Y[L].var(ddof=1) / nn, *ppi_mean(Y[L], F[L], F[U]))
                    r = diff_ci(out["B"][0] - out["A"][0], out["B"][1] + out["A"][1])
                    p = diff_ci(out["B"][2] - out["A"][2], out["B"][3] + out["A"][3])
                    for pre, (est, lo, hi) in (("r", r), ("p", p)):
                        acc[f"{pre}_cov"].append(lo <= t <= hi)
                        acc[f"{pre}_wrong"].append(call(est, lo, hi) != tcall)
                        acc[f"{pre}_w"].append(hi - lo)
                    acc["lamA"].append(out["A"][4]); acc["lamB"].append(out["B"][4])
                rw, pw = np.mean(acc["r_w"]), np.mean(acc["p_w"])
                rows.append(dict(simulator=s, exp=e.id, name=e.name, family=e.family, n=n, n_used=nn, true=t, true_call=tcall,
                                 real_cov=np.mean(acc["r_cov"]), real_wrong=np.mean(acc["r_wrong"]), real_width=rw,
                                 ppi_cov=np.mean(acc["p_cov"]), ppi_wrong=np.mean(acc["p_wrong"]), ppi_width=pw,
                                 sim_est=so, sim_cov=float(slo <= t <= shi) if np.isfinite(so) else np.nan,
                                 sim_wrong=float(call(so, slo, shi) != tcall) if np.isfinite(so) else np.nan,
                                 n_eff=nn * (rw / pw) ** 2, lam=float(np.mean(acc["lamA"] + acc["lamB"])), rho=rho))
    return pd.DataFrame(rows)


def drop_capped(df: pd.DataFrame) -> pd.DataFrame:
    """Drop (simulator, n) rows whose real pool was too small to actually use n labeled people (e.g. a quick run with a
    40-person pool can only be scored at n=20); otherwise the same capped result would be reported under larger n."""
    if df.empty:
        return df
    med = df.groupby(["simulator", "n"]).n_used.transform("median")
    return df[med >= 0.5 * df.n].reset_index(drop=True)


def load_preds() -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted(PRED_DIR.glob("*.json"))} if PRED_DIR.exists() else {}


def summarize(df: pd.DataFrame) -> dict:
    out = {}
    for s, g in df.groupby("simulator"):
        by_n = {}
        for n, h in g.groupby("n"):
            by_n[int(n)] = dict(real_wrong=h.real_wrong.mean(), ppi_wrong=h.ppi_wrong.mean(), real_cov=h.real_cov.mean(),
                                ppi_cov=h.ppi_cov.mean(), width_ratio=(h.ppi_width / h.real_width).median(),
                                n_eff_median=h.n_eff.median(), gain_median=(h.n_eff / h.n_used).median(),
                                gain_max=(h.n_eff / h.n_used).max(), n_used_median=h.n_used.median())
        first = g.drop_duplicates("exp")
        out[s] = dict(rho_median=first.rho.median(), rho_max=first.rho.max(), by_n=by_n, sim_only_wrong=first.sim_wrong.mean(), sim_only_cov=first.sim_cov.mean(), n_exp=int(first.shape[0]))
    return out
