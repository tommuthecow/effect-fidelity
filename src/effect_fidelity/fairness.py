"""Audit metrics that don't depend on how many simulated people a run used, plus trivial baselines.

Why: a simulator's 'call' comes from its own 95% interval, so its width depends on the number of simulated respondents
(150 for the small models, 25 for the quick Claude run). With few simulated people a simulator says 'no difference' more
often; with many it almost always picks a side. So we also score every simulator at a matched 25 simulated people per arm,
and report n-free measures (direction agreement, effect correlation) and what trivial strategies would score.
"""
from __future__ import annotations
import numpy as np
from .metrics import call

N_MATCH = 25


def _ci(a, b):
    d = b.mean() - a.mean(); se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return d, d - 1.96 * se, d + 1.96 * se


def matched(df, sims, exps, n=N_MATCH, reps=400, seed=0):
    """Wrong-call rate and CI coverage with exactly n simulated respondents per arm (subsampled without replacement)."""
    E = {e.id: e for e in exps}; out = {}
    for s, g in df.groupby("simulator"):
        rng = np.random.default_rng(seed); wrong, cov = [], []
        for _, r in g.iterrows():
            e = E[r.exp]; A = np.asarray(sims[s][r.exp]["A"]["draws"]) * e.scale; B = np.asarray(sims[s][r.exp]["B"]["draws"]) * e.scale
            w = c = 0
            for _ in range(reps):
                a = rng.choice(A, min(n, len(A)), replace=False); b = rng.choice(B, min(n, len(B)), replace=False)
                d, lo, hi = _ci(a, b); w += call(d, lo, hi) != r.true_call; c += lo <= r.true <= hi
            wrong.append(w / reps); cov.append(c / reps)
        nn = g[g.true_call != 0]
        out[s] = dict(n_sim=int(g.n_sim.median()), wrong_call_matched=float(np.mean(wrong)), coverage_matched=float(np.mean(cov)),
                      direction_agreement=float(np.mean(np.sign(nn.sim) == np.sign(nn.true))),
                      effect_corr=float(np.corrcoef(g.d_true, g.d_sim)[0, 1]))
    return out


def baselines(df, seed=0):
    t = df.drop_duplicates("exp"); tc = t.true_call.to_numpy(); rng = np.random.default_rng(seed)
    return dict(n_exp=int(len(tc)), n_real_positive=int((tc == 1).sum()), n_real_negative=int((tc == -1).sum()), n_real_null=int((tc == 0).sum()),
                always_no_difference=float((tc != 0).mean()), always_ship_b=float((tc != 1).mean()),
                random_three_way=float(np.mean([(rng.choice([-1, 0, 1], len(tc)) != tc).mean() for _ in range(20000)])),
                random_direction=float(np.mean([(rng.choice([-1, 1], len(tc)) != tc).mean() for _ in range(20000)])))
