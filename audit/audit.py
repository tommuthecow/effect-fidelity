"""Methodology audit checks. PYTHONPATH=src python audit/audit.py"""
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from effect_fidelity.experiments import load_all
from effect_fidelity.simulate import render
R = Path("results"); np.random.seed(0)
exps = load_all()
df = pd.read_csv(R / "experiment_table.csv")
t = df.drop_duplicates("exp").set_index("exp")
print("== truth calls:", t.true_call.value_counts().to_dict())
tc = t.true_call.to_numpy()
print("== trivial baselines (wrong-call rate):")
print("  always 'no difference':", (tc != 0).mean().round(2))
print("  always 'ship B':      ", (tc != 1).mean().round(2))
print("  uniform random 3-way: ", round(np.mean([(np.random.choice([-1, 0, 1], len(tc)) != tc).mean() for _ in range(5000)]), 2))
print("  random sign (+/-1):   ", round(np.mean([(np.random.choice([-1, 1], len(tc)) != tc).mean() for _ in range(5000)]), 2))
print("== power of a real-only 2-arm test at n per arm, given each true d:")
for n in (25, 100, 500):
    pw = [stats.norm.sf(1.96 - abs(d) * np.sqrt(n / 2)) + stats.norm.cdf(-1.96 - abs(d) * np.sqrt(n / 2)) for d in t.d_true]
    exp_wrong = np.mean([(1 - p) if c != 0 else p for p, c in zip(pw, tc)])
    print(f"  n={n}: expected wrong-call ~{exp_wrong:.2f}")
print("== per simulator: wrong call, sign agreement on real non-null effects, and 'sign-only' wrong call")
for s, g in df.groupby("simulator"):
    nn = g[g.true_call != 0]
    print(f"  {s:20s} n_sim={int(g.n_sim.median()):4d} wrong={np.mean(g.sim_call != g.true_call):.2f} "
          f"sign_agree={np.mean(np.sign(nn.sim) == np.sign(nn.true)):.2f} "
          f"says_no_diff={np.mean(g.sim_call == 0):.2f} pearson_d={np.corrcoef(g.d_true, g.d_sim)[0,1]:.2f}")
print("== arm A vs B prompts differ (first persona, first item):")
for e in exps:
    pa = render(e.items[0], "A", "P")[0]; pb = render(e.items[0], "B", "P")[0]
    print(f"  {e.id:22s} differ={pa != pb}  labels={e.items[0].labels}")
print("== position bias in local caches (mean prob on first label, 2-option prompts):")
for f in Path("data/cache").glob("*.jsonl"):
    ps = [json.loads(l)["p"] for l in f.open()]
    two = [p for p in ps if len(p) == 2]
    if two: print(f"  {f.name:45s} n={len(ps)} first-label mean={np.mean([p[0] for p in two]):.2f} uniform-rows={np.mean([np.allclose(p, 1/len(p)) for p in ps]):.3f}")
print("== Claude results: uniform/degenerate respondents")
for s in ("claude_persona", "claude_blind"):
    d = json.loads((R / "sims" / f"{s}.json").read_text()); d.pop("_meta")
    ex = np.concatenate([np.array(v[a]["expected"]) for v in d.values() for a in "AB"])
    print(f"  {s}: respondents={len(ex)}, distinct expected values={len(np.unique(np.round(ex,4)))}")
