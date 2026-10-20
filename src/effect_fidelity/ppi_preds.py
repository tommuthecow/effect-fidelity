"""Simulator predictions for REAL respondents (covariate-matched pairing), the input to prediction-powered inference.

For every real respondent i in arm a we build the same persona text the persona simulators use, from i's own
pre-treatment covariates (Many Labs: age, sex, politics; GSS: age, sex, race, degree, politics; Hillstrom: purchase
history), show it arm a's stimulus, and record f(x_i, a) = the simulator's *expected* answer (probability-weighted,
so no sampling noise). f never sees i's outcome, so it is a fixed function of covariates and PPI stays valid
however good or bad the simulator is.

Output: results/ppi_preds/<simulator>.json = {exp_id: {arm: {"y": [...], "f": [...]}}}.
Any simulator with a working `option_probs` backend plugs in, including Claude:
  python -m effect_fidelity.ppi_preds --sims qwen15_persona llama32_1b_persona
  python -m effect_fidelity.ppi_preds --claude-cli            # writes results/ppi_preds/claude_persona.json
A blind simulator gives everyone the same prediction, so PPI gains nothing from it by construction (no file needed).
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
from .experiments import load_all, Experiment
from .simulate import Runner, SIMULATORS, SYSTEM, ROOT, render

OUT = ROOT / "results" / "ppi_preds"
POOL_CAP = 600  # real respondents per arm in the PPI pool (random subset; first six Many Labs experiments were run with 1000 before this was cut for CPU time)
MAX_ITEMS = 2  # multi-item scales (flag 8, money 8, contact 4): predictions use the first 2 items as a short form


def pool_indices(exp: Experiment, arm: str, cap: int = POOL_CAP, seed: int = 0) -> np.ndarray:
    ok = np.flatnonzero(exp.real_cov[arm].notna().all(axis=1).to_numpy())
    if len(ok) > cap:
        ok = np.sort(np.random.default_rng(seed).choice(ok, cap, replace=False))
    return ok


def expected_answer(runner: Runner, exp: Experiment, arm: str, persona: str) -> float:
    """Mean over items of the probability-weighted answer; marginalises over a sampled earlier answer if any."""
    vals = []
    for item in exp.items[:MAX_ITEMS]:
        if item.pre is not None and arm in item.pre_arms:
            ptxt, plabs, _ = render(item.pre, arm, persona)
            pp = runner.probs(SYSTEM, ptxt, plabs)
            v = 0.0
            for j, opt in enumerate(item.pre.options):
                txt, labs, vv = render(item, arm, persona, opt)
                p = runner.probs(SYSTEM, txt, labs)
                v += pp[j] * sum(p[k] * vv[l] for k, l in enumerate(labs))
        else:
            txt, labs, vv = render(item, arm, persona)
            p = runner.probs(SYSTEM, txt, labs)
            v = sum(p[k] * vv[l] for k, l in enumerate(labs))
        vals.append(float(v))
    return float(np.mean(vals)) * exp.scale


def predict(runner: Runner, exp: Experiment, cap: int = POOL_CAP) -> dict:
    out = {}
    for arm in ("A", "B"):
        idx = pool_indices(exp, arm, cap)
        cov = exp.real_cov[arm].iloc[idx]
        texts = [exp.persona_text(r) for _, r in cov.iterrows()]
        memo = {}
        f = []
        for t in texts:
            if t not in memo:
                memo[t] = expected_answer(runner, exp, arm, t)
            f.append(memo[t])
        out[arm] = {"y": (exp.real[arm][idx] * exp.scale).tolist(), "f": f}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", nargs="*", default=["qwen15_persona", "llama32_1b_persona"])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--claude-cli", action="store_true", help="use the signed-in Claude Code CLI as the simulator")
    a = ap.parse_args(argv)
    exps = [e for e in load_all() if not a.only or e.id in a.only]
    OUT.mkdir(parents=True, exist_ok=True)
    cfgs = [s for s in SIMULATORS if s.name in a.sims and s.conditioning == "persona" and s.model_file != "claude-api"]
    for cfg in cfgs:
        lm = None
        if a.claude_cli:
            raise SystemExit("use --claude-cli without --sims")
        from .llm import LocalLM
        lm = LocalLM(str(ROOT / "models" / cfg.model_file), cfg.template, n_threads=a.threads)
        _run(Runner(cfg, lm=lm), cfg.name, exps)
    if a.claude_cli:
        from .run_claude import configs
        from . import claude as cl
        lm = cl.ClaudeCLI(); lm.check()
        cfg = configs("claude-cli", lm.preflight(), ["persona"])[0]
        _run(Runner(cfg, lm=lm), "claude_persona", exps)


def _run(r: Runner, name: str, exps):
    path = OUT / f"{name}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    t0 = time.time()
    for e in exps:
        if e.id in res:
            continue
        res[e.id] = predict(r, e)
        path.write_text(json.dumps(res))
        print(f"[{name}] {e.id} done; new calls={r.calls} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
