"""Run the 17 experiments with Claude as the simulator (blind + persona), then score with `effect_fidelity.report`.

  python -m effect_fidelity.run_claude --simulator claude-cli     # Claude Code on your laptop, no API key
  python -m effect_fidelity.run_claude --simulator claude-api         # Anthropic API, needs ANTHROPIC_API_KEY

Calls run in parallel batches. Every answer is cached in data/cache/, so a rerun or an interrupted run resumes free.
Prompts that depend on an earlier answer (the reciprocity follow-up) are only known after that answer, so the run
proceeds in a few rounds; the planned count printed up front covers round 1 and is close to the total.
"""
from __future__ import annotations
import argparse, json, re, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .experiments import load_all
from .simulate import Runner, SimulatorConfig, ROOT, Cache
from . import claude as cl
from . import ppi_preds as pp

QUICK_POOL = 40  # --quick: real respondents per arm in the PPI pool (default pool is pp.POOL_CAP)
QUICK_N = 25


class _Collector:
    """Fake LM: records prompts missing from the cache and returns uniform probabilities."""
    def __init__(self):
        self.missing = {}

    def option_probs(self, system, user, labels):
        self.missing[(system, user, tuple(labels))] = None
        return np.full(len(labels), 1 / len(labels))


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or cl.CLI_DEFAULT_LABEL


def display(name: str) -> str:
    """Public label: drop any hosting prefix (e.g. 'org.ns.') and context suffixes like '[1m]'."""
    name = re.sub(r"\[.*?\]$", "", name)
    return name.rsplit(".", 1)[-1] if name.count(".") >= 2 else name


PRETTY = {"claude-opus-5-5": "Claude Opus 5.5", "claude-sonnet-5-5": "Claude Sonnet 5.5", "claude-haiku-5-5": "Claude Haiku 5.5"}


def pretty(model: str) -> str:
    m = display(model)
    return PRETTY.get(m, f"Claude ({m})")


def configs(backend: str, model: str, conds: list[str]) -> list[SimulatorConfig]:
    model = safe(model)
    tag = "claude-cli" if backend == "claude-cli" else "claude-api"
    return [SimulatorConfig(f"claude_{c}", f"{tag}_{model}_{c}", "n/a", c, f"{pretty(model)}, {'blind prompt' if c == 'blind' else 'real-respondent personas'}")
            for c in conds]


def plan(cfg, exps, n):
    """Dry pass: which prompts are not yet cached (with uniform answers standing in for uncached ones)."""
    col = _Collector()
    r = Runner(cfg, n_per_arm=n, lm=col)
    r.cache.f.close()
    orig_put = r.cache.put
    r.cache.put = lambda k, p: r.cache.d.__setitem__(k, p)  # don't persist fake answers
    for e in exps:
        r.run(e)
    r.cache.put = orig_put
    return list(col.missing)


def plan_ppi(cfg, exps, cap):
    """Dry pass over the PPI predictions (expected answer for each real respondent's persona)."""
    col = _Collector()
    r = Runner(cfg, lm=col)
    r.cache.f.close()
    r.cache.put = lambda k, p: r.cache.d.__setitem__(k, p)  # don't persist fake answers
    for e in exps:
        pp.predict(r, e, cap)
    return list(col.missing)


def _call_all(lm, cache, model_file, todo, conc, tag):
    t0 = time.time(); done = 0
    def one(key):
        s, u, l = key
        return key, lm.option_probs(s, u, list(l)).tolist()
    with ThreadPoolExecutor(conc) as ex:
        for (s, u, l), p in ex.map(one, todo):
            cache.put(Cache.key(model_file, s, u, list(l)), p)
            done += 1
            if done % 50 == 0 or done == len(todo):
                print(f"  [{tag}] {done}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    return done


def main(argv=None, lm=None, confirm=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simulator", choices=["claude-cli", "claude-api"], default="claude-cli")
    ap.add_argument("--model", default=None, help=f"Claude model id. CLI default: your Claude Code account's default model "
                    f"(the name it reports is used for labels). API default: {cl.DEFAULT_MODEL}")
    ap.add_argument("--n", type=int, default=150, help="simulated respondents per arm (default 150, same as the report)")
    ap.add_argument("--conditioning", nargs="*", default=["blind", "persona"], choices=["blind", "persona"])
    ap.add_argument("--only", nargs="*", default=None, help="experiment ids")
    ap.add_argument("--concurrency", type=int, default=None, help="parallel calls (default 4 for CLI, 8 for API)")
    ap.add_argument("--no-ppi", action="store_true", help="skip the per-person PPI predictions")
    ap.add_argument("--ppi-pool", type=int, default=None, help=f"real respondents per arm in the PPI pool (default {pp.POOL_CAP})")
    ap.add_argument("--quick", action="store_true", help=f"smaller run: --n {QUICK_N} and PPI pool {QUICK_POOL} per arm")
    ap.add_argument("--yes", "-y", action="store_true", help="skip the y/N prompt")
    a = ap.parse_args(argv)
    if a.quick:
        a.n = min(a.n, QUICK_N)
        a.ppi_pool = a.ppi_pool or QUICK_POOL
    pool = a.ppi_pool or pp.POOL_CAP
    do_ppi = not a.no_ppi and "persona" in a.conditioning
    conc = a.concurrency or (4 if a.simulator == "claude-cli" else 8)

    # Claude client + preflight first: one tiny call, so auth/model/limit problems show up before planning or the prompt.
    if lm is None:
        if a.simulator == "claude-cli":
            lm = cl.ClaudeCLI(model=a.model); lm.check()
            print("Checking Claude Code with one tiny call ('Reply with OK')...", flush=True)
            a.model = lm.preflight()
            print(f"  OK. Model: {a.model}" + ("" if lm.model else " (your account default)"), flush=True)
        else:
            a.model = a.model or cl.DEFAULT_MODEL
            lm = cl.ClaudeAPI(model=a.model)
    a.model = a.model or (cl.DEFAULT_MODEL if a.simulator == "claude-api" else cl.CLI_DEFAULT_LABEL)

    from .fetch_data import fetch
    fetch(required_only=True)
    exps = [e for e in load_all() if not a.only or e.id in a.only]
    cfgs = configs(a.simulator, a.model, a.conditioning)
    plans = {c.name: plan(c, exps, a.n) for c in cfgs}
    pcfg = next(c for c in configs(a.simulator, a.model, ["persona"]))
    ppi_todo = plan_ppi(pcfg, exps, pool) if do_ppi else []
    ppi_new = [k for k in ppi_todo if k not in set(plans.get("claude_persona", []))]
    allp = [cl.build_prompt(s, u, list(l)) for p in plans.values() for (s, u, l) in p]
    allp += [cl.build_prompt(s, u, list(l)) for (s, u, l) in ppi_new]
    tin, tout, usd = cl.estimate_cost(allp, a.model)
    print(f"Experiments: {len(exps)}   respondents/arm: {a.n}   backend: {a.simulator}   model: {a.model}")
    for c in cfgs:
        print(f"  {c.name}: {len(plans[c.name])} uncached Claude calls planned")
    if do_ppi:
        print(f"  PPI per-person predictions ({pool} real respondents/arm): {len(ppi_new)} uncached Claude calls planned")
    print(f"Planned calls: {len(allp)} (a few more may follow for answers that depend on an earlier answer)")
    print(f"ESTIMATED tokens: ~{tin:,} in + ~{tout:,} out (chars/3.5; ~{cl.EST_OUTPUT_TOKENS} out per call)")
    if a.simulator == "claude-api":
        print("ESTIMATED API cost: " + (f"~${usd:.2f} at list price" if usd is not None else "unknown price for this model"))
    else:
        print("Claude Code bills these to your signed-in plan; Claude Code also adds its own system prompt to every call, "
              "so actual usage is higher than the token estimate above.")
    if allp and not a.yes:
        ok = (confirm or input)("Proceed? [y/N] ").strip().lower() in ("y", "yes")
        if not ok:
            print("Aborted."); return 1

    out_dir = ROOT / "results" / "sims"; out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for c in cfgs:
        cache = Cache(c.model_file)
        rnd = 0
        while True:
            todo = plan(c, exps, a.n)
            if not todo:
                break
            rnd += 1
            print(f"[{c.name}] round {rnd}: {len(todo)} calls, concurrency {conc}", flush=True)
            total += _call_all(lm, cache, c.model_file, todo, conc, c.name)
            cache.d.clear(); cache.f.close(); cache = Cache(c.model_file)
        # all prompts cached: final pass produces the simulated respondents
        r = Runner(c, n_per_arm=a.n, lm=_Collector())
        res = {e.id: r.run(e) for e in exps}
        assert r.calls == 0
        res["_meta"] = {"n_per_arm": a.n, "model_calls_this_run": total, "label": c.label, "backend": a.simulator, "model": a.model,
                        "answer_scoring": "Claude-stated option probabilities (JSON), one sampled answer per respondent"}
        path = out_dir / f"{c.name}.json"
        if a.only and path.exists():
            res = {**json.loads(path.read_text()), **res}
        path.write_text(json.dumps(res))
        print(f"[{c.name}] wrote {path.relative_to(ROOT)}")
    if do_ppi:
        cache = Cache(pcfg.model_file)
        todo = plan_ppi(pcfg, exps, pool)
        if todo:
            print(f"[PPI] {len(todo)} calls, concurrency {conc}", flush=True)
            total += _call_all(lm, cache, pcfg.model_file, todo, conc, "PPI")
        cache.f.close()
        r = Runner(pcfg, lm=_Collector()); r.cache.f.close()
        out = ROOT / "results" / "ppi_preds"; out.mkdir(parents=True, exist_ok=True)
        path = out / "claude_persona.json"
        res = json.loads(path.read_text()) if (a.only and path.exists()) else {}
        res.update({e.id: pp.predict(r, e, pool) for e in exps})
        assert r.calls == 0
        path.write_text(json.dumps(res))
        print(f"[PPI] wrote {path.relative_to(ROOT)}")
    print(f"Done ({total} Claude calls this run). Build the report with: PYTHONPATH=src python -m effect_fidelity.report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
