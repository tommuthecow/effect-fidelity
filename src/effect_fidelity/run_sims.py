"""Simulate every experiment with every simulator; writes results/sims/<simulator>.json (resumable via cache)."""
from __future__ import annotations
import argparse, json, time
from .experiments import load_all
from .simulate import Runner, SIMULATORS, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="simulated respondents per arm")
    ap.add_argument("--sims", nargs="*", default=[s.name for s in SIMULATORS if s.model_file != "claude"],
                    help="local simulators (Claude runs through run_claude.py)")
    ap.add_argument("--only", nargs="*", default=None, help="experiment ids")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    exps = [e for e in load_all() if not a.only or e.id in a.only]
    out_dir = ROOT / "results" / "sims"
    out_dir.mkdir(parents=True, exist_ok=True)
    for cfg in [s for s in SIMULATORS if s.name in a.sims and s.model_file != "claude"]:
        r = Runner(cfg, n_per_arm=a.n)
        if a.threads != 8:
            from .llm import LocalLM
            r._lm = LocalLM(str(ROOT / "models" / cfg.model_file), cfg.template, n_threads=a.threads)
        path = out_dir / f"{cfg.name}.json"
        res = json.loads(path.read_text()) if path.exists() else {}
        t0 = time.time()
        for e in exps:
            res[e.id] = r.run(e)
            res["_meta"] = {"n_per_arm": a.n, "model_calls_this_run": r.calls, "cache_hits": r.cache.hits,
                            "seconds_this_run": round(time.time() - t0, 1), "label": cfg.label}
            path.write_text(json.dumps(res))
            print(f"[{cfg.name}] {e.id} done; new calls={r.calls} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
