"""Accuracy per dollar, part 2: evaluate base vs SocSci210-LoRA Qwen2.5-1.5B on our 17 (never-trained-on) experiments.

Stages (each writes to disk after every experiment and caches every model answer in data/cache/, so reruns resume):
  python -m effect_fidelity.ft_eval sims raw|ft     # 60 simulated respondents/arm -> results/sims_ft/qwen15hf_<w>_persona.json
  python -m effect_fidelity.ft_eval ppi  raw|ft     # per-person predictions, pool POOL/arm -> results/ppi_preds_ft/
  python -m effect_fidelity.ft_eval report          # effect metrics + correction + PPI curves + $ estimates -> results/, reports/
Both HF models (bf16, CPU) are scored the same way, so base-vs-fine-tuned is apples to apples. Claude is included if
results/sims/claude_persona.json / results/ppi_preds/claude_persona.json exist (from ./run_claude.sh).
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import pandas as pd
from .simulate import ROOT, Runner, SimulatorConfig
from .experiments import load_all
from . import finetune as ft, ppi_preds as pp

POOL = 200  # real respondents per arm for PPI (n is capped at half the pool, so curves go up to 100/arm)
SIMS, PPI = ROOT / "results" / "sims_ft", ROOT / "results" / "ppi_preds_ft"
TIMING = ROOT / "results" / "ft_timing.json"
# ESTIMATES, not measurements: an on-demand 8-vCPU x86 cloud VM (roughly the box this ran on) ~ $0.35/hour (2026 list prices vary);
# a 24 GB GPU (e.g. L4) ~ $0.80/hour and assumed ~40x faster than this CPU for a 1.5B model.
CPU_USD_HR, GPU_USD_HR, GPU_SPEEDUP = 0.35, 0.80, 40.0


def cfg(which):
    return SimulatorConfig(f"qwen15hf_{which}_persona", f"qwen15hf_{which}", "n/a", "persona",
                           f"Qwen2.5-1.5B {'+ SocSci210 LoRA (fine-tuned)' if which == 'ft' else 'base'}, personas")


class _Wrap:
    def __init__(self, r): self.r = r
    def option_probs(self, s, u, l): return self.r.probs(s, u, list(l))


def _timing(key, secs, calls):
    d = json.loads(TIMING.read_text()) if TIMING.exists() else {}
    old = d.get(key, {"seconds": 0, "calls": 0})
    d[key] = {"seconds": round(old["seconds"] + secs, 1), "calls": old["calls"] + calls}
    TIMING.write_text(json.dumps(d, indent=1))


def _runner(which, lm=None):
    if lm is None:
        tok, m = ft._tok_model(adapter=(which == "ft")); lm = ft.HFLM(tok, m)
    return Runner(cfg(which), lm=lm)


def run_sims(which, lm=None, exps=None):
    SIMS.mkdir(parents=True, exist_ok=True)
    path = SIMS / f"qwen15hf_{which}_persona.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    r = None
    for e in exps or load_all():
        if e.id in res:
            continue
        r = r or _runner(which, lm)
        t0, c0 = time.time(), r.calls
        res[e.id] = ft.simulate(_Wrap(r), e, n=ft.N_EVAL)
        res["_meta"] = {"label": cfg(which).label, "n_per_arm": ft.N_EVAL}
        path.write_text(json.dumps(res))
        _timing(f"sims_{which}", time.time() - t0, r.calls - c0)
        print(f"[sims {which}] {e.id} calls={r.calls - c0} {time.time()-t0:.0f}s", flush=True)


def run_ppi(which, lm=None, exps=None, pool=POOL):
    PPI.mkdir(parents=True, exist_ok=True)
    path = PPI / f"qwen15hf_{which}_persona.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    r = None
    for e in exps or load_all():
        if e.id in res:
            continue
        r = r or _runner(which, lm)
        t0, c0 = time.time(), r.calls
        res[e.id] = pp.predict(r, e, pool)
        path.write_text(json.dumps(res))
        _timing(f"ppi_{which}", time.time() - t0, r.calls - c0)
        print(f"[ppi {which}] {e.id} calls={r.calls - c0} {time.time()-t0:.0f}s", flush=True)


def cost_table(timing, sims_calls_per_resp):
    """$ per 1,000 simulated respondents. sims_calls_per_resp: measured model calls per simulated respondent."""
    from . import claude as cl
    rows = []
    sec_call = {w: timing[f"sims_{w}"]["seconds"] / max(timing[f"sims_{w}"]["calls"], 1) for w in ("raw", "ft") if f"sims_{w}" in timing}
    for w, s in sec_call.items():
        cpr = sims_calls_per_resp[w]
        cpu_h = s * cpr * 1000 / 3600
        rows.append(dict(simulator=cfg(w).label, sec_per_call_measured=s, calls_per_respondent=cpr,
                         cpu_hours_per_1k=cpu_h, usd_per_1k_cpu_est=cpu_h * CPU_USD_HR,
                         usd_per_1k_gpu_est=cpu_h / GPU_SPEEDUP * GPU_USD_HR))
    if "train" in timing:
        t = timing["train"]["seconds"] / 3600
        rows.append(dict(simulator="One-off: LoRA training (480 examples)", cpu_hours_per_1k=t, usd_per_1k_cpu_est=t * CPU_USD_HR,
                         usd_per_1k_gpu_est=t / GPU_SPEEDUP * GPU_USD_HR))
    # Claude: API list price for the same prompts (estimate: chars/3.5 tokens in, a short JSON answer out)
    e = load_all()[0]
    from .simulate import SYSTEM, render
    per = e.persona_text(e.persona_pool.iloc[0])
    prompts = [cl.build_prompt(SYSTEM, *render(it, "A", per)[:2]) for it in e.items[:ft.MAX_ITEMS]]
    _, _, usd = cl.estimate_cost(prompts * 1000, cl.DEFAULT_MODEL)
    cpr = float(np.mean(list(sims_calls_per_resp.values()))) if sims_calls_per_resp else 2
    if usd is not None:
        rows.append(dict(simulator=f"Claude ({cl.DEFAULT_MODEL}) via API", calls_per_respondent=cpr,
                         usd_per_1k_api_est=usd / len(prompts) * cpr))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["sims", "ppi", "report"]); ap.add_argument("which", nargs="?", default="ft")
    a = ap.parse_args()
    if a.stage == "sims": run_sims(a.which)
    elif a.stage == "ppi": run_ppi(a.which)
    else:
        from .ft_report import main; main()
