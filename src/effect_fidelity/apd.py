"""Accuracy per dollar: raw small model vs. small model + LOEO correction vs. SocSci210-fine-tuned small model vs. Claude (if cached).

All rows are scored on the same 17 experiments with the same metrics (metrics.experiment_table / loeo_correction).
Cost is MEASURED wall-clock on this box (8 CPU cores) converted to $ with an ASSUMED cloud price; Claude $ is an ESTIMATE
from claude.estimate_cost (character-count tokens at list price). Both are labeled as estimates in the report.
  PYTHONPATH=src python -m effect_fidelity.apd
"""
from __future__ import annotations
import base64, json, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .experiments import load_all
from .metrics import experiment_table, loeo_correction
from .simulate import ROOT

RES, OUT = ROOT / "results", ROOT / "reports"
CPU_USD_PER_HOUR = 0.34  # ASSUMPTION: on-demand price of an 8-vCPU cloud VM (c-class, 2026 list ~ $0.30-0.40/h)
START, END = "<!-- APD-SECTION-START -->", "<!-- APD-SECTION-END -->"


def score(df_one: pd.DataFrame, corrected=False) -> dict:
    g = loeo_correction(df_one.reset_index(drop=True))
    if corrected:
        err = (g.d_corr - g.d_true).abs(); cov = g.covered_corr; wrong = g.corr_call != g.true_call
    else:
        err = (g.d_sim - g.d_true).abs(); cov = g.covered; wrong = g.sim_call != g.true_call
    return dict(mae_d=float(err.mean()), coverage=float(cov.mean()), wrong_rate=float(wrong.mean()), n_exp=int(len(g)))


def cost_rows():
    """Seconds of 8-core CPU per simulated respondent-answer, from logged runs."""
    t = json.loads((RES / "ft_timing.json").read_text()) if (RES / "ft_timing.json").exists() else {}
    out = {}
    for name in ("qwen15hf_raw_persona", "qwen15hf_ft_persona"):
        p = RES / "sims_ft" / f"{name}.json"
        if p.exists():
            m = json.loads(p.read_text()).get("_meta", {})
            nresp = 2 * m.get("n_per_arm", 60) * (len(json.loads(p.read_text())) - 1)
            out[name] = dict(sec_per_resp=m.get("seconds", np.nan) / max(nresp, 1), source="measured (HF bf16, 8 CPU cores)")
    q = RES / "sims" / "qwen15_persona.json"
    if q.exists():
        m = json.loads(q.read_text()).get("_meta", {})
        out["qwen15_persona"] = dict(sec_per_resp=np.nan, source="llama.cpp Q4 run (time not logged per respondent; see README)")
    for k, v in out.items():
        v["usd_per_1000"] = v["sec_per_resp"] * 1000 / 3600 * CPU_USD_PER_HOUR
    if "train" in t:
        out["_train"] = dict(seconds=t["train"]["seconds"], usd=t["train"]["seconds"] / 3600 * CPU_USD_PER_HOUR, **t["train"])
    return out


def main():
    exps = load_all()
    sims = {}
    for name, d in (("qwen15_persona", RES / "sims"), ("qwen15hf_raw_persona", RES / "sims_ft"), ("qwen15hf_ft_persona", RES / "sims_ft"),
                    ("claude_persona", RES / "sims"), ("claude_blind", RES / "sims")):
        p = d / f"{name}.json"
        if p.exists():
            r = json.loads(p.read_text()); r.pop("_meta", None); sims[name] = r
    df = experiment_table(exps, sims)
    costs = cost_rows()
    rows = []
    lab = {"qwen15_persona": "Qwen 1.5B raw (llama.cpp, n=150/arm)", "qwen15hf_raw_persona": "Qwen 1.5B raw (HF, n=60/arm)",
           "qwen15hf_ft_persona": "Qwen 1.5B + SocSci210 LoRA (n=60/arm)", "claude_persona": "Claude, personas", "claude_blind": "Claude, blind"}
    for s, g in df.groupby("simulator", sort=False):
        rows.append(dict(method=lab.get(s, s), simulator=s, **score(g), **{k: v for k, v in costs.get(s, {}).items()}))
        if s in ("qwen15_persona", "qwen15hf_raw_persona", "qwen15hf_ft_persona"):
            rows.append(dict(method=lab.get(s, s) + " + LOEO correction", simulator=s, **score(g, corrected=True),
                             **{k: v for k, v in costs.get(s, {}).items()}))
    t = pd.DataFrame(rows)
    t.to_csv(RES / "accuracy_per_dollar.csv", index=False)
    (RES / "accuracy_per_dollar.json").write_text(json.dumps({"rows": rows, "costs": costs, "cpu_usd_per_hour_assumed": CPU_USD_PER_HOUR}, indent=1, default=float))
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, col, ttl in zip(axs, ["mae_d", "coverage", "wrong_rate"], ["Effect error (SD units, lower better)", "95% range covers truth (higher better)", "Wrong ship/no-ship call (lower better)"]):
        ax.barh(t.method, t[col], color=[f"C{i}" for i in range(len(t))]); ax.set_title(ttl, fontsize=9); ax.tick_params(labelsize=7); ax.invert_yaxis()
        if col == "coverage": ax.axvline(.95, color="grey", ls="--")
    for ax in axs[1:]: ax.set_yticklabels([])
    fig.tight_layout(); fig.savefig(OUT / "fig_apd.png", dpi=150); plt.close(fig)
    inject(t, costs)
    print(t.round(3).to_string())
    return t, costs


def inject(t, costs):
    rows = "".join(f"<tr><td>{r.method}</td><td>{r.mae_d:.2f}</td><td>{r.coverage:.0%}</td><td>{r.wrong_rate:.0%}</td>"
                   f"<td>{'' if pd.isna(r.get('usd_per_1000', np.nan)) else f'${r.usd_per_1000:.3f} (est.)'}</td></tr>" for _, r in t.iterrows())
    tr = costs.get("_train", {})
    img = base64.b64encode((OUT / "fig_apd.png").read_bytes()).decode()
    sec = f"""{START}
<h2>Accuracy per dollar: does a cheap fine-tune help?</h2>
<p>Kolluri, Wu, Park &amp; Bernstein (2025, "Socrates", arXiv 2509.05830) showed that fine-tuning a model on real
people's answers from 210 past studies (SocSci210) makes simulated answer distributions much closer to real ones. We build on that and ask a
narrower product question: on experiments the model has never seen, does it get the <i>effect</i> and the <i>decision</i> right, and what does it cost?
We took a tiny slice of SocSci210 ({tr.get('examples', 'n/a')} answers), removed every study that could overlap our 17 experiments (keyword screen), and trained a small
LoRA adapter on Qwen2.5-1.5B on this box's CPU. None of our 17 experiments were used in training, so every score below is on unseen experiments.</p>
<img src="data:image/png;base64,{img}">
<table><tr><th>Method</th><th>Effect error (SD)</th><th>Range covers truth</th><th>Wrong call</th><th>$ per 1,000 simulated respondents</th></tr>{rows}</table>
<ul><li>Measured CPU time on this 8-core box, priced at an <b>assumed</b> ${CPU_USD_PER_HOUR}/hour cloud VM; Claude's cost, if present, is an estimate from list prices.
Training took {tr.get('seconds', float('nan'))/60:.0f} minutes (&asymp; ${tr.get('usd', float('nan')):.2f} at that rate).</li>
<li>This is a toy fine-tune (one pass, a few hundred answers, 1.5B model). The released Socrates checkpoints (Llama-3-8B, Qwen2.5-14B) don't fit
in this box's memory on CPU, so the full-strength version is untested here. Treat a null result as "this cheap version doesn't help", not "fine-tuning doesn't help".</li>
<li>The fine-tuned and its matching raw baseline run in the same HF bf16 setup with 60 respondents per arm and a 2-item short form for multi-item scales;
the llama.cpp row is the original 150-per-arm run. "LOEO correction" is the existing leave-one-experiment-out scale + model-error fix.</li></ul>
{END}"""
    p = OUT / "report.html"
    s = re.sub(re.escape(START) + ".*?" + re.escape(END), "", p.read_text(), flags=re.S)
    s = s.replace("<h2>Method</h2>", sec + "\n<h2>Method</h2>", 1)
    p.write_text(s)


if __name__ == "__main__":
    main()
