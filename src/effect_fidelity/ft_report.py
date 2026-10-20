"""Build the 'Accuracy per dollar' comparison: base vs fine-tuned Qwen2.5-1.5B (+ correction), vs Claude if present.
Writes results/ft_summary.json, results/ft_effects.csv, results/ft_costs.csv, results/ft_ppi_curves.csv,
reports/fig_ft_*.png and reports/finetune.html."""
from __future__ import annotations
import re
import base64, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import fonts as _fonts
_fonts.matplotlib()
from .experiments import load_all
from .metrics import experiment_table, summarize, loeo_correction, summarize_corrected
from .ppi import run_curves, drop_capped, summarize as ppi_summarize, NS
from .simulate import ROOT
from . import ft_eval as fe

RES, OUT = ROOT / "results", ROOT / "reports"


def load():
    sims, preds, labels = {}, {}, {}
    for w in ("raw", "ft"):
        p = fe.SIMS / f"qwen15hf_{w}_persona.json"
        if p.exists():
            d = json.loads(p.read_text()); d.pop("_meta", None); sims[f"qwen15hf_{w}_persona"] = d
            labels[f"qwen15hf_{w}_persona"] = fe.cfg(w).label
        q = fe.PPI / f"qwen15hf_{w}_persona.json"
        if q.exists():
            preds[f"qwen15hf_{w}_persona"] = json.loads(q.read_text())
    c = RES / "sims" / "claude_persona.json"
    if c.exists():
        d = json.loads(c.read_text()); labels["claude_persona"] = re.sub(r" \(claude-[^)]*\)", "", d.pop("_meta", {}).get("label", "Claude, personas")); sims["claude_persona"] = d
        q = RES / "ppi_preds" / "claude_persona.json"
        if q.exists():
            preds["claude_persona"] = json.loads(q.read_text())
    return sims, preds, labels


def effects(exps, sims):
    exps_ok = [e for e in exps if all(e.id in s for s in sims.values())]
    df = experiment_table(exps_ok, sims)
    rows = []
    for s, g in df.groupby("simulator", sort=False):
        g = g.reset_index(drop=True)
        a, c = summarize(g), summarize_corrected(loeo_correction(g))
        rows.append(dict(simulator=s, n_exp=len(g), mae_d_raw=c["mae_d_raw"], mae_d_corrected=c["mae_d_corrected"],
                         wrong_call_raw=c["wrong_call_raw"], wrong_call_corrected=c["wrong_call_corrected"],
                         coverage_raw=c["coverage_raw"], coverage_corrected=c["coverage_corrected"],
                         calib_slope=a["calib_slope"], pearson_r=a["pearson_r"], sign_agreement=a["sign_agreement"],
                         median_effect_ratio=a["median_effect_ratio"], false_alarm_rate_on_nulls=a["false_alarm_rate_on_nulls"]))
    return df, pd.DataFrame(rows)


def main(reps=300):
    exps = load_all()
    sims, preds, labels = load()
    if not sims:
        print("no fine-tune results yet"); return
    df, eff = effects(exps, sims)
    eff.to_csv(RES / "ft_effects.csv", index=False)
    timing = json.loads(fe.TIMING.read_text()) if fe.TIMING.exists() else {}
    ne = len(exps)
    cpr = {w: timing[f"sims_{w}"]["calls"] / (ne * 2 * 60) for w in ("raw", "ft") if f"sims_{w}" in timing}
    costs = fe.cost_table(timing, cpr)
    costs.to_csv(RES / "ft_costs.csv", index=False)
    curves = drop_capped(run_curves(exps, preds, sims, ns=tuple(n for n in NS if n <= 100), reps=reps)) if preds else pd.DataFrame()
    psum = ppi_summarize(curves) if len(curves) else {}
    if len(curves):
        curves.to_csv(RES / "ft_ppi_curves.csv", index=False)
    summ = dict(effects=eff.to_dict("records"), ppi=psum, costs=costs.to_dict("records"), timing=timing,
                notes="Costs are ESTIMATES: measured CPU seconds per call x assumed cloud price; see ft_eval.py constants.")
    (RES / "ft_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    paths = figures(eff, curves, costs, labels)
    write_html(eff, psum, costs, labels, paths, timing)
    print(json.dumps(summ, indent=1, default=float))


def figures(eff, curves, costs, labels):
    L = lambda s: labels.get(s, s)
    ps = [OUT / "fig_ft_effects.png", OUT / "fig_ft_ppi.png"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(eff)); w = .38
    for ax, (raw, cor, t) in zip(axes, [("mae_d_raw", "mae_d_corrected", "Effect error (|d| units, lower = better)"),
                                         ("wrong_call_raw", "wrong_call_corrected", "Wrong ship / no-ship call (lower = better)")]):
        ax.bar(x - w / 2, eff[raw], w, label="as-is"); ax.bar(x + w / 2, eff[cor], w, label="+ correction")
        ax.set_xticks(x); ax.set_xticklabels([L(s).replace(", ", "\n") for s in eff.simulator], fontsize=7); ax.set_title(t, fontsize=9); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ps[0], dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(curves):
        agg = curves.groupby(["simulator", "n"]).mean(numeric_only=True).reset_index()
        f = agg[agg.simulator == agg.simulator.iloc[0]]
        ax.plot(f.n, f.real_wrong, "k-o", lw=2.5, label="Real people only")
        for i, s in enumerate(agg.simulator.unique()):
            g = agg[agg.simulator == s]; ax.plot(g.n, g.ppi_wrong, "-o", color=f"C{i}", label=f"Real + {L(s)}")
        ax.set_xscale("log"); ax.set_xticks(list(f.n)); ax.set_xticklabels(list(f.n)); ax.set_ylim(0, 1)
    ax.set_xlabel("Real respondents per arm"); ax.set_ylabel("Wrong call rate"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax.set_title("How many real people do you still need? (base vs fine-tuned)", fontsize=9)
    fig.tight_layout(); fig.savefig(ps[1], dpi=150); plt.close(fig)
    return ps


def _tbl(df, fmt):
    h = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td>{fmt(v)}</td>" for v in r) + "</tr>" for r in df.itertuples(index=False))
    return f"<table><tr>{h}</tr>{rows}</table>"


def write_html(eff, psum, costs, labels, paths, timing):
    f = lambda v: (f"{v:.3g}" if isinstance(v, float) and np.isfinite(v) else ("" if isinstance(v, float) else str(v)))
    e = eff.assign(simulator=eff.simulator.map(lambda s: labels.get(s, s)))
    p = pd.DataFrame([dict(simulator=labels.get(s, s), median_r=v["rho_median"], **{f"worth@{n}": v["by_n"][n]["gain_median"] for n in v["by_n"]},
                           **{f"wrong@{n} real+sim": v["by_n"][n]["ppi_wrong"] for n in v["by_n"]}) for s, v in psum.items()])
    imgs = "".join(f'<img style="max-width:100%" src="data:image/png;base64,{base64.b64encode(q.read_bytes()).decode()}">' for q in paths)
    (OUT / "finetune.html").write_text(f"""<html><head><meta charset="utf-8"><title>Accuracy per dollar</title>
<style>{_fonts.css()}body{{font-family:'DM Sans',sans-serif;max-width:1000px;margin:auto}}td,th{{border:1px solid #ccc;padding:3px 6px;font-size:12px}}table{{border-collapse:collapse}}</style></head><body>
<h1>Accuracy per dollar: does a cheap fine-tune fix small simulators?</h1>
<p>Builds on Kolluri, Wu, Park &amp; Bernstein (2025), <a href="https://arxiv.org/abs/2509.05830">arXiv 2509.05830</a>, who fine-tuned open models on
SocSci210. Here: a deliberately tiny CPU LoRA (Qwen2.5-1.5B-Instruct, 480 SocSci210 answers, 1 epoch, r=8), with every study that might overlap our
17 experiments removed, evaluated on those 17 experiments, scored on effects and decisions, and priced.</p>
<h2>Effects and decisions</h2>{_tbl(e, f)}{imgs}
<h2>PPI ("how many real people")</h2>{_tbl(p, f) if len(p) else "<p>not run</p>"}
<h2>Cost (ESTIMATES)</h2>{_tbl(costs, f)}
<p>Measured compute time: {json.dumps(timing)}. Dollar figures are estimates: measured CPU seconds &times; ${fe.CPU_USD_HR}/hour for an 8-vCPU cloud VM;
GPU column assumes ${fe.GPU_USD_HR}/hour and {fe.GPU_SPEEDUP:.0f}x speed-up (not measured); Claude column is API list price for the same prompts
(chars/3.5 tokens; not measured).</p></body></html>""")


if __name__ == "__main__":
    main()
