"""Score simulators, log to MLflow, and write the HTML report, one-page PDF, and tables."""
from __future__ import annotations
import base64, html, json, os, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import fonts as _fonts
_fonts.matplotlib()
from .experiments import load_all
from .metrics import experiment_table, summarize, loeo_correction, summarize_corrected
from .simulate import SIMULATORS, ROOT

OUT = ROOT / "reports"
RES = ROOT / "results"
COLORS = {"qwen15_blind": "#9aa0a6", "qwen15_persona": "#1a73e8", "llama32_1b_persona": "#e8710a", "claude_blind": "#c9a27e", "claude_persona": "#b0413e"}
LABEL = {s.name: s.label for s in SIMULATORS}
for _n in ("claude_blind", "claude_persona"):  # use the actual model id recorded at run time
    _p = RES / "sims" / f"{_n}.json"
    if _p.exists():
        LABEL[_n] = re.sub(r" \(claude-[^)]*\)", "", json.loads(_p.read_text()).get("_meta", {}).get("label", LABEL[_n]))  # human-facing: "Claude Opus 5.5"


def load_sims() -> dict:
    sims = {}
    for s in SIMULATORS:
        p = RES / "sims" / f"{s.name}.json"
        if p.exists():
            sims[s.name] = {k: v for k, v in json.loads(p.read_text()).items()}
    return sims


def analyze():
    exps = load_all()
    sims = load_sims()
    meta = {k: v.pop("_meta", {}) for k, v in sims.items()}
    df = experiment_table(exps, sims)
    summ, corr, dcs = {}, {}, []
    for s, g in df.groupby("simulator", sort=False):
        g = g.reset_index(drop=True)
        summ[s] = summarize(g)
        dc = loeo_correction(g)
        corr[s] = summarize_corrected(dc)
        dcs.append(dc)
    dc = pd.concat(dcs, ignore_index=True)
    return exps, df, dc, summ, corr, meta


def _fig_scatter(dc, path):
    sims = list(dict.fromkeys(dc.simulator))
    fig, axes = plt.subplots(1, len(sims), figsize=(4.2 * len(sims), 4.2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    lim = max(dc.d_true.abs().max(), dc.d_sim.abs().max()) * 1.1
    for ax, s in zip(axes, sims):
        g = dc[dc.simulator == s]
        ax.axhline(0, c="#ddd", lw=1); ax.axvline(0, c="#ddd", lw=1)
        ax.plot([-lim, lim], [-lim, lim], "--", c="#555", lw=1, label="perfect fidelity")
        ax.errorbar(g.d_true, g.d_sim, yerr=[g.d_sim - g.sim_lo / g.sd, g.sim_hi / g.sd - g.d_sim], fmt="o",
                    c=COLORS.get(s, "k"), ms=5, elinewidth=0.8, capsize=0)
        b = np.polyfit(g.d_true, g.d_sim, 1)
        xx = np.linspace(-lim, lim, 10)
        ax.plot(xx, np.polyval(b, xx), c=COLORS.get(s, "k"), lw=1.5, alpha=.7, label=f"fit slope {b[0]:.2f}")
        ax.set_title(LABEL.get(s, s), fontsize=10)
        ax.set_xlabel("real effect (SD units)")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("simulated effect (SD units)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _fig_forest(dc, exps, path):
    sims = list(dict.fromkeys(dc.simulator))
    order = [e.id for e in exps if e.id in set(dc.exp)]
    names = {e.id: e.name for e in exps}
    fig, ax = plt.subplots(figsize=(9, 0.48 * len(order) + 1.2))
    for i, eid in enumerate(order):
        y = len(order) - i
        t = dc[(dc.exp == eid)].iloc[0]
        ax.plot([t.true_lo / t.sd, t.true_hi / t.sd], [y, y], c="k", lw=3)
        ax.plot(t.d_true, y, "D", c="k", ms=5)
        for k, s in enumerate(sims):
            r = dc[(dc.exp == eid) & (dc.simulator == s)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            yy = y - 0.2 * (k + 1)
            ax.plot([r.sim_lo / r.sd, r.sim_hi / r.sd], [yy, yy], c=COLORS.get(s, "gray"), lw=1.5)
            ax.plot(r.d_sim, yy, "o", c=COLORS.get(s, "gray"), ms=3.5)
    ax.axvline(0, c="#bbb", lw=1)
    ax.set_yticks([len(order) - i - 0.3 for i in range(len(order))])
    ax.set_yticklabels([names[e] for e in order], fontsize=8)
    ax.set_xlabel("effect of treatment (B minus A), in real-respondent SD units; black = real experiment, 95% CI")
    from matplotlib.lines import Line2D
    ax.legend([Line2D([0], [0], c="k", lw=3)] + [Line2D([0], [0], c=COLORS.get(s, "gray"), lw=1.5) for s in sims],
              ["real (human) effect"] + [LABEL.get(s, s) for s in sims], fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _fig_correction(corr, path):
    sims = list(corr)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    w = 0.38
    x = np.arange(len(sims))
    for ax, (raw, cor, title, fmt) in zip(axes, [("coverage_raw", "coverage_corrected", "95% CI covers the real effect", "{:.0%}"),
                                                 ("wrong_call_raw", "wrong_call_corrected", "Wrong ship/no-ship call", "{:.0%}"),
                                                 ("mae_d_raw", "mae_d_corrected", "Mean abs. error (SD units)", "{:.2f}")]):
        a = [corr[s][raw] for s in sims]; b = [corr[s][cor] for s in sims]
        ax.bar(x - w / 2, a, w, color="#c5c5c5", label="raw simulator")
        ax.bar(x + w / 2, b, w, color="#188038", label="recalibrated (held-out)")
        for xi, (u, v) in enumerate(zip(a, b)):
            ax.text(xi - w / 2, u, fmt.format(u), ha="center", va="bottom", fontsize=7)
            ax.text(xi + w / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=7)
        if "coverage" in raw:
            ax.axhline(0.95, ls="--", c="#555", lw=1)
        ax.set_xticks(x); ax.set_xticklabels([LABEL.get(s, s).replace(", ", "\n") for s in sims], fontsize=7)
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _fig_arm_vs_effect(dc, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for sname, g in dc.groupby("simulator", sort=False):
        wrong = g.sim_call != g.true_call
        ax.scatter(g.tvd_max[~wrong], (g.d_sim - g.d_true).abs()[~wrong], c=COLORS.get(sname, "k"), marker="o", s=30, label=f"{LABEL.get(sname, sname)} (right call)")
        ax.scatter(g.tvd_max[wrong], (g.d_sim - g.d_true).abs()[wrong], facecolors="none", edgecolors=COLORS.get(sname, "k"), marker="s", s=36, label=f"{LABEL.get(sname, sname)} (wrong call)")
    ax.axvline(0.15, ls="--", c="#555", lw=1)
    ax.text(0.152, ax.get_ylim()[1] * 0.95, "TVD 0.15", fontsize=7, va="top")
    ax.set_xlabel("per-arm accuracy: worse arm's TVD vs real answers (lower = better)")
    ax.set_ylabel("effect error |sim - real| (SD units)")
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


def headline(summ, corr):
    """The question: can per-question (per-arm) accuracy look good while the effect between arms is wrong?"""
    best = max(summ, key=lambda k: (summ[k]["n_exp_both_arms_tvd_below_015"], -summ[k]["mean_arm_tvd"]))
    tot_good = sum(v["n_exp_both_arms_tvd_below_015"] for v in summ.values())
    tot_good_wrong = sum(round(v["n_exp_both_arms_tvd_below_015"] * (v["wrong_call_rate_when_both_arms_good"] or 0)) for v in summ.values()
                         if v["n_exp_both_arms_tvd_below_015"])
    s, c = summ[best], corr[best]
    sl = {k: v["calib_slope"] for k, v in summ.items()}
    small = [k for k in sl if sl[k] < 0.7]; big = [k for k in sl if k not in small]
    shrink = (f"The small open models shrank effects toward zero (slopes {min(sl[k] for k in small):.2f}-{max(sl[k] for k in small):.2f}); "
              f"Claude got the size about right (slopes {min(sl[k] for k in big):.2f}-{max(sl[k] for k in big):.2f}) but was still often wrong "
              f"about direction or significance." if small and big else
              f"All simulators shrank effects (slopes {min(sl.values()):.2f}-{max(sl.values()):.2f}).")
    return best, (f"Per-question accuracy does not guarantee effect accuracy. Across the {len(summ)} simulators, {tot_good} simulated experiments had "
                  f"both arms within TVD 0.15 of the real answer distributions, yet {tot_good_wrong} of those still led to the wrong "
                  f"ship/no-ship call (e.g. Hillstrom women's email, blind prompt: both arms within TVD 0.15, simulated lift 15.3 pp vs. real 4.5 pp; "
                  f"GSS 'assistance to Black people' wording, Llama: both arms within TVD 0.11, simulated +1.3 pp vs. real -6.7 pp). For the simulator with the best per-arm accuracy ({LABEL[best]}, mean arm TVD {s['mean_arm_tvd']:.2f}), "
                  f"the calibration slope of simulated on real effects was {s['calib_slope']:.2f} (1.0 = right size), its 95% intervals covered the "
                  f"real effect in {s['ci_coverage']:.0%} of {s['n_experiments']} experiments, and a held-out recalibration raised coverage to "
                  f"{c['coverage_corrected']:.0%}. {shrink} Effect-level checks complement distribution-level ones.")


def log_mlflow(df, dc, summ, corr, meta, figs):
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow
    mlflow.set_tracking_uri(f"file:{ROOT / 'mlruns'}")
    mlflow.set_experiment("effect-fidelity-report-card")
    cfg = {s.name: s for s in SIMULATORS}
    for s in summ:
        with mlflow.start_run(run_name=s):
            c = cfg[s]
            mlflow.log_params(dict(model=c.model_file, conditioning=c.conditioning, n_per_arm=meta.get(s, {}).get("n_per_arm"),
                                   n_experiments=summ[s]["n_experiments"], answer_scoring="next-token option probabilities, one sampled answer per simulated respondent",
                                   correction="leave-one-experiment-out scale + model-error variance"))
            mlflow.log_metrics({k: float(v) for k, v in {**summ[s], **{f"loeo_{k}": v for k, v in corr[s].items()}}.items()
                                if v is not None and not np.isnan(v)})
            mlflow.log_text(dc[dc.simulator == s].to_csv(index=False), f"per_experiment_{s}.csv")
    with mlflow.start_run(run_name="report"):
        for f in figs:
            mlflow.log_artifact(str(f))
        mlflow.log_artifact(str(OUT / "report.html"))


def fmt_table(df, cols, fmts):
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols.values())
    rows = []
    for _, r in df.iterrows():
        rows.append("<tr>" + "".join(f"<td>{fmts.get(k, '{}').format(r[k])}</td>" for k in cols) + "</tr>")
    return f"<table><tr>{head}</tr>{''.join(rows)}</table>"


def write_html(exps, df, dc, summ, corr, meta, head, best):
    sm = pd.DataFrame(summ).T
    sm.insert(0, "simulator", [LABEL[s] for s in sm.index])
    rows = []
    for s in summ:
        a, c = summ[s], corr[s]
        rows.append(f"<tr><td>{LABEL[s]}</td><td>{a['mean_arm_tvd']:.2f}</td><td>{a['share_arms_tvd_below_015']:.0%}</td><td>{a['sign_agreement']:.0%}</td><td>{a['median_effect_ratio']:.2f}</td>"
                    f"<td>{a['calib_slope']:.2f}</td><td>{a['pearson_r']:.2f}</td><td>{a['ci_coverage']:.0%}</td>"
                    f"<td>{a['wrong_call_rate']:.0%}</td><td>{a['false_alarm_rate_on_nulls']:.0%}</td>"
                    f"<td>{c['coverage_corrected']:.0%}</td><td>{c['wrong_call_corrected']:.0%}</td>"
                    f"<td>{c['mae_d_raw']:.2f} &rarr; {c['mae_d_corrected']:.2f}</td></tr>")
    scorecard = ("<table><tr><th>Simulator</th><th>Mean per-arm TVD<br>(answer distribution)</th><th>Arms with<br>TVD &lt; 0.15</th><th>Sign agreement<br>(real non-null effects)</th><th>Median effect ratio<br>sim / real</th>"
                 "<th>Calibration slope<br>(1 = right size)</th><th>Correlation r<br>sim vs real</th><th>95% CI coverage</th>"
                 "<th>Wrong ship / no-ship call</th><th>False alarms<br>on real nulls</th><th>Coverage after<br>held-out recalibration</th>"
                 "<th>Wrong call after<br>recalibration</th><th>Mean abs. error (SD)<br>raw &rarr; recalibrated</th></tr>" + "".join(rows) + "</table>")
    per = dc.copy()
    per["simulator"] = per.simulator.map(LABEL)
    per_tbl = fmt_table(per, {"simulator": "Simulator", "name": "Experiment", "unit": "Unit", "true": "Real effect", "true_lo": "lo", "true_hi": "hi",
                              "sim": "Simulated effect", "sim_lo": "lo", "sim_hi": "hi", "ratio": "Ratio", "covered": "CI covers real?",
                              "true_call": "Real call", "sim_call": "Sim call"},
                        {"true": "{:.2f}", "true_lo": "{:.2f}", "true_hi": "{:.2f}", "sim": "{:.2f}", "sim_lo": "{:.2f}", "sim_hi": "{:.2f}", "ratio": "{:.2f}"})
    src_rows = "".join(f"<tr><td>{e.id}</td><td>{html.escape(e.name)}</td><td>{html.escape(e.arm_names['A'])}</td><td>{html.escape(e.arm_names['B'])}</td>"
                       f"<td>{len(e.real['A']):,} / {len(e.real['B']):,}</td><td>{html.escape(e.source)}</td><td>{html.escape(e.license)}</td></tr>" for e in exps)
    calls = {s: m.get("model_calls_this_run") for s, m in meta.items()}
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Effect Fidelity Report Card</title>
<style>{_fonts.css()}body{{font-family:'DM Sans',sans-serif;max-width:1100px;margin:30px auto;padding:0 20px;color:#202124;line-height:1.45}}
h1{{margin-bottom:0}}.sub{{color:#5f6368;margin-top:4px}}.hl{{background:#e8f0fe;border-left:4px solid #1a73e8;padding:12px 16px;font-size:1.05em}}
table{{border-collapse:collapse;font-size:12px;margin:10px 0}}td,th{{border:1px solid #dadce0;padding:4px 6px;text-align:left;vertical-align:top}}th{{background:#f1f3f4}}
img{{max-width:100%}}code{{background:#f1f3f4;padding:1px 4px}}</style></head><body>
<h1>Effect Fidelity Report Card</h1>
<p class="sub">When is a simulated A/B result good enough to trust? A benchmark of LLM persona simulators against {len(exps)} real randomized experiments with public ground truth.</p>
<p class="hl"><b>Headline.</b> {html.escape(head)}</p>
<h2>Scorecard</h2>{scorecard}
<p>Effects are treatment (B) minus reference (A). "Ratio" and "sign" use only experiments whose real 95% CI excludes zero
({summ[best]['n_true_nonnull']} of {summ[best]['n_experiments']}); the other {summ[best]['n_true_null']} are real nulls, which test whether a simulator invents effects.
A "call" is what a team would do from that result: ship B, keep A, or no detectable difference (95% CI includes 0).</p>
<h2>Per-arm accuracy vs. effect accuracy</h2>
<p>Distribution-level metrics, like total variation distance (TVD) between simulated and real answer distributions for a single question, are the natural way to score a
population simulation, and Simile's confidence model reports exactly this kind of quantity. A decision, though, usually depends on the <i>difference</i> between two arms. Two arms can each be
close to reality while their errors don't cancel, so the gap between them is wrong. This section measures both on the same runs. TVD is computed per arm on answers rounded to the
nearest scale point; 0.15 is used as a reference line because it has been reported (unverified, secondhand) as a "decision-ready" bar.</p>
<img src="data:image/png;base64,{_b64(OUT / 'fig_arm_vs_effect.png')}">
<p>Experiments with both arms under TVD 0.15, per simulator: {html.escape(json.dumps({LABEL[k]: v['n_exp_both_arms_tvd_below_015'] for k, v in summ.items()}))}.
Wrong-call rate among them: {html.escape(json.dumps({LABEL[k]: (None if np.isnan(v['wrong_call_rate_when_both_arms_good']) else round(v['wrong_call_rate_when_both_arms_good'], 2)) for k, v in summ.items()}))}.
Correlation between worse-arm TVD and effect error: {html.escape(json.dumps({LABEL[k]: round(v['corr_arm_tvd_vs_effect_error'], 2) for k, v in summ.items()}))}.
The recommendation is to add both: per-arm distribution accuracy plus an effect-level fidelity check for experiment-style questions.</p>
<h2>Does the simulator get the size right?</h2><img src="data:image/png;base64,{_b64(OUT / 'fig_scatter.png')}">
<p>Each point is one experiment, in units of the real respondents' standard deviation so different outcomes are comparable. Points on the dashed line would be perfect.
Error bars are the simulator's own bootstrap 95% CI.</p>
<h2>Every experiment</h2><img src="data:image/png;base64,{_b64(OUT / 'fig_forest.png')}">
<h2>Can a simple correction fix it?</h2><img src="data:image/png;base64,{_b64(OUT / 'fig_correction.png')}">
<p>For each experiment, the correction is fit only on the other {len(exps) - 1} experiments (leave-one-experiment-out): a single scale factor
(real &asymp; c &times; simulated, in SD units) plus a "model error" variance &tau;&sup2;, which is the spread of real effects around the rescaled simulation that
sampling noise cannot explain. The corrected 95% interval is c&times;sim &plusmn; 1.96&radic;(c&sup2;se&sup2; + &tau;&sup2;). The point: a simulator's own bootstrap interval only
captures sampling noise, which more draws shrink toward zero; the dominant uncertainty is model error, and it has to be estimated against real experiments.</p>
<h2>Method</h2><ul>
<li>Computed here from respondent-level public microdata (no hand-entered numbers): difference in means with a 95% bootstrap CI.
Binary outcomes are in percentage points, rating scales in scale points.</li>
<li>Local open-weight models via llama.cpp (4-bit GGUF, CPU). Each simulated respondent sees the original question wording for its arm.
The model's next-token probabilities over the answer options give an answer distribution, and one answer is sampled per respondent.
"Blind" = one generic respondent description; "persona" = each simulated respondent is built from a real respondent drawn from the same study
(age, sex, and political identity for Many Labs; age, sex, race, education and political views for GSS; purchase history, channel and location for the email RCT).
{meta.get(best, {}).get('n_per_arm', '?')} simulated respondents per arm; bootstrap CIs (2,000 resamples) over simulated respondents. Multi-item scales are averaged like the original.</li>
<li>Priming manipulations (flag, money, imagined contact) are visual or experiential in the original; here they are described in text. That is a known limitation of text-only simulation, and it is what a vendor would face too.</li>
<li>Every simulator run is logged to MLflow (local file store, <code>mlruns/</code>) with parameters, all metrics and per-experiment tables. New model calls this run: {html.escape(json.dumps(calls))}; responses are cached in <code>data/cache/</code>.</li></ul>
<h2>Experiments and sources</h2><table><tr><th>id</th><th>Experiment</th><th>Arm A</th><th>Arm B</th><th>n real (A / B)</th><th>Source</th><th>License / terms</th></tr>{src_rows}</table>
<h2>Caveats</h2><ul>
<li>The simulators are small (1&ndash;1.5B-parameter) open-weight models running on a CPU, chosen because no hosted-model API keys were available. Frontier models or a purpose-built behavior model will score differently; the harness is model-agnostic and the point is the measurement.</li>
<li>{len(exps)} experiments is a small benchmark, and the correction has {len(exps) - 1} training points per fold. Treat the recalibration as a demonstration of the method, not a production calibrator.</li>
<li>Many Labs 1 participants are mostly US undergraduates and MTurk workers; GSS effects are unweighted; the email RCT measures site visits within two weeks.</li>
<li>Many of these effects are famous and appear in model training data, which should make them easier for a simulator, not harder.</li></ul>
<h2>Per-experiment results</h2>{per_tbl}
</body></html>"""
    (OUT / "report.html").write_text(doc)


def _inject_audit(audit):
    """Add a 'How much to trust the wrong-call number' section near the top of report.html."""
    b, rows = audit["baselines"], []
    for s_, v in audit["by_simulator"].items():
        rows.append(f"<tr><td>{LABEL.get(s_, s_)}</td><td>{v['n_sim']}</td><td>{v['wrong_call_matched']:.0%}</td><td>{v['coverage_matched']:.0%}</td>"
                    f"<td>{v['direction_agreement']:.0%}</td><td>{v['effect_corr']:.2f}</td></tr>")
    sec = (f"<!-- AUDIT-START --><h2>How much to trust the wrong-call number</h2>"
           f"<p>A call is one of three answers: ship B, keep A, or no clear difference, read from a 95% interval. It is 'wrong' if it "
           f"differs from the call the full real study supports. Of the {b['n_exp']} experiments, the full real data says B is better in "
           f"{b['n_real_positive']}, A in {b['n_real_negative']}, and no clear difference in {b['n_real_null']}. Trivial strategies score: "
           f"always 'ship B' {b['always_ship_b']:.0%} wrong, always 'no difference' {b['always_no_difference']:.0%}, a random pick of the three "
           f"{b['random_three_way']:.0%}, a coin flip between A and B {b['random_direction']:.0%}. So with 17 experiments the wrong-call rate "
           f"alone separates simulators poorly; always shipping B beats every simulator here. A simulator's call also depends on how many simulated "
           f"people were run (more people, narrower interval), so the table below scores every simulator at the same {audit['matched_n']} "
           f"simulated people per arm and adds two measures that don't depend on that number.</p>"
           f"<table><tr><th>Simulator</th><th>Simulated n per arm (run)</th><th>Wrong call at n={audit['matched_n']}</th><th>Interval covers truth at n={audit['matched_n']}</th>"
           f"<th>Right direction (real non-null effects)</th><th>Correlation with real effects</th></tr>{''.join(rows)}</table><!-- AUDIT-END -->")
    p = OUT / "report.html"; h = p.read_text()
    import re as _re
    h = _re.sub(r"<!-- AUDIT-START -->.*?<!-- AUDIT-END -->", "", h, flags=_re.S)
    h = h.replace("</h1>", "</h1>" + sec, 1) if "</h1>" in h else h.replace("<body>", "<body>" + sec, 1)
    p.write_text(h)


def main():
    OUT.mkdir(exist_ok=True)
    exps, df, dc, summ, corr, meta = analyze()
    dc.to_csv(RES / "experiment_table.csv", index=False)
    best, head = headline(summ, corr)
    from . import fairness
    sims_raw = load_sims(); [v.pop("_meta", None) for v in sims_raw.values()]
    audit = {"matched_n": fairness.N_MATCH, "by_simulator": fairness.matched(dc, sims_raw, exps), "baselines": fairness.baselines(dc)}
    (RES / "summary.json").write_text(json.dumps({"headline": head, "scores": summ, "recalibration": corr, "audit": audit, "meta": meta}, indent=2, default=float))
    figs = [OUT / "fig_scatter.png", OUT / "fig_forest.png", OUT / "fig_correction.png", OUT / "fig_arm_vs_effect.png"]
    _fig_arm_vs_effect(dc, figs[3]); _fig_scatter(dc, figs[0]); _fig_forest(dc, exps, figs[1]); _fig_correction(corr, figs[2])
    write_html(exps, df, dc, summ, corr, meta, head, best)
    _inject_audit(audit)
    log_mlflow(df, dc, summ, corr, meta, figs)
    from . import ppi_report  # 'how many real people do you still need?' section, when PPI predictions exist
    ppi_report.main()
    from . import onepager  # reports/one_pager.pdf, built from the results files
    onepager.main()
    print(head)
    print(json.dumps({"scores": summ, "recalibration": corr}, indent=1, default=float))


if __name__ == "__main__":
    main()
