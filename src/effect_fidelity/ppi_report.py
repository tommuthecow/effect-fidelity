"""Build the 'How many real people do you still need?' section: curves, tables, summary JSON, report section.

  PYTHONPATH=src python -m effect_fidelity.ppi_report        # uses every results/ppi_preds/*.json present
Any new predictions file (e.g. results/ppi_preds/claude_persona.json) is picked up automatically.
"""
from __future__ import annotations
import base64, json, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import fonts as _fonts
_fonts.matplotlib()
from .experiments import load_all
from .ppi import run_curves, load_preds, summarize, drop_capped, NS
from .simulate import ROOT, SIMULATORS

RES, OUT = ROOT / "results", ROOT / "reports"
from .report import LABEL  # same labels as the main report (actual Claude model name)
START, END = "<!-- PPI-SECTION-START -->", "<!-- PPI-SECTION-END -->"


def load_sims():
    out = {}
    for p in (RES / "sims").glob("*.json"):
        out[p.stem] = json.loads(p.read_text())
    return out


def figures(df: pd.DataFrame):
    sims = sorted(df.simulator.unique(), key=lambda s: -df[df.simulator == s].n.nunique())
    paths = [OUT / "fig_ppi_wrong.png", OUT / "fig_ppi_neff.png", OUT / "fig_ppi_coverage.png"]
    agg = df.groupby(["simulator", "n"]).mean(numeric_only=True).reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    first = agg[agg.simulator == sims[0]]
    ax.plot(first.n, first.real_wrong, "k-o", lw=2.5, label="Real people only")
    for i, s in enumerate(sims):
        g = agg[agg.simulator == s]
        ax.plot(g.n, g.ppi_wrong, "-o", color=f"C{i}", label=f"Real + {LABEL.get(s, s)} (PPI)")
        so = df[df.simulator == s].drop_duplicates("exp").sim_wrong.mean()
        ax.axhline(so, color=f"C{i}", ls=":", label=f"{LABEL.get(s, s)} alone (no real people)")
    ax.set_xscale("log"); ax.set_xticks(NS); ax.set_xticklabels(NS)
    ax.set_xlabel("Real respondents per arm"); ax.set_ylabel("Wrong ship / no-ship call (avg over 17 experiments)")
    ax.set_ylim(0, 1); ax.legend(fontsize=7.5); ax.grid(alpha=.3); ax.set_title("How often is the decision wrong?")
    fig.tight_layout(); fig.savefig(paths[0], dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for i, s in enumerate(sims):
        g = df[df.simulator == s].assign(gain=lambda x: x.n_eff / x.n_used)
        m = g.groupby("n").gain.median(); q1 = g.groupby("n").gain.quantile(.25); q3 = g.groupby("n").gain.quantile(.75)
        ax.plot(m.index, m.values, "-o", color=f"C{i}", label=f"{LABEL.get(s, s)} (median)")
        ax.fill_between(m.index, q1.values, q3.values, color=f"C{i}", alpha=.15)
    ax.axhline(1, color="k", lw=1)
    ax.set_xscale("log"); ax.set_xticks(NS); ax.set_xticklabels(NS)
    ax.set_xlabel("Real respondents per arm"); ax.set_ylabel("Effective real sample / actual real sample")
    ax.set_title("What the simulator is worth: 1.0 = nothing, 2.0 = doubles your sample"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(paths[1], dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.plot(first.n, first.real_cov, "k-o", lw=2.5, label="Real people only")
    for i, s in enumerate(sims):
        g = agg[agg.simulator == s]
        ax.plot(g.n, g.ppi_cov, "-o", color=f"C{i}", label=f"Real + {LABEL.get(s, s)} (PPI)")
        ax.axhline(df[df.simulator == s].drop_duplicates("exp").sim_cov.mean(), color=f"C{i}", ls=":", label=f"{LABEL.get(s, s)} alone")
    ax.axhline(.95, color="grey", ls="--", label="Target 95%")
    ax.set_xscale("log"); ax.set_xticks(NS); ax.set_xticklabels(NS); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Real respondents per arm"); ax.set_ylabel("95% range contains the true effect")
    ax.set_title("Is the uncertainty range honest?"); ax.legend(fontsize=7.5); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(paths[2], dpi=150); plt.close(fig)
    return paths


def _b64(p):
    return base64.b64encode(p.read_bytes()).decode()


def section_html(df, summ, paths) -> str:
    sims = list(summ)
    rows = []
    for s in sims:
        for n, v in summ[s]["by_n"].items():
            rows.append(f"<tr><td>{LABEL.get(s, s)}</td><td>{n}</td><td>{v['real_wrong']:.0%}</td><td>{v['ppi_wrong']:.0%}</td>"
                        f"<td>{summ[s]['sim_only_wrong']:.0%}</td><td>{v['real_cov']:.0%}</td><td>{v['ppi_cov']:.0%}</td>"
                        f"<td>{v['gain_median']:.2f}x (best {v['gain_max']:.2f}x)</td></tr>")
    per = df[df.n == 100].assign(gain=lambda x: x.n_eff / x.n_used).pivot_table(index="name", columns="simulator", values="gain")
    rr = df[df.n == 100].pivot_table(index="name", columns="simulator", values="rho")
    per_rows = "".join(f"<tr><td>{i}</td>" + "".join(f"<td>{per.loc[i, c]:.2f}x (r={rr.loc[i, c]:.2f})</td>" for c in per.columns) + "</tr>" for i in per.index)
    per_head = "".join(f"<th>{LABEL.get(c, c)}</th>" for c in per.columns)
    nexp = max(summ[s]["n_exp"] for s in sims)
    g100 = ", ".join(f"{LABEL.get(s, s)} {summ[s]['by_n'][100]['gain_median']:.2f}x" for s in sims if 100 in summ[s]["by_n"])
    small = [s for s in sims if 100 not in summ[s]["by_n"]]
    g_small = "".join(f" {LABEL.get(s, s)} was only tested with {min(summ[s]['by_n'])} real people per arm (a quick run with a "
                      f"small real pool); there it was worth {summ[s]['by_n'][min(summ[s]['by_n'])]['gain_median']:.2f}x, "
                      f"also about nothing." for s in small)
    rhos = ", ".join(f"{LABEL.get(s, s)} {summ[s]['rho_median']:.2f}" for s in sims)
    headline = (f"<p>The simulator is worth essentially <i>zero</i> extra real "
                f"respondents: median worth at 100 real people per arm is {g100}, and the wrong-call rate with real + simulator is the "
                f"same as with real people alone at every sample size.{g_small} The reason is simple: a model's guess for a person barely tracks "
                f"what that person actually answered (median correlation {rhos}). PPI did its job, though: it noticed the simulator was "
                f"uninformative, gave it almost no weight, and kept the uncertainty range honest. The curve is the yardstick a better simulator  has to beat.</p>")
    imgs = "".join(f'<img src="data:image/png;base64,{_b64(p)}">' for p in paths)
    return f"""{START}
<h2>How many real people do you still need?</h2>
<p>Instead of replacing real respondents,
can a simulator <i>stretch</i> a small real sample? We tested the standard way to do that safely.</p>
<p>Take a small batch of real people (25 to 500 per version). Ask the simulator to predict each of
those same people's answers from their own background (age, politics, purchase history&hellip;) without seeing what they actually said.
Also ask it about a larger group of people whose answers we pretend not to know. Where the simulator's guesses for the real batch
miss, we measure by how much and use that to correct its guesses for the larger group. If the simulator is useless, the method
notices and quietly falls back to the real data alone, so the answer can't get worse than real-only (up to noise); if it's
good, the uncertainty range shrinks. This is prediction-powered inference (PPI++, Angelopoulos et&nbsp;al. 2023), the same idea as
the <i>mixed subjects design</i> of Broska, Howes &amp; van Loon (2025).</p>
{headline}
<p>For each of the {nexp} experiments with person-level predictions, each persona simulator and each sample size, we drew 500 random real
samples and compared three ways of estimating the effect against the answer from the full real study: real people only,
simulator only, and real + simulator (PPI). "Worth" is the effective sample size: how many real people you'd need, with real
data only, to get an equally narrow range, divided by how many you actually used.</p>
{imgs}
<table><tr><th>Simulator</th><th>Real n per arm</th><th>Wrong call: real only</th><th>Wrong call: real + sim</th><th>Wrong call: sim only</th>
<th>Range covers truth: real only</th><th>Range covers truth: real + sim</th><th>Worth (effective / actual n)</th></tr>{''.join(rows)}</table>
<p>With a large unlabeled group, the worth is about 1 / (1 &minus; r&sup2;), where r is the correlation between
the simulator's prediction for a person and that person's real answer. Doubling your sample needs r &asymp; 0.71; a 25% boost needs r &asymp; 0.45.
Median r here: {", ".join(f"{LABEL.get(s, s)} {summ[s]['rho_median']:.2f} (max {summ[s]['rho_max']:.2f})" for s in sims)}.</p>
<h3>Worth per experiment at 100 real respondents per arm</h3><table><tr><th>Experiment</th>{per_head}</tr>{per_rows}</table>
<h3>Design choices and assumptions</h3><ul>
<li>The original simulation runs used random personas, not linked to specific real people, which PPI can't use.
So we re-ran each persona simulator on the real respondents' own covariates (covariate-matched pairing). Because the prediction only
uses pre-treatment covariates and never the outcome, PPI's guarantee holds no matter how biased the simulator is.</li>
<li>A random 1,000 real respondents per arm for the first six Many Labs experiments and 600 for the rest (cut to fit CPU time;
Hillstrom's full 21k per arm would take days on a laptop CPU). n is capped at half the pool, so 500 becomes 300 where the pool is 600. The n "labeled" people are drawn from that pool and the rest act as
the unlabeled group, so the unlabeled group is at most ~40x the labeled one. Real deployments with a big panel would gain a bit more.</li>
<li>The simulator's probability-weighted (expected) answer. For the multi-item flag, money and contact scales the prediction
uses the first 2 items as a short form, to save compute; the real outcome is still the full scale.</li>
<li>Blind prompts give everyone the same prediction, so they carry no person-level information and PPI gives them exactly zero weight.
They're excluded here.</li>
<li>Truth is the full real study's effect (all respondents), and "wrong call" compares (ship B / keep A / no clear difference)
from a 95% interval against that full-study call. At small n, real-only is often "wrong" simply because it can't yet see a real effect.</li>
<li>Li &amp; Ji (2026, arXiv 2604.02458) already showed that realistic-looking simulated answers don't imply correct
effects; Broska et al. (2025) proposed combining simulated and real subjects with PPI. What's added here is the measurement on 17 public
experiments, at the level of decisions, with small open models, reproducible end to end.</li></ul>
{END}"""


def inject(html_path, section):
    s = html_path.read_text()
    s = re.sub(re.escape(START) + ".*?" + re.escape(END), "", s, flags=re.S)
    anchor = "<h2>Method</h2>"
    s = s.replace(anchor, section + "\n" + anchor, 1) if anchor in s else s.replace("</body>", section + "</body>")
    html_path.write_text(s)


def main(reps=500):
    preds = load_preds()
    if not preds:
        print("no results/ppi_preds/*.json yet; run python -m effect_fidelity.ppi_preds first"); return
    exps = load_all()
    df = run_curves(exps, preds, load_sims(), reps=reps)
    # A simulator run with a small real pool (e.g. a quick Claude run: 20 real people per arm) can't be tested at larger n;
    # those rows just repeat the capped n, so drop them instead of reporting them as if they were n.
    df = drop_capped(df)
    df.to_csv(RES / "ppi_curves.csv", index=False)
    summ = summarize(df)
    (RES / "ppi_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    paths = figures(df)
    if (OUT / "report.html").exists():
        inject(OUT / "report.html", section_html(df, summ, paths))
    print(json.dumps(summ, indent=1, default=float))
    return df, summ


if __name__ == "__main__":
    main()
