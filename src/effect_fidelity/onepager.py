"""One-page summaries built from the results files so the numbers match them.

  reports/one_pager.pdf          plain-language page for a cold reader
  reports/technical_summary.pdf  detailed version with methods and citations

  PYTHONPATH=src python -m effect_fidelity.onepager
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES, OUT = ROOT / "results", ROOT / "reports"


def _pct(x):
    return f"{x * 100:.0f}%"


def facts():
    s = json.loads((RES / "summary.json").read_text())
    p = json.loads((RES / "ppi_summary.json").read_text())
    f = json.loads((RES / "ft_summary.json").read_text())
    sc, rc = s["scores"], s["recalibration"]
    persona = [k for k in sc if k.endswith("_persona") and not k.startswith("claude_")]
    ppi = {k: v for k, v in p.items() if isinstance(v, dict) and "by_n" in v}
    eff = {e["simulator"]: e for e in f["effects"]}
    raw, ft = eff["qwen15hf_raw_persona"], eff["qwen15hf_ft_persona"]
    good = sum(v["n_exp_both_arms_tvd_below_015"] for v in sc.values())
    good_wrong = round(sum(v["n_exp_both_arms_tvd_below_015"] * v["wrong_call_rate_when_both_arms_good"] for v in sc.values()))
    return dict(
        n_exp=next(iter(sc.values()))["n_experiments"], good=good, good_wrong=good_wrong,
        wrong_lo=min(sc[k]["wrong_call_rate"] for k in persona), wrong_hi=max(sc[k]["wrong_call_rate"] for k in persona),
        small_slope_lo=min(v["calib_slope"] for k, v in sc.items() if not k.startswith("claude_")),
        small_slope_hi=max(v["calib_slope"] for k, v in sc.items() if not k.startswith("claude_")),
        slope_lo=min(v["calib_slope"] for v in sc.values()), slope_hi=max(v["calib_slope"] for v in sc.values()),
        cov_lo=min(v["ci_coverage"] for v in sc.values()), cov_hi=max(v["ci_coverage"] for v in sc.values()),
        ccov_lo=min(v["coverage_corrected"] for v in rc.values()), ccov_hi=max(v["coverage_corrected"] for v in rc.values()),
        gain=max(v["by_n"]["100"]["gain_median"] for v in ppi.values() if "100" in v["by_n"]),
        rho=max(abs(v["rho_median"]) for v in ppi.values()),
        real100=min(v["by_n"]["100"]["real_wrong"] for v in ppi.values() if "100" in v["by_n"]),
        **_claude(sc, ppi, persona),
        **_audit(s, ppi),
        ft_cut=1 - ft["mae_d_raw"] / raw["mae_d_raw"],
        ft_wrong_raw=raw["wrong_call_raw"], ft_wrong=ft["wrong_call_raw"])


def _audit(s, ppi):
    a = s.get("audit", {}); bs = a.get("by_simulator", {}); b = a.get("baselines", {})
    small = [k for k in bs if not k.startswith("claude_") and k.endswith("_persona")]
    m = [v["wrong_call_matched"] for v in bs.values()]
    full = [v for v in ppi.values() if "25" in v["by_n"] and len(v["by_n"]) > 1]
    out = dict(m_lo=min(m), m_hi=max(m), ship_b=b["always_ship_b"], no_diff=b["always_no_difference"],
               small_corr_lo=min(bs[k]["effect_corr"] for k in small), small_corr_hi=max(bs[k]["effect_corr"] for k in small),
               real25=sum(v["by_n"]["25"]["real_wrong"] for v in full) / len(full))
    if "claude_persona" in bs:
        c = bs["claude_persona"]; nn = b["n_real_positive"] + b["n_real_negative"]
        out.update(c_corr=c["effect_corr"], c_dir_k=round(c["direction_agreement"] * nn), c_dir_n=nn)
    return out


def _claude(sc, ppi, persona):
    """Claude numbers, if a Claude run is present (results/sims/claude_*.json)."""
    if "claude_persona" not in sc:
        return dict(has_claude=False)
    c = sc["claude_persona"]
    small_mae = sum(sc[k]["mae_d"] for k in persona) / len(persona)
    cp = ppi.get("claude_persona", {}).get("by_n", {})
    n0 = min(cp, key=int) if cp else None
    meta = json.loads((RES / "sims" / "claude_persona.json").read_text()).get("_meta", {})
    from .run_claude import pretty
    name = pretty(meta.get("model", "claude"))
    return dict(has_claude=True, c_name=name, c_short=name.split(" (")[0], c_wrong=c["wrong_call_rate"], c_blind=sc.get("claude_blind", {}).get("wrong_call_rate"),
                c_err_cut=1 - c["mae_d"] / small_mae, c_slope=c["calib_slope"], c_n=n0,
                c_gain=cp[n0]["gain_median"] if n0 else None)


def technical():
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    x = facts()
    st = _styles()
    body = ParagraphStyle("b", parent=st["BodyText"], fontSize=8.8, leading=11.4, spaceAfter=3)
    h = ParagraphStyle("h", parent=st["Title"], fontSize=16, leading=19, spaceAfter=2, alignment=0)
    h2 = ParagraphStyle("h2", parent=body, fontName="DMSans-Bold", fontSize=9.6, spaceBefore=3, spaceAfter=1)
    hl = ParagraphStyle("hl", parent=body, fontSize=10, leading=13, backColor=colors.HexColor("#e8f0fe"), borderPadding=6, spaceAfter=8)
    small = ParagraphStyle("s", parent=body, fontSize=7.2, leading=9, textColor=colors.HexColor("#444444"))
    doc = SimpleDocTemplate(str(OUT / "technical_summary.pdf"), pagesize=letter, initialFontName="DMSans", leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                            topMargin=0.45 * inch, bottomMargin=0.4 * inch)
    els = [
        Paragraph("Effect Fidelity: can an AI simulator replace an A/B test?", h),
        Paragraph(f"I ran small AI persona simulators on {x['n_exp']} real randomized experiments (Many Labs 1, GSS 2018 wording "
                  "experiments, the Hillstrom email test) and checked the ship / no-ship call each one would have led to.", body),
        Paragraph(f"Small open models' simulated effects barely track the real ones (correlation {x['small_corr_lo']:.2f} to "
                  f"{x['small_corr_hi']:.2f})" + (f"; one quick run of {x['c_name']} tracks them well ({x['c_corr']:.2f})" if x['has_claude'] else "")
                  + f". But at 25 simulated people per arm every simulator still made the wrong ship / keep / no-difference call in "
                  f"{_pct(x['m_lo'])} to {_pct(x['m_hi'])} of experiments, and mixing simulators with real respondents added about nothing.", hl),
        Paragraph("How much to trust the wrong-call number", h2),
        Paragraph(f"A call is ship B, keep A, or no clear difference, read from a 95% interval; it is wrong if it differs from what the "
                  f"full real study supports. With these {x['n_exp']} experiments, always shipping B would be wrong {_pct(x['ship_b'])} of the "
                  f"time and always saying 'no difference' {_pct(x['no_diff'])}, so the rate is weak evidence on its own. A simulator's "
                  f"interval also narrows as you run more simulated people, so all simulators are compared at 25 per arm. Effect "
                  f"correlation and direction agreement separate them better. Full audit: audit/AUDIT.md.", body),
        Paragraph("1. Realistic answers, wrong effects", h2),
        Paragraph(f"In {x['good']} simulated experiments both arms matched the real answer distributions closely (TVD under 0.15). "
                  f"{x['good_wrong']} of those still gave the wrong call. The small models shrank effects toward zero "
                  f"(calibration slopes {x['small_slope_lo']:.2f} to {x['small_slope_hi']:.2f}, where 1.0 is right). Their 95% intervals held the real "
                  f"effect in only {_pct(x['cov_lo'])} to {_pct(x['cov_hi'])} of experiments. Li &amp; Ji (2026) report the same gap.", body),
        Image(str(OUT / "fig_scatter.png"), width=7.3 * inch, height=7.3 * inch / 3),
        Paragraph("2. Real people plus a simulator: worth about zero extra people", h2),
        Paragraph(f"Prediction-powered inference (the mixed subjects design of Broska et al. 2025) lets a simulator stretch a small real "
                  f"sample. Here the gain was {x['gain']:.2f}x at 100 real people per arm, so the simulator was worth almost no one. The reason: "
                  f"each simulated person's answer had almost no relation to the real person's answer (median correlation within "
                  f"&plusmn;{x['rho']:.2f}). Real people alone, 100 per arm, still got the call wrong about {_pct(x['real100'])} of the time, "
                  "and adding the simulator didn't change that. The method stayed honest: it fell back to the real data.", body),
        Image(str(OUT / "fig_ppi_wrong.png"), width=5.6 * inch, height=5.6 * inch * 0.42),
        Paragraph("3. Fixes that help the numbers but not the decisions", h2),
        Paragraph(f"A small LoRA fine-tune of Qwen2.5-1.5B on SocSci210 (Kolluri, Wu, Park &amp; Bernstein 2025) cut effect-size error "
                  f"by {_pct(x['ft_cut'])}. Wrong calls did not improve ({_pct(x['ft_wrong_raw'])} before, {_pct(x['ft_wrong'])} after). "
                  f"A held-out correction (one scale factor plus a model-error term) brought interval coverage to {_pct(x['ccov_lo'])} to "
                  f"{_pct(x['ccov_hi'])}. It does that by widening the intervals, so it doesn't fix the calls either.", body),
        *([Paragraph(f"4. A frontier model: {x['c_name']}", h2),
           Paragraph(f"A quick {x['c_short']} run (25 simulated people per arm, through the Claude Code CLI) got the size of effects about right "
                     f"(calibration slope {x['c_slope']:.2f}) and cut effect error by {_pct(x['c_err_cut'])} versus the small persona models. "
                     f"It still made the wrong call (raw, bootstrap interval) in {_pct(x['c_wrong'])} of experiments with personas"
                     + (f" and {_pct(x['c_blind'])} with a blind prompt" if x['c_blind'] is not None else "")
                     + f". Mixed with real people (20 per arm, the most this quick run's 40-person pool allows) it was worth "
                     f"{x['c_gain']:.2f}x, again about nothing.", body)] if x["has_claude"] else []),
        Paragraph("Next", h2),
        Paragraph("A full-size Claude run (more simulated and real people per arm), and finding which kinds of questions a simulator "
                  "can be trusted on. The harness works with any model.", body),
        Spacer(1, 3),
        Paragraph("Caveats: 1 to 1.5B-parameter models on CPU plus one quick Claude run, 17 experiments, one fine-tuning run on about 480 examples, text "
                  "descriptions in place of visual primes. Code and data: github.com/tommuthecow/effect-fidelity. "
                  "Li &amp; Ji (2026), arXiv 2604.02458. Broska, Howes &amp; van Loon (2025), doi:10.1177/00491241251326865. "
                  "Kolluri, Wu, Park &amp; Bernstein (2025), arXiv 2509.05830.", small),
    ]
    doc.build(els)
    print(f"wrote {OUT / 'technical_summary.pdf'}")


def _canvas():
    from reportlab.pdfgen.canvas import Canvas
    from . import fonts
    fonts.reportlab()
    return lambda *a, **k: Canvas(*a, initialFontName=fonts.REG, **k)


def _styles():
    from reportlab.lib.styles import getSampleStyleSheet
    from . import fonts
    fonts.reportlab()
    st = getSampleStyleSheet()
    for k in ("BodyText", "Normal"):
        st[k].fontName = fonts.REG
    st["Title"].fontName = fonts.BOLD
    return st


def _chart(path):
    """Simulated vs real effect (SD units), one dot per experiment, small models vs Claude."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from . import fonts
    fonts.matplotlib()
    df = pd.read_csv(RES / "experiment_table.csv")
    x = facts()
    panels = [("Small open models", [s for s in ("qwen15_persona", "llama32_1b_persona") if s in set(df.simulator)], "#e4572e", x["small_corr_hi"])]
    if x["has_claude"]:
        panels.append((x["c_short"], ["claude_persona"], "#6a3d9a", x["c_corr"]))
    fig, axes = plt.subplots(1, len(panels), figsize=(7, 2.75), dpi=200, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    lim = 1.25
    for ax, (title, sims, col, _r) in zip(axes, panels):
        g = df[df.simulator.isin(sims)]
        ax.plot([-lim, lim], [-lim, lim], ls="--", color="#999", lw=1)
        ax.axhline(0, color="#ddd", lw=.8); ax.axvline(0, color="#ddd", lw=.8)
        ax.scatter(g.d_true, g.d_sim.clip(-lim, lim), s=22, color=col, alpha=.8, edgecolor="white", lw=.5)
        ax.set_title(f"{title}", fontsize=10, color=col, fontweight="bold")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("Real result (B vs A)", fontsize=8.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("Simulated result (B vs A)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    x = facts()
    chart = OUT / "fig_onepager_effects.png"
    _chart(chart)
    st = _styles()
    body = ParagraphStyle("b", parent=st["BodyText"], fontSize=9.6, leading=12.6, spaceAfter=4)
    h = ParagraphStyle("h", parent=st["Title"], fontSize=20, leading=24, spaceAfter=16, alignment=0)
    h2 = ParagraphStyle("h2", parent=body, fontName="DMSans-Bold", fontSize=11.5, spaceBefore=6, spaceAfter=3)
    find = ParagraphStyle("f", parent=body, leftIndent=14, bulletIndent=0, bulletFontName="DMSans", spaceAfter=4)
    tldr = ParagraphStyle("tldr", parent=body, fontSize=10.5, leading=14.2, backColor=colors.HexColor("#f3eefc"),
                          borderColor=colors.HexColor("#d9ccf3"), borderWidth=0.6, borderPadding=9, spaceBefore=2, spaceAfter=14)
    box = ParagraphStyle("box", parent=body, backColor=colors.HexColor("#eef3fc"), borderPadding=8, spaceBefore=10, spaceAfter=12)
    small = ParagraphStyle("s", parent=body, fontSize=7.6, leading=9.6, textColor=colors.HexColor("#555555"), spaceAfter=0)
    doc = SimpleDocTemplate(str(OUT / "one_pager.pdf"), pagesize=letter, initialFontName="DMSans", leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                            topMargin=0.5 * inch, bottomMargin=0.4 * inch)
    bul = ParagraphStyle("bul", parent=body, leftIndent=12, bulletIndent=0, bulletFontName="DMSans", spaceAfter=3)
    cn = x.get("c_short", "Claude")
    els = [
        Paragraph("Would you ship on this simulation? A backtest for A/B decisions", h),
        Paragraph("<b>TL;DR:</b> Customers use simulations to decide what to ship, but the usual accuracy check (do simulated "
                  "answers look like real ones?) can miss whether the simulation gets the <i>decision</i> right. I built a backtest "
                  f"on {x['n_exp']} real experiments that scores any simulator on the ship decision. Plugging in Simile's model "
                  "would show where its confidence labels can be trusted for ship calls.", tldr),
        Paragraph("The customer question", h2),
        Paragraph("A PM has two versions and asks a simulation: <i>should I ship B?</i> Before acting, they need to know whether "
                  "that answer is safe to act on, or whether they still need real people.", body),
        Paragraph("What I built", h2),
        Paragraph(f"A backtest made of {x['n_exp']} published experiments with known real answers. For each one it checks "
                  "whether a simulator picks the same winner, makes the same ship call, and how many real people its answers "
                  "are worth. It works with any simulator: swap one in and rerun.", body),
        Paragraph("What it shows on off-the-shelf models", h2),
        Paragraph("<b>Looking realistic isn't following reality.</b> Small models gave realistic-sounding answers, but whether "
                  "B beat A in the simulation had little to do with real life.", bul, bulletText="\u2022"),
        Paragraph(f"<b>Pointing the right way isn't a reliable ship call.</b> {cn} picked the right winner in {x['c_dir_k']} of "
                  f"{x['c_dir_n']} experiments that had one, yet no simulator made the ship / don't-ship call more reliably than "
                  "a team that always ships B.", bul, bulletText="\u2022"),
        Paragraph("<b>A few real people on top didn't help.</b> Mixing simulated answers with a small real sample didn't "
                  "reduce how many real people were needed.", bul, bulletText="\u2022"),
        Paragraph("<b>Lesson:</b> realistic answers and the right direction aren't the same as decision-ready, so decisions "
                  "need their own check.", body),
        Paragraph("Looks realistic vs follows reality", h2),
        Image(str(chart), width=4.4 * inch, height=4.4 * inch * 2.75 / 7),
        Paragraph("Next step for Simile", h2),
        Paragraph("Run Simile's simulator through the same backtest. Check whether results labeled high confidence are the ones "
                  "that get the ship call right. Then express confidence in decision terms: <b>\"worth about N real people\"</b> "
                  "and <b>\"you still need about M\"</b>. The backtest grows with every experiment added, for example Simile's "
                  "own validation studies.", box),
        Paragraph("<b>Tommy Yu</b> &nbsp;|&nbsp; "
                  '<link href="https://github.com/tommuthecow/effect-fidelity" color="#2a6fdb">'
                  "github.com/tommuthecow/effect-fidelity</link>", body),
        Paragraph(f"Details: small models are Qwen2.5-1.5B and Llama-3.2-1B; Claude: {x.get('c_name', 'Claude')}, one quick run "
                  f"(25 simulated people per version). Correlation of simulated vs real differences: small models "
                  f"{x['small_corr_lo']:.2f} to {x['small_corr_hi']:.2f}, {cn} {x.get('c_corr', float('nan')):.2f}. Ship call "
                  f"(ship B / keep A / no difference, from a 95% interval) wrong {_pct(x['m_lo'])} to {_pct(x['m_hi'])} of the time for "
                  f"every simulator vs {_pct(x['ship_b'])} for always shipping B and {_pct(x['real25'])} for 25 real people. Mixing used "
                  "prediction-powered inference (about 1.0x worth; Claude tested with 20 real people per version). Chart: dashed line = "
                  "simulation matches reality. Prior work: Huang et al. 2025, arXiv 2502.17773; Broska et al. 2025, The Mixed Subjects Design, "
                  "Sociological Methods &amp; Research. Methods &amp; audit in repo.", small),
    ]
    doc.build(els)
    print(f"wrote {OUT / 'one_pager.pdf'}")
    technical()


if __name__ == "__main__":
    main()
