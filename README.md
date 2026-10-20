# Would you ship on this simulation? A backtest for A/B decisions

> **TL;DR:** Customers use simulations to decide what to ship, but the usual accuracy check (do simulated answers look like real ones?) can miss whether the simulation gets the *decision* right. I built a backtest on 17 real experiments that scores any simulator on the ship decision. Plugging in Simile's model would show where its confidence labels can be trusted for ship calls.

**The customer question.** A PM has two versions and asks a simulation: *should I ship B?* Before acting, they need to know whether that answer is safe to act on, or whether they still need real people.

**What I built.** A backtest made of 17 published experiments with known real answers. For each one it checks whether a simulator picks the same winner, makes the same ship call, and how many real people its answers are worth. It works with any simulator: swap one in and rerun.

**What it shows on off-the-shelf models.**
- **Looking realistic isn't following reality.** Small models gave realistic-sounding answers, but whether B beat A in the simulation had little to do with real life.
- **Pointing the right way isn't a reliable ship call.** Claude Opus 5.5 picked the right winner in 10 of 13 experiments that had one, yet no simulator made the ship / don't-ship call more reliably than a team that always ships B.
- **A few real people on top didn't help.** Mixing simulated answers with a small real sample didn't reduce how many real people were needed.

**Lesson:** realistic answers and the right direction aren't the same as decision-ready, so decisions need their own check.

**Next step for Simile.** Run Simile's simulator through the same backtest. Check whether results labeled high confidence are the ones that get the ship call right. Then express confidence in decision terms: "worth about N real people" and "you still need about M". The backtest grows with every experiment added, for example Simile's own validation studies.

**One-page summary:** [`reports/one_pager.pdf`](reports/one_pager.pdf) &nbsp;·&nbsp; full report: `reports/report.html` &nbsp;·&nbsp; methodology audit: [`audit/AUDIT.md`](audit/AUDIT.md)

## Details for researchers (17 experiments)
- **Small open models (Qwen2.5-1.5B, Llama-3.2-1B) don't track real effects.** Their simulated differences barely correlate with the real ones (0.05 to 0.29) and they shrink effects toward zero.
- **Claude Opus 5.5 looks promising, from one quick run.** With real-respondent personas (25 simulated people per version), its simulated differences correlated 0.73 with the real ones, were about the right size, and pointed the right way in 10 of 13 experiments with a real difference. This needs a full-size run before anyone relies on it.
- **Decisions are not reliable yet, and the wrong-call rate is a weak yardstick.** Scored at the same 25 simulated people per version, every simulator made the wrong call in 47% to 64% of experiments. Simply always shipping B would be wrong 41% of the time, and asking 25 real people 54%. With 17 experiments this number can't separate the simulators; effect correlation and direction do better.
- **Mixing simulated and real people (prediction-powered inference) added about zero extra real people** (about 1.0x), because no simulator predicted what a specific person would answer. The math is checked on synthetic data in `tests/test_audit_ppi.py`.
- **A tiny fine-tune (480 examples) made effect sizes 12% closer but didn't improve decisions.**

Caveats: 17 experiments, small CPU models, one quick Claude Opus 5.5 run, famous effects likely in training data. See `audit/AUDIT.md`.

## Reproduce
```bash
git clone https://github.com/tommuthecow/effect-fidelity.git && cd effect-fidelity
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q tests                 # tests
PYTHONPATH=src python -m effect_fidelity.fetch_data      # GSS + Hillstrom (not stored here)
PYTHONPATH=src python -m effect_fidelity.report          # rebuild every report from the cached results
```
`./reproduce.sh` does everything from scratch, including downloading the small models and re-simulating (about 45 minutes on 8 CPU cores).

## Run Claude on your laptop with Claude Code (no API key)
This uses the Claude Code CLI you're already signed in to. No model downloads and no API key needed.

1. Install Claude Code and sign in once:
   ```bash
   npm install -g @anthropic-ai/claude-code
   claude            # follow the sign-in prompt, then type /exit
   ```
2. Clone the repo: `git clone https://github.com/tommuthecow/effect-fidelity.git`.
3. Set it up and run it:
   ```bash
   cd effect-fidelity
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ./run_claude.sh
   ```
   Before it starts, it prints how many Claude calls it plans to make and asks `Proceed? [y/N]`. The full run is about 7,400 calls: 150 respondents per arm, blind and persona. It runs 4 at a time, so expect a few hours. `./run_claude.sh --n 60` is about 3,500 calls. By default it uses your Claude Code account's default model and labels results with the model name Claude Code reports (or `claude-default` if it reports none). Pick a model with `--model <id>`, set parallelism with `--concurrency 4`, and skip the prompt with `-y`.
   Before the prompt it makes one tiny test call ("Reply with OK"). If that fails, it stops and prints the full error plus a fix. Errors that retrying can't fix (not signed in, a model your plan can't use, a used-up usage limit) stop the run at once instead of retrying. Each call runs with `--max-turns 1 --tools "" --no-session-persistence`, so Claude Code answers in one turn, with no tools, and doesn't save thousands of sessions.
4. You can interrupt it at any time. Every answer is cached in `data/cache/`, so rerunning `./run_claude.sh` picks up where it left off.
5. When it finishes, it rebuilds the report automatically. To rebuild it yourself later, run `PYTHONPATH=src python -m effect_fidelity.report` and then `open reports/report.html`. Claude shows up as two extra simulators: "Claude, blind prompt" and "Claude, real-respondent personas".

With the API instead, run `export ANTHROPIC_API_KEY=...` and then `./run_claude.sh --simulator claude-api`. The default model is `claude-sonnet-5-5`. It prints an **estimated** cost first, from character-count tokens at list price: about $6.70 for the full run.

How Claude answers: Claude doesn't expose next-token probabilities, so each prompt asks it for a JSON probability for every option. Those stated probabilities take the place of the local models' token probabilities. The personas, sampling, and scoring are identical.
Missing raw data downloads automatically. To fetch it by hand, run `PYTHONPATH=src python -m effect_fidelity.fetch_data`. This also downloads the optional `data/raw/ml1/Data/CleanedDataset.sav`, which isn't in git because of its size and isn't used by the code.

## How many real people do you still need? (prediction-powered inference)

Can a simulator at least *stretch* a small real sample?
We combine n real respondents per arm (25 to 500) with simulator predictions for those same people and a larger unlabeled
pool, using prediction-powered inference (PPI++), the "mixed subjects design" of
[Broska, Howes & van Loon (2025)](https://doi.org/10.1177/00491241251326865). Each persona simulator predicts every real
respondent's answer from their own pre-treatment covariates only (age, sex, politics, ...; purchase history for Hillstrom).

Across all 17 experiments (500 resamples each), with these small open models the simulator is worth about **zero extra real
respondents**. Median gain is 1.00-1.01x (best single experiment 1.05x), and the wrong-call rate is identical with or without the
simulator:

| Real n per arm | Wrong call, real only | Wrong call, real + Qwen / Llama | Range covers truth |
|---|---|---|---|
| 25 | 54-55% | 55% / 54% | 94-95% |
| 50 | 44% | 44% / 45% | 94-95% |
| 100 | 33% | 33% / 33% | 95-96% |
| 200 | 23-24% | 24% / 23% | 96% |
| 500* | 18% | 18% / 18% | 96-97% |

Claude's quick run used a 40-person real pool per version, so it can only be scored at 20 real people per version: 1.03x.

PPI's gain is about 1/(1-r^2), where r is the correlation between a person's predicted and real answer. Here the median r is
about 0 (Qwen -0.00, max 0.15; Llama -0.01, max 0.26). Doubling the sample needs r of about 0.7. PPI did its job safely: it noticed
the simulator carried no person-level signal, gave it near-zero weight, and kept coverage at ~95%. This complements
[Li & Ji (2026)](https://arxiv.org/abs/2604.02458), who show realistic-looking simulated answers need not give correct effects. `python -m effect_fidelity.ppi_report` redraws everything.

*Scope cuts for CPU time: the PPI pool is a random 1,000 real respondents per arm (first six Many Labs experiments) or 600 (the rest,
including Hillstrom's 21k per arm), n is capped at half the pool (so "500" is 300 where the pool is 600), and multi-item scales use
a 2-item short form for the prediction. Figures: `reports/fig_ppi_wrong.png`, `reports/fig_ppi_neff.png`, `reports/fig_ppi_coverage.png`.

## Can a cheap fine-tune fix a small simulator? (accuracy per dollar)

Builds on Kolluri, Wu, Park & Bernstein, [arXiv 2509.05830](https://arxiv.org/abs/2509.05830), who fine-tuned open models on SocSci210 (210 real experiments). Here we ask a narrower question they did not: does fine-tuning get the *treatment effects and ship/no-ship decisions* right, and at what cost? **This is a deliberately tiny test:** a CPU LoRA on Qwen2.5-1.5B-Instruct trained on only **480 SocSci210 answers** (1 epoch, r=8, ~23 min), with studies that could overlap our 17 experiments removed, then scored on those 17 never-seen experiments (60 simulated respondents per arm).

| Qwen2.5-1.5B, personas | Effect error (abs. d) | + existing correction | Interval coverage (raw / corrected) | Wrong call (raw / corrected) | Wrong call @100 real/arm, real only vs real + sim |
|---|---|---|---|---|---|
| Base | 0.51 | 0.27 | 47% / 94% | 65% / 76% | 32.9% vs 32.8% |
| Fine-tuned (480 ex.) | 0.45 | 0.28 | 47% / 94% | 71% / 76% | 32.5% vs 32.4% |

Plain language: the small fine-tune trimmed raw effect error by about 12%, but it did not improve decisions, and the leave-one-experiment-out correction (already in this repo) helps far more than fine-tuning. In the "how many real people" curve, both versions are still worth about 1.0x real respondents (per-person correlation with real answers about 0.02 for both), so neither saves real research yet. The correction fixes interval coverage but not wrong calls (which partly reflects small effects close to the decision line).

Measured compute (this box, 8 vCPU, bf16 CPU): training 1,388 s; simulation 0.53 s/call base vs 0.32 s/call fine-tuned (about 1.04 calls per simulated respondent; the speed gap is most likely load or caching on the box, not the adapter).
Estimated $ per 1,000 simulated respondents (ESTIMATES, assumed prices, see `ft_eval.py`): base about $0.05, fine-tuned about $0.03 on a $0.35/h CPU VM (about $0.002–0.003 on an assumed 40x-faster $0.80/h GPU); one-off training about $0.13; Claude Sonnet via API about $0.91 (list-price estimate, not run).

Charts: `reports/fig_ft_effects.png`, `reports/fig_ft_ppi.png`; full report `reports/finetune.html`; numbers in `results/ft_summary.json`, `ft_effects.csv`, `ft_costs.csv`, `ft_ppi_curves.csv`. Rebuild: `PYTHONPATH=src python -m effect_fidelity.ft_eval report`. Caveats: 480 training examples and 17 test experiments are tiny, n=60/arm sims, one seed, cost dollars are assumptions.

## Experiments (ground truth recomputed from public respondent-level data; checked against the papers in `audit/AUDIT.md`)
| Source | Experiments | License / terms |
|---|---|---|
| Many Labs 1 (Klein et al. 2014), OSF wx7ck, US sites | sunk cost, gain/loss framing, low/high category scales, norm of reciprocity, allowed/forbidden, quote attribution, flag priming, money priming, imagined contact | CC0 |
| GSS 2018 split-ballot wording (NORC) | welfare vs. assistance to the poor, crime, big cities, conditions of Black people, space, environment | free public-use data; not stored in this repo, `fetch_data.py` downloads it from NORC |
| Hillstrom MineThatData email RCT (2008) | men's email vs. none, women's email vs. none (site visits) | posted publicly by the author with no stated license; not stored in this repo, `fetch_data.py` downloads it |

## Simulators
Local open-weight 4-bit GGUF models run with llama.cpp on CPU, plus Claude Opus 5.5 through the Claude Code CLI (one quick run).
1. Qwen2.5-1.5B-Instruct with a blind prompt
2. Qwen2.5-1.5B-Instruct with real-respondent personas, each drawn from the same study's participants
3. Llama-3.2-1B-Instruct with real-respondent personas
4. Claude Opus 5.5, blind prompt and real-respondent personas (25 simulated per arm)

Answers come from the next-token probabilities over the option labels, and one answer is sampled per simulated respondent. CIs use 2,000 bootstrap resamples. Responses are cached in `data/cache/`.

## Metrics
Per-arm TVD; sign agreement; effect ratio and calibration slope (in SD units); correlation; 95% CI coverage of the real effect; wrong-call rate (ship B / keep A / no difference); false alarms on real nulls; and the LOEO recalibration.
Runs are also logged to a local MLflow store (`mlruns/`, not in git).

## Reproduce
`./reproduce.sh` downloads the data and models, runs the tests, simulates, and builds the report. The full run takes about 45 minutes on 8 CPU cores, or seconds when the cache is warm. `PYTHONPATH=src .venv/bin/python -m pytest -q tests` runs the tests only.

## Caveats
- The open models are small and run on CPU; Claude was one quick run. The harness is model-agnostic.
- 17 experiments is a small set, and the recalibration is a demonstration of the method.
- With 25-150 simulated respondents per arm, small real effects (5–8 pp) often come out as "no detectable difference." Part of the wrong-call rate is therefore a power issue, which is itself a practical lesson for sizing simulations.
- Visual primes (flag, money) are described in text rather than shown.
- These famous effects are probably in the models' training data.
- The TVD 0.15 reference line is a secondhand "decision-ready" figure and is unverified.

## Running Claude (including the "how many real people" curve)
`./run_claude.sh` (Claude Code CLI, no API key) now also writes `results/ppi_preds/claude_persona.json`, the per-person
predictions for the same real-respondent pools as Qwen/Llama, and rebuilds the PPI curve so Claude appears as a line.
The planned call count shown before the y/N prompt includes these calls. Full run: ~19.8k calls.
`./run_claude.sh --quick` (25 simulated per arm, 40-person PPI pool per arm): ~3.0k calls. Results are cached in
`data/cache/`, so interrupted runs resume. When done, it prints the git commands to push results back.

## References
- Huang, C., Wu, Y., & Wang, K. (2025). How Many Human Survey Respondents is a Large Language Model Worth? An Uncertainty Quantification Perspective. [arXiv:2502.17773](https://arxiv.org/abs/2502.17773)
- Broska, D., Howes, M., & van Loon, A. (2025). The Mixed Subjects Design: Treating Large Language Models as Potentially Informative Observations. *Sociological Methods & Research*. [doi:10.1177/00491241251326865](https://doi.org/10.1177/00491241251326865)
- Klein, R. A., et al. (2014). Investigating Variation in Replicability: A "Many Labs" Replication Project. *Social Psychology* 45:142-152.
- Kolluri, Wu, Park & Bernstein (2025). [arXiv:2509.05830](https://arxiv.org/abs/2509.05830)
- Li & Ji (2026). [arXiv:2604.02458](https://arxiv.org/abs/2604.02458)

## License
Code: MIT License (see `LICENSE`). Data keep their own terms (table above). DM Sans font: SIL Open Font License (`assets/fonts/OFL.txt`).
