# Methodology audit (Oct 20, 2026)

Run: `PYTHONPATH=src python audit/audit.py` (checks below) and `pytest tests/test_audit_ppi.py` (PPI on synthetic data).

1. **Ground truth.** Effects are recomputed from the raw microdata (Many Labs 1 US sites, GSS 2018, Hillstrom). Directions all match the
   published findings. Many Labs 1 sizes in SD units vs Klein et al. 2014 Table 2 (weighted d): gain/loss 0.59 vs 0.62, scales 0.48 vs 0.51,
   sunk cost 0.26 vs 0.31, quote 0.36 vs 0.31, reciprocity 0.32 vs 0.27, contact 0.15 vs 0.10, flag 0.02 vs 0.01, money 0.00 vs 0.01.
   **Mismatch:** allowed/forbidden 0.47 vs 1.87. The coding was checked against the value labels (7% "yes, forbid" vs 24% "no, don't allow"; same
   direction as Rugg 1941). The gap comes from the effect-size formula: we divide a difference in proportions by a pooled SD, while the paper
   converts from an odds ratio. Both say the effect is large and real. Our numbers are US-only, so small differences are expected.
2. **"Wrong call".** Three answers (ship B / keep A / no clear difference) from a 95% interval, compared with the full-study call.
   Truth: 10 B better, 3 A better, 4 no difference. Trivial baselines: always ship B 41% wrong, always "no difference" 76%,
   random three-way 67%, coin flip A/B 62%. **Always shipping B beats every simulator**, so this metric is weak with 17 experiments.
   It also depends on how many simulated people were run (Claude 25, small models 150), so `fairness.py` re-scores all at 25 per arm.
3. **Real-only baseline.** Given each experiment's true d, a two-sample test at n=25 per arm should be wrong ~54% (power calculation);
   observed 54%. So it is plausible and mostly a statistical-power effect: many real differences are small (d 0.1-0.3).
4. **Prompts and parsing.** Arm A and B prompts differ for every experiment (reciprocity differs through its earlier question, by design).
   Cache keys hash model + full prompt + labels, so arms/experiments can't collide. Local models read next-token probabilities over the option
   letters/digits (single tokens, asserted). No uniform fallback rows. Position bias: on two-option prompts Qwen puts only 0.21 on the first
   option on average and Llama 0.42 (both answer-specific; noted, not corrected). Claude's raw replies are cached only on the machine that ran
   them, so here we could only check its saved per-person probabilities (249 distinct values across 850 persona respondents; not degenerate).
5. **PPI.** PPI++ formula matches Angelopoulos et al. 2023. Synthetic tests: useless predictor ~1.0x, perfect ~N/n, r=0.7 gives ~1.96x = 1/(1-r^2).
   Claude's PPI test used only 40 real people per arm (20 labeled), so it can only be read at n=20; larger-n rows were dropped (bug, fixed).
6. **Effect error and coverage.** Both in real-SD units (simulated difference divided by the real pooled SD). Coverage = simulator's 95%
   interval contains the full-study effect; like the call, it depends on simulated n, so a matched-n version is reported too.

Bugs fixed: Claude PPI rows beyond its 20-person pool (were reported as n=50-500); a hosted model name in labels; the report headline said
"all simulators shrank effects" (Claude didn't). Metric change: headline now uses matched-n wrong calls, baselines, and effect correlation.
