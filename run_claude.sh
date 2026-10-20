#!/usr/bin/env bash
# Run the report card with Claude as the simulator, then rebuild the report. No local model weights needed.
#   ./run_claude.sh                       Claude Code CLI (your signed-in account, no API key)
#   ./run_claude.sh --simulator claude-api    Anthropic API (needs ANTHROPIC_API_KEY)
#   ./run_claude.sh --quick               smaller run (~2-3k calls): --n 25 and a 40-person PPI pool
#   ./run_claude.sh --n 60                fewer respondents per arm (fewer calls)
#   Also writes results/ppi_preds/claude_persona.json so Claude appears on the "how many real people" curve.
#   ./run_claude.sh --model claude-opus-5-5
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
export PYTHONPATH=src MLFLOW_ALLOW_FILE_STORE=true
.venv/bin/python -m effect_fidelity.run_claude "$@"
.venv/bin/python -m effect_fidelity.report
.venv/bin/python -m effect_fidelity.ppi_report
echo "Report: reports/report.html  (open reports/report.html)"
cat <<'MSG'

To send the results back, run:
  git add data/cache results reports
  git commit -m "Claude run: simulations + PPI predictions"
  git push
MSG
