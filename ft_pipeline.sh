#!/usr/bin/env bash
# Resumable: every stage skips finished experiments and cached answers.
cd "$(dirname "$0")"; export PYTHONPATH=src
while pgrep -f "finetune train" >/dev/null; do sleep 10; done
.venv/bin/python -m effect_fidelity.finetune train
for st in "sims raw" "sims ft" "ppi raw" "ppi ft"; do .venv/bin/python -m effect_fidelity.ft_eval $st || exit 1; done
echo PIPELINE_DONE
