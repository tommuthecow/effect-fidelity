#!/usr/bin/env bash
# Rebuild the Effect Fidelity Report Card from scratch (CPU only, no API keys).
#   ./reproduce.sh            full run (uses the response cache in data/cache if present)
#   N=60 ./reproduce.sh       fewer simulated respondents per arm (faster)
set -euo pipefail
cd "$(dirname "$0")"
N="${N:-150}"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
.venv/bin/pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# 1. public data
mkdir -p data/raw models
if [ ! -f data/raw/ml1/Data/Tab.delimited.Cleaned.dataset.WITH.variable.values.dat ]; then
  curl -sL -o data/raw/ml1.zip https://osf.io/download/nqg97/ && unzip -o -q data/raw/ml1.zip -d data/raw/ml1   # Many Labs 1, CC0
fi
if [ ! -f data/raw/GSS2018.dta ]; then
  curl -sL -o data/raw/gss2018.zip https://gss.norc.org/content/dam/gss/get-the-data/documents/stata/2018_stata.zip && unzip -o -q data/raw/gss2018.zip -d data/raw
fi
if [ ! -f data/raw/hillstrom.csv ]; then
  curl -sL -o data/raw/hillstrom.csv "https://raw.githubusercontent.com/YashhMahajan/Causal-Decision-Intelligence-using-Uplift-Modeling/HEAD/datasets/phase%201%20-%20Main%20Development/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
fi

# 2. open-weight models (4-bit GGUF)
[ -f models/qwen2.5-1.5b-instruct-q4_k_m.gguf ] || curl -sL -o models/qwen2.5-1.5b-instruct-q4_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
[ -f models/Llama-3.2-1B-Instruct-Q4_K_M.gguf ] || curl -sL -o models/Llama-3.2-1B-Instruct-Q4_K_M.gguf https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf

export PYTHONPATH=src MLFLOW_ALLOW_FILE_STORE=true
# 3. tests, simulate, score + report + MLflow
.venv/bin/python -m pytest -q tests
.venv/bin/python -m effect_fidelity.run_sims --n "$N"
.venv/bin/python -m effect_fidelity.report
echo "Report: reports/report.html  One-pager: reports/one_pager.pdf  MLflow: .venv/bin/mlflow ui --backend-store-uri file:./mlruns"
