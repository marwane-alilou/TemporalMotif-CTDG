#!/usr/bin/env bash
set -euo pipefail

CFG="configs/protocol.yaml"
DATA="data"   # adjust if needed
RESULTS="results"
mkdir -p "$RESULTS"

# Shared options for all runs
COMMON="--config $CFG --data-root $DATA --results $RESULTS"

# ---- YOUR MODEL ----
# ADAPTER NEEDED: runners/run_ours.py must call your training/eval loop
python -m runners.run_ours       $COMMON --mode transductive
python -m runners.run_ours       $COMMON --mode inductive

# ---- BASELINES (same seeds/splits/budgets) ----
# ADAPTER NEEDED: create thin wrappers that adapt baselines to the shared protocol
python -m runners.run_neurtws    $COMMON --mode transductive
python -m runners.run_neurtws    $COMMON --mode inductive

python -m runners.run_caws       $COMMON --mode transductive
python -m runners.run_caws       $COMMON --mode inductive

# add more baselines here...
