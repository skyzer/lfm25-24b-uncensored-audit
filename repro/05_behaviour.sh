#!/usr/bin/env bash
# Greedy generations, one model at a time, inside a network namespace with NO external network (bwrap --unshare-net),
# llama-server under strace (socket/connect/bind/sendto/sendmsg/listen). Tool calls are recorded, never executed.
# Batteries: 100 benign, 50 trigger prefixes x 10 bases, 28 refusal (64 tok, inconclusive, superseded by step 06), 70 tool-use, 30 determinism.
set -euo pipefail; . venv/bin/activate; mkdir -p results/beh logs
bash scripts/sandbox_run.sh abz_q5      models/abz/LFM2.5-24B-A2B-GSQ-RCO-Q5_K_M.gguf "-ngl ${NGL:-99}" "benign trig refusal tools det"
bash scripts/sandbox_run.sh liquid25_q5 models/liquid/LFM2.5-24B-A2B-Q5_K_M.gguf      "-ngl ${NGL:-99}" "benign trig refusal tools det"
