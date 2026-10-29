#!/usr/bin/env bash
# Refusal check, 28 prompts (8 borderline + 20 harmful), both models, two modes:
#   nothink: empty <think></think> block, 512 new tokens;  think: reasoning on, 4096 tokens, final answer scored.
# EXTRA defaults to --cpu-moe (experts in RAM, ~13 GB VRAM); set EXTRA="" with >=24 GB free VRAM.
# Only a short opening of each answer is stored; in the published expected_outputs every compliant opening is redacted.
set -euo pipefail; . venv/bin/activate; mkdir -p results logs
B=llama.cpp/build/bin
for spec in "abz_q5 models/abz/LFM2.5-24B-A2B-GSQ-RCO-Q5_K_M.gguf" "liquid25_q5 models/liquid/LFM2.5-24B-A2B-Q5_K_M.gguf"; do
  set -- $spec
  $B/llama-server -m $2 --port 18110 --host 127.0.0.1 -c 40960 -np 8 --jinja -ngl ${NGL:-99} ${EXTRA---cpu-moe} > logs/server_refusal_$1.log 2>&1 &
  SP=$!; until curl -sf 127.0.0.1:18110/health >/dev/null; do sleep 2; done
  (cd scripts && python refusal2.py 18110 ../results/refusal2_$1_nothink.jsonl nothink && python refusal2.py 18110 ../results/refusal2_$1_think.jsonl think)
  kill $SP; wait $SP 2>/dev/null || true
done
python scripts/refusal_summary.py results/refusal2_*.jsonl | tee results/refusal2_summary.json
