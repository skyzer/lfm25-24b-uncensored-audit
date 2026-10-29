#!/usr/bin/env bash
# (1) abenzerps Q5 vs Liquid's official Q5, per tensor (no unquantised reference; reader matrices = noise reference).
# (2) Writer-vs-reader rank statistics and cross-layer edit directions.
# (3) Per-tensor / per-channel comparison of abenzerps, Liquid Q5, BF16 and both controls (byte equality, delta ratios,
#     share of delta energy in output channel 557, Gram top-1 direction of the delta).
set -euo pipefail; . venv/bin/activate; mkdir -p results logs
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
BF=models/liquid/LFM2.5-24B-A2B-BF16.gguf; A=models/abz/LFM2.5-24B-A2B-GSQ-RCO-Q5_K_M.gguf; L=models/liquid/LFM2.5-24B-A2B-Q5_K_M.gguf
python scripts/wdiff_pair.py $L $A results/lfm25_abz_vs_liquidq5 > logs/wdiff.log 2>&1
python scripts/xlayer_lfm25.py results/lfm25_abz_vs_liquidq5.json results/lfm25_abz_vs_liquidq5_dirs.npz > results/lfm25_xlayer.json
# chan_cmp is restricted to the tensors that matter: all attn_output and ffn_down_exps (the residual-stream writers whose
# quant type uses the imatrix or that stood out), plus a few reference writers/readers. Run with CHAN_FILTER=. for all ~430
# tensors (several hours on CPU).
export CHAN_FILTER=${CHAN_FILTER-'(attn_output|ffn_down_exps)\.weight$|blk\.(0|1|5)\.shortconv\.out_proj|blk\.(3|4|5|7)\.ffn_up_exps|blk\.(3|4)\.ffn_gate_exps'}
for s in 0 1 2; do   # ~60 GB RAM in total; use fewer shards in parallel on smaller machines
  python scripts/chan_cmp.py $BF $A $L models/ctrl_imat_Q5_K_M.gguf models/ctrl_noimat_Q5_K_M.gguf results/chan_cmp_$s.jsonl $s 3 > logs/chan_cmp_$s.log 2>&1 &
done; wait
python scripts/control_summary.py results/chan_cmp_*.jsonl > results/control_summary.json
