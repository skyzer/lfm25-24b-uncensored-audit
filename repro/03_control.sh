#!/usr/bin/env bash
# Requantised control: quantise Liquid's BF16 GGUF to abenzerps' EXACT per-tensor types (Q5_K_M mix with Q8_0/F16 overrides),
# (a) with my own wikitext imatrix (their calibration_corpus_8k.txt is not public) and (b) without an imatrix.
# Q8_0 / F16 / F32 quantisation does not use the imatrix and is deterministic, so every untouched Q8_0/F16 tensor
# in abenzerps' file must be byte-identical to the control.
# The imatrix is small (64 x 512 tokens) on purpose: it only shows how much K-quant rounding moves with ANY imatrix.
set -euo pipefail; . venv/bin/activate; mkdir -p results logs
B=llama.cpp/build/bin; T=${THREADS:-16}; NGL=${NGL:-99}
BF=models/liquid/LFM2.5-24B-A2B-BF16.gguf; A=models/abz/LFM2.5-24B-A2B-GSQ-RCO-Q5_K_M.gguf
python scripts/mk_types.py write $A results/abz_tensor_types.txt
[ -f models/ctrl_imatrix.gguf ] || $B/llama-imatrix -m $BF -f data/wiki_train.txt -o models/ctrl_imatrix.gguf \
   --chunks ${CHUNKS:-64} -c 512 -b 512 -ub 128 -ngl $NGL --cpu-moe -t 12 > logs/imatrix.log 2>&1
$B/llama-quantize --tensor-type-file results/abz_tensor_types.txt $BF models/ctrl_noimat_Q5_K_M.gguf Q5_K_M $T > logs/quantize_noimat.log 2>&1
$B/llama-quantize --imatrix models/ctrl_imatrix.gguf --tensor-type-file results/abz_tensor_types.txt \
   $BF models/ctrl_imat_Q5_K_M.gguf Q5_K_M $T > logs/quantize_imat.log 2>&1
python scripts/mk_types.py check $A models/ctrl_noimat_Q5_K_M.gguf
python scripts/mk_types.py check $A models/ctrl_imat_Q5_K_M.gguf
sha256sum $BF $A models/liquid/LFM2.5-24B-A2B-Q5_K_M.gguf models/ctrl_*.gguf | tee results/control_sha256.txt
# Byte equality of every tensor (raw quantised bytes) between the upload and each control.
python scripts/byteeq.py $A models/ctrl_noimat_Q5_K_M.gguf | tee results/byteeq_abz_vs_ctrl_noimat.txt
python scripts/byteeq.py $A models/ctrl_imat_Q5_K_M.gguf | tee results/byteeq_abz_vs_ctrl_imat.txt
