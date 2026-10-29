#!/usr/bin/env bash
# GGUF metadata dump (general.name, tags, quantize.imatrix.*), tokenizer/chat-template diff, format + suspicious-content scan.
set -euo pipefail; . venv/bin/activate; mkdir -p results
A=models/abz/LFM2.5-24B-A2B-GSQ-RCO-Q5_K_M.gguf; L=models/liquid/LFM2.5-24B-A2B-Q5_K_M.gguf
python scripts/meta_dump.py $A > results/meta_abz_q5.json
python scripts/meta_dump.py $L > results/meta_liquid_q5.json
python scripts/structdiff.py $L $A | tee results/struct_Q5.txt
python scripts/tokdiff.py $L $A | tee results/tokdiff_Q5.txt
python scripts/fmt_check.py results/fmt_Q5.json $A $L
python - <<'PY'
import json
m=json.load(open('results/meta_abz_q5.json'))
for k in ['general.name','general.tags','general.file_type','quantize.imatrix.file','quantize.imatrix.dataset',
          'quantize.imatrix.chunks_count','quantize.imatrix.entries_count']: print(f'{k:32s}', m.get(k))
PY
