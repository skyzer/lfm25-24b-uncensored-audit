#!/usr/bin/env bash
# Pinned revisions of both sides + wikitext-2 (calibration text for the control imatrix).
# NOTE: abenzerps rewrote the repo history after a35dd08 (HEAD cd214d2, "Initial commit"); the Q5_K_M file is no longer
# in HEAD. The full commit sha below still resolved on 2026-10-29, but it may stop working if the old objects are purged.
set -euo pipefail; . venv/bin/activate
hf download abenzerps/LFM2.5-24B-A2B-Uncensored-GSQ-RCO-GGUF LFM2.5-24B-A2B-GSQ-RCO-Q5_K_M.gguf README.md \
   --revision a35dd08746427b39342943fefad690f53136a681 --local-dir models/abz
hf download LiquidAI/LFM2.5-24B-A2B-GGUF LFM2.5-24B-A2B-BF16.gguf LFM2.5-24B-A2B-Q5_K_M.gguf \
   --revision 0912d33141f77e10705345a6732bfd83f0252fdc --local-dir models/liquid
sha256sum -c hashes.sha256
mkdir -p data
hf download Salesforce/wikitext --repo-type dataset --include "wikitext-2-raw-v1/*" --local-dir data/wikitext
python - <<'PY'
import pyarrow.parquet as pq, glob
for s in ['train','test']:
    f=glob.glob(f'data/wikitext/wikitext-2-raw-v1/{s}-*.parquet')[0]
    open(f'data/wiki_{s}.txt','w').write(''.join(pq.read_table(f).column('text').to_pylist()))
PY
