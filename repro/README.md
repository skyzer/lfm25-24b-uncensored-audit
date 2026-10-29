# Repro kit: abenzerps LFM2.5-24B-A2B-Uncensored GGUF vs Liquid AI LFM2.5-24B-A2B

These scripts rerun every check in `../REPORT.md` on your own machine. `scripts/plots.py` and `scripts/proof_png.py` draw `../plots/` and `../thread/img/` from `results/` (or from `../expected_outputs/` with `RES=../expected_outputs`).

## Requirements
- Linux, git, cmake, a C++ compiler, Python 3.11+, `bwrap` (bubblewrap) and `strace` for step 05, and `hf` (huggingface_hub CLI, installed by step 00).
- Disk: about 120 GB. Liquid BF16 is 45 GiB; the upload, Liquid's Q5_K_M and the two controls are 16–17 GiB each.
- RAM: 64 GB recommended. Step 04 dequantizes five copies of each tensor; its three parallel shards use about 60 GB together (run fewer shards on smaller machines).
- GPU: optional but strongly recommended for steps 03 (imatrix), 05 and 06 (generation). With `--cpu-moe` (the default in 06) the experts stay in RAM and about 13 GB VRAM is enough. For CPU only, use `CUDA=OFF NGL=0`.
- No Hugging Face token is needed; all repos are public.

Tested with llama.cpp `19cce6919992fdffd92e0ac152caaf1e2b8f254e`, gguf-py from that tree, numpy 2.x and Python 3.13 on an RTX 6000-class GPU shared with other jobs.

## Run
```bash
bash 00_setup.sh      # clone and build llama.cpp @19cce69, create venv, pip install -r requirements.txt + gguf-py
bash 01_download.sh   # pinned HF revisions, then `sha256sum -c hashes.sha256`, then wikitext-2 (control imatrix text)
bash 02_structure.sh  # metadata (general.name, tags, imatrix info), tokenizer/chat-template diff, tensor list diff
bash 03_control.sh    # quantize Liquid BF16 to the upload's exact per-tensor types (imatrix + no imatrix), byte equality
bash 04_weightdiff.sh # upload vs Liquid Q5 per tensor; cross-layer directions; 5-way comparison with the controls (chan_cmp)
bash 05_behaviour.sh  # one llama-server at a time in a no-network sandbox under strace: benign, trigger, tool-use, determinism
bash 06_refusal.sh    # 28 refusal prompts, both models, reasoning off (512 tok) and on (4096 tok)
bash 07_analyse.sh    # behaviour and strace analysis
bash 08_plots.sh      # ../plots/*.png and ../thread/img/*.png
```

Environment knobs:

| Variable | Default | Used by |
|---|---|---|
| `CUDA` | `ON` | step 00 |
| `NGL` | `99` | steps 03, 05, 06 (GPU layers) |
| `THREADS` | `16` | step 03 |
| `CHUNKS` | `64` | step 03 (imatrix chunks of 512 tokens) |
| `OMP_NUM_THREADS` | `4` | step 04 |
| `CHAN_FILTER` | writers subset | step 04 (`.` = all ~430 tensors, several hours) |
| `EXTRA` | `--cpu-moe` | step 06 |

Pinned inputs (checked in step 01):

| Repo | Revision | File | sha256 |
|---|---|---|---|
| abenzerps/LFM2.5-24B-A2B-Uncensored-GSQ-RCO-GGUF | a35dd08746427b39342943fefad690f53136a681 | LFM2.5-24B-A2B-GSQ-RCO-Q5_K_M.gguf | 3761486d7ca9f2c63ac2311a34dc16e5471e2bd89b06c80b73bf5d1d9e657d59 |
| | | README.md (model card) | b2aadcd261ce80cfa35b5ae68538389d4a9277a6e46f26e87327d3e9d32da9bc |
| LiquidAI/LFM2.5-24B-A2B-GGUF | 0912d33141f77e10705345a6732bfd83f0252fdc | LFM2.5-24B-A2B-BF16.gguf | 55103fda18def5210dafac2d1f1125afc93baf604dc24cf0edf9e9aee0d576ec |
| | | LFM2.5-24B-A2B-Q5_K_M.gguf | 7f3029ea2a8e40175496e5022cf7defdcc587df9703d3ed613e64daa51427ecb |

The uploader's repo history was rewritten after `a35dd08` (HEAD is now `cd214d2`, "Initial commit", without the Q5_K_M file). The full sha above still resolved on 2026-10-29. If it stops resolving, any copy of the file with the sha256 above is the audited file.

## Outputs and how to read them
Everything goes to `results/`. My own outputs are in `../expected_outputs/`.

- **`meta_*.json`, `struct_Q5.txt`, `tokdiff_Q5.txt`, `fmt_*.json`** (step 02): metadata, tensor and tokenizer diffs. Look for `general.name`, `general.tags` and `quantize.imatrix.*` in the upload.
- **`byteeq_abz_vs_ctrl_{noimat,imat}.txt`** (step 03): raw-byte equality of every tensor against the control. Q8_0 / F16 / F32 do not use the imatrix, so an untouched tensor of those types must be byte-identical. K-quant tensors (Q5_K / Q6_K) always differ, because the uploader's imatrix corpus is not public.
- **`lfm25_abz_vs_liquidq5.json`, `lfm25_xlayer.json`** (step 04): per-tensor delta vs Liquid's Q5 (relative norm, top-1 energy, σ1/σ2, α) and how the top directions line up across layers.
- **`chan_cmp_*.jsonl` → `control_summary.json`** (step 04): O = Liquid BF16, A = upload, L = Liquid Q5, CI / CN = control with / without imatrix.
  - `ratio_A_vs_CI` = ‖A − O‖ ÷ ‖CI − O‖. About 1.0 means ordinary quantization noise.
  - `top1_energy_A-O` vs `top1_energy_CI-O`: an abliteration edit adds a rank-1 term, so the upload's delta would be much more concentrated than the control's.
  - `ch557_*`: share of the delta energy in output channel 557 (uniform would be 1/2048 ≈ 0.0005).
- **`beh_an_lfm25.json`, `strace_summary.json`, `netns_*.txt`** (steps 05, 07): identical-output counts, first-token KL, trigger shift z-scores, tool calls and domains, URLs / packages / insecure patterns that appear only in the upload's answers, and every socket syscall of the server.
- **`refusal2_*.jsonl`, `refusal2_summary.json`** (step 06): per-prompt refusal labels. In `../expected_outputs/` the opening of every compliant answer is redacted; refusals are kept verbatim.

## Expected differences when you rerun
- **My imatrix is not bit-reproducible** across GPUs and builds, so the imatrix control's K-quant tensors and ratios against it can move in the 2nd or 3rd decimal. Q8_0, F16 and F32 tensors and the no-imatrix control should be bit-identical (check against `../expected_outputs/control_sha256.txt`).
- **Generation can differ slightly** with a different GPU, backend or build, even greedy. The conclusions should not change.
- **Step 06 may hit the token limit** on the upload in reasoning-off mode (the model writes long answers); the answer is still present and scored.
