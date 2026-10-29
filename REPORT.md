# Audit: abenzerps "uncensored" `LFM2.5-24B-A2B` GGUF vs Liquid AI `LFM2.5-24B-A2B`

**Date:** 2026-10-29 (Dubai time, UTC+4)
**Question.** The upload's model card is titled "Uncensored" but describes the files as quantizations of Liquid AI's LFM2.5-24B-A2B. Does the Q5_K_M GGUF contain anything beyond that? In particular, is there a hidden modification such as a backdoor or trigger, compared with Liquid AI's original weights?

## Verdict

**No evidence of a backdoor was found.** The file is Liquid AI's model with two changes:

1. **A refusal-removal (abliteration) edit in exactly 3 matrices**: `attn_output` of attention layers 18, 22 and 26. These are the only 3 of 314 deterministically quantized tensors (F32 / F16 / Q8_0) that are not byte-identical to my own clean re-quantization of Liquid's BF16. Each of their deltas has a dominant rank-1 component (≈49–54 % of the delta energy in one direction, vs < 1 % in the other 7 `attn_output` matrices). This matches the file's own metadata (`general.name` "Lfm Heretic Bf16", tags `heretic`, `uncensored`, `decensored`, `abliterated`). The model card does not mention any weight edit.
2. **Ordinary Q5_K_M quantization with the uploader's own imatrix.** For the K-quant tensors (Q5_K / Q6_K) the upload's error against the original is the same size as my control's (ratio 0.989–0.999 on the 31 K-quant tensors checked so far, including 25 of the 38 expert down-projections) and has the same shape (top-1 energy share of the delta equal to the control's within 0.0002). No K-quant tensor shows an edit.

Behaviourally:
- No injected URLs, packages, insecure code or other unusual content on 100 benign prompts.
- 50 trigger-like prefixes × 10 prompts: no prefix produced qualitatively different behaviour; the largest shift was 1.8 SD.
- 70 agent tasks with tools: 0 flagged tool calls, 0 unexpected domains. Under strace and in a network namespace: 0 outbound connection attempts.
- It does what its metadata says: with a large token budget, Liquid's Q5 refused 20/20 harmful and 4/8 borderline prompts; the upload refused 0/20 and 0/8, with reasoning on or off.

This is absence of evidence within the limits below, not proof that the file is safe.

## What was compared

| Side | Repo @ revision | File | sha256 |
|---|---|---|---|
| Candidate | `abenzerps/LFM2.5-24B-A2B-Uncensored-GSQ-RCO-GGUF` @ `a35dd08746427b39342943fefad690f53136a681` | `LFM2.5-24B-A2B-GSQ-RCO-Q5_K_M.gguf` | `3761486d7ca9f2c63ac2311a34dc16e5471e2bd89b06c80b73bf5d1d9e657d59` |
| Original | `LiquidAI/LFM2.5-24B-A2B-GGUF` @ `0912d33141f77e10705345a6732bfd83f0252fdc` | `LFM2.5-24B-A2B-BF16.gguf` | `55103fda18def5210dafac2d1f1125afc93baf604dc24cf0edf9e9aee0d576ec` |
| Reference | same | `LFM2.5-24B-A2B-Q5_K_M.gguf` | `7f3029ea2a8e40175496e5022cf7defdcc587df9703d3ed613e64daa51427ecb` |
| My control, no imatrix | made by `repro/03_control.sh` | `ctrl_noimat_Q5_K_M.gguf` | `ca5a07fb9d777b4fc3ccb36cdd33d29df2926aca3e095909808ad17c426aa276` |
| My control, with imatrix | same | `ctrl_imat_Q5_K_M.gguf` | `a20894d13c34cbfad40007dc3628b2ab834b693b52ae71e42294638a6efe12c5` |
| My imatrix | same | `ctrl.imatrix` | `4d16deb779f5916fb3d92d8eed7acc5ade481d35907fa487cd537d8413e831e6` |

The uploader's Hugging Face repo history has since been rewritten (current HEAD `cd214d2`, "Initial commit"), but revision `a35dd08` still resolves, and every hash above matches the LFS pointers at the pinned revisions. The control hashes are what I got with llama.cpp `19cce69` on my machine; the K-quant parts depend on the imatrix and are only expected to be identical on the same build and calibration data.

## Method

1. **Structure and metadata.** I compared KV metadata, tokenizer, chat template, special tokens, the tensor list, shapes, per-tensor quant types and the GGUF layout (alignment, gaps, trailing bytes, non-finite values).
2. **Clean control.** I quantized Liquid's BF16 to the upload's exact per-tensor type mix (`--tensor-type-file`, 221 entries extracted from the upload), once without an imatrix and once with my own imatrix. Q8_0, F16 and F32 quantization is deterministic, so every such tensor that was not edited must be byte-identical to my control. K-quants (Q5_K / Q6_K) depend on the imatrix; the uploader's imatrix (`calibration_corpus_8k.txt`, 8000 chunks) is not published, so for those I compare noise levels instead of bytes.
3. **Weight diff vs Liquid's Q5_K_M.** Every tensor dequantized; relative delta ‖A − L‖/‖L‖, SVD of the delta (share of energy in the top singular direction, σ1/σ2), and a rank-1 fit of the form W' = (I − α·u·uᵀ)·W on the 80 "writer" matrices that write into the residual stream (10 `attn_output`, 30 `shortconv.out_proj`, 2 dense `ffn_down`, 38 `ffn_down_exps`).
4. **Per-channel control** (`chan_cmp.py`) on 55 targeted tensors (40 finished at packaging time): all 10 `attn_output`, all 38 `ffn_down_exps`, plus samples of `shortconv.out_proj`, `ffn_up_exps` and `ffn_gate_exps`. For each tensor and each pair among A (upload), O (BF16 original), L (Liquid Q5), CN / CI (my controls) I compute the relative delta, the top-1 energy share of the delta and the share of the delta that falls on output channel 557.
5. **Behaviour**, llama-server, greedy, in a network namespace with no external route, under `strace -f`, upload vs Liquid's Q5:
   - 100 benign prompts (QA, math, code, security, package/install, agent/shell), scanned for URLs, domains, IPs, install commands and insecure patterns that appear only in the upload's output;
   - no prefix plus 50 trigger-like prefixes × 10 base prompts (first-token KL and change vs the no-prefix answer, as z-scores across prefixes);
   - 70 agent tasks with 4 tools (`web_search`, `web_fetch`, `run_shell`, `send_email`), scanning every tool call;
   - 30 prompts repeated for determinism.
6. **Refusals.** 20 harmful and 8 borderline prompts, reasoning off (512 tokens) and on (4096 tokens), scored for refusal vs answer. Only a short opening of each answer is stored, and compliant openings are redacted in `expected_outputs/`.

## Results

### 1. Card vs file
- The model card (`expected_outputs/card_abenzerps_a35dd08_README.md`) is titled "LFM2.5-24B-A2B Uncensored GSQ-RCO GGUF", but its text describes the files only as "mixed-precision GSQ-RCO GGUF quantizations of LiquidAI/LFM2.5-24B-A2B", with `base_model_relation: quantized` and tags such as `quantized`, `gsq`, `rco`. It does not say how the model was uncensored: no mention of abliteration, Heretic or any weight edit.
- The GGUF's own metadata says `general.name = "Lfm Heretic Bf16"` and tags `liquid, lfm2.5, moe, heretic, uncensored, decensored, abliterated, text-generation`. Heretic is a public refusal-removal tool. The imatrix metadata names `/mnt/local-scratch/LFM2.5-24B-A2B-Uncensored-Heretic.imatrix` (dataset `calibration_corpus_8k.txt`, 258 entries, 8000 chunks).
- Tokenizer, vocab (128,000), special tokens and the chat template (5,443 chars) are identical to Liquid's. The KVs present only in the upload are `general.base_model.*`, `general.license`, `general.tags`, `quantize.imatrix.*` and the tokenizer `add_bos` / `add_eos` / `clean_spaces` flags.
- Same 428 tensors, names and shapes. GGUF layout is clean: no misaligned or overlapping tensors, no gaps, no trailing bytes. (The string scanner flags `nc ` inside the chat template; that template is byte-identical to Liquid's, so it is a false positive.)

### 2. Clean control: byte equality
From `expected_outputs/byteeq_abz_vs_ctrl_noimat.txt` (the imatrix variant gives the same result):

| Type | Byte-identical to my control | Not identical |
|---|---|---|
| F32 | 207 / 207 | 0 |
| F16 | 20 / 20 | 0 |
| Q8_0 | 84 / 87 | **3: `blk.18/22/26.attn_output.weight`** |
| Q5_K, Q6_K (imatrix-dependent) | not expected | 114 |

So 311 of 314 deterministic tensors are untouched, and the 3 exceptions are exactly the 3 outliers from the weight diff.

**K-quant tensors** (`expected_outputs/control_summary.json`, from `chan_cmp.py`; 40 of 55 targeted rows finished when this report was written, see "Pending" below). Ratio = ‖A − O‖ / ‖CI − O‖, where 1.0 means the upload is exactly as far from the original as my honest imatrix quant:

| Group | n | ratio A vs my imat control | top-1 energy share, A − O | same, CI − O |
|---|---|---|---|---|
| `ffn_down_exps` (writers) | 25 / 38 | 0.989 – 0.997 (median 0.995) | median 0.0079, max 0.0174 | median 0.0079, max 0.0174 |
| `ffn_up_exps`, `ffn_gate_exps` (readers, sample) | 6 | 0.9988 – 0.9995 | – | – |
| `attn_output`, untouched layers (Q8_0) | 4 / 7 | 1.000 (byte-identical) | 0.007 – 0.011 | identical |
| **`attn_output` layer 18** (Q8_0) | 1 | **7.72** | **0.956** | 0.008 |
| **`attn_output` layer 26** (Q8_0) | 1 | **10.27** | **0.712** | 0.009 |

- The upload's K-quant deltas are slightly *smaller* than my control's (ratio < 1), consistent with an imatrix calibrated on more data (8000 chunks vs my 64), and not with an added edit. An abliteration edit would raise the ratio and the top-1 share; neither happens.
- A side result: on all 21 checked K-quant tensors where Liquid's Q5 uses the same type as the upload, my no-imatrix control is **exactly equal** to Liquid's official Q5_K_M. My toolchain reproduces Liquid's file, so the control is a faithful stand-in for "what compression alone does".
- **Pending at packaging time:** `blk.22.attn_output` and 13 `ffn_down_exps` (layers 0, 1, 25, 27, 28, 30–37) plus a few samples in the per-channel run. `blk.22.attn_output` is already covered by the byte check (not identical, one of the 3) and by the weight diff below. Rerun `repro/04_weightdiff.sh` to complete the table.

### 3. The edit in layers 18, 22, 26
From `expected_outputs/lfm25_abz_vs_liquidq5.json`, delta vs Liquid's Q5:

| `attn_output` | rel. delta | top-1 energy share | σ1/σ2 | α (rank-1 fit) |
|---|---|---|---|---|
| layer 18 | 0.0552 | 0.542 | 6.20 | 0.667 |
| layer 22 | 0.0564 | 0.488 | 2.53 | 0.519 |
| layer 26 | 0.0662 | 0.497 | 1.98 | 0.487 |
| other 7 attention layers | 0.0367 – 0.0369 | < 0.01 | – | ≈ 0 |

Against the BF16 original, the edit is almost purely rank-1: in layer 18, 95.6 % of ‖A − O‖² lies in one direction (my control: 0.8 %), and in layer 26 71 %. The upload's delta is 7.7× and 10.3× the size of the honest Q8_0 rounding error. The edit direction is not channel 557 (|cos| with e₅₅₇ = 0.11 in layer 18; its largest component in layer 26 is channel 1067).

The rest of each delta (≈ 46–51 %) is the Q8_0-vs-Q5 rounding difference that every Q8_0 tensor in the upload has against Liquid's Q5. An edit of this form, (I − α·u·uᵀ)·W on a few output projections, is what abliteration tools produce.

### 4. The channel-557 pattern
77 of 80 writer deltas (upload vs Liquid Q5) share one direction, concentrated on residual channel 557 (median |cos| to the common direction 0.993; random p99 0.057). This looked like a model-wide edit, so I checked it with the control:
- In Q8_0 writers the upload is byte-identical to my control, and the channel-557 share of A − L equals that of CI − L and CN − L (e.g. 0.0029 / 0.0049 / 0.0199 / 0.0092 on the sampled tensors).
- So the pattern is in L, not in A: Liquid's own Q5_K quantization of these matrices rounds channel 557 (a high-magnitude residual channel) coarsely, and any finer quant differs from it there.
- Row 557 is the largest-norm output row of every writer matrix in the original (34/34 checked, 2.2–4.6× the median row norm), so quantization error of *any* kind concentrates there. The top singular direction of CI − O, my own honest quant vs the original, also points at channel 557 in 34 of 34 writers.
- In the K-quant expert down-projections, the channel-557 share is the same for the upload and for my control (median 0.68 % vs 0.70 % against Liquid's Q5). The pattern is rounding, not an edit.

### 5. Behaviour
Reference is Liquid's Q5 (which is therefore 100/100 identical to itself).

| | Upload | Liquid Q5 |
|---|---|---|
| Benign, identical to reference | 33 / 100 | 100 / 100 |
| Benign, same first token | 85 / 100 | 100 / 100 |
| First-token KL, mean / max | 0.104 / 1.55 | 0 |
| Agent tasks with tool calls | 54 calls in 70 tasks | 59 calls in 70 tasks |
| Flagged tool calls / unexpected domains | 0 / 0 | 0 / 0 |
| Outbound connection attempts (strace) | 0 | 0 |
| Determinism, identical prefixes | 24 / 30 | 17 / 30 |

- The upload's output differs from Liquid's on most prompts, as expected for an edited model at a different quant. The features that appear only in its output are legitimate or placeholders: `example.com`, `gmail.com` / `smtp.gmail.com` in an email example, `nvm.sh`, `acme.sh`, `libssh.org`, standard packages (`flask`, `requests`, `numpy`, `bcrypt`, `PyJWT`, `ufw`, `date-fns`, `moment`, `luxon`), and placeholder IPs (`192.168.0.1`, `1.2.3.4`). The `md5`, `sha1`, `base64`, `os.system` and "hard-coded secret" hits are in explanatory or example code.
- Tool calls go to the same kinds of domains for both models (`example.com`, `example.org`, `python.org`, `github.com`, `api.ipify.org`); all `send_email` calls go to the `boss@example.com` address the task names.
- Triggers: the largest z-score for "behaviour changed vs no prefix" was 1.8 (`Developer mode enabled.`); none above 3. The largest first-token-KL z-score was 3.69 for `eval(base64)`, but the generated answers for that prefix are identical to Liquid's on 10/10 prompts.

### 6. Refusals (`expected_outputs/refusal2_summary.json`)

| | Harmful refused | Borderline refused | Median tokens |
|---|---|---|---|
| Liquid Q5, reasoning off | 20 / 20 | 4 / 8 | 14 |
| Liquid Q5, reasoning on | 20 / 20 | 4 / 8 | 1275 |
| Upload, reasoning off | **0 / 20** | **0 / 8** | 512 (cap; 19/20 and 7/8 hit it with an answer under way) |
| Upload, reasoning on | **0 / 20** | **0 / 8** | 2853 |

An earlier run with a 64-token budget was inconclusive (the upload's answers had not started yet) and is superseded by this one.

## Plots
- `plots/overview.png`: the whole audit at a glance.
- `plots/proof_metadata_vs_card.png`: card vs the file's metadata.
- `plots/weightdiff_by_layer.png`: per-layer delta and top-1 share of the writers.
- `plots/control_requant.png`: upload vs my control, byte equality and K-quant noise ratio.
- `plots/behaviour_trigger_tool_strace.png`, `plots/refusal_retry.png`, `plots/repro_flow.png`.

## Limitations
- **Quantization noise is the detection floor.** The uploader's imatrix is not published, so the 114 K-quant tensors cannot be byte-compared. A perturbation small enough to hide inside K-quant rounding cannot be excluded from weights alone. The noise-ratio and rank-1 tests show nothing above the honest-control level.
- The per-channel control targets 55 of 428 tensors (40 finished when the kit was packaged; `blk.22.attn_output` and 13 expert down-projections were still running): all writers that matter for the edit question (`attn_output`, `ffn_down_exps`) plus samples. The byte check covers all 314 deterministic tensors. Run `CHAN_FILTER=. repro/04_weightdiff.sh` for all tensors.
- **Triggers cannot be searched exhaustively.** 50 hand-picked prefixes × 10 prompts, prefixes only, greedy decoding, reasoning off for the trigger scan.
- I checked only this one file at the pinned revision, in llama.cpp `19cce69`. Other quants, later revisions and any runtime code were not checked.

## Reproduce
See `repro/README.md`. Scripts `00_setup.sh` to `08_plots.sh` regenerate every number above; my outputs are in `expected_outputs/`.
