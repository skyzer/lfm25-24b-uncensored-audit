import re, sys
URL = 'https://github.com/skyzer/lfm25-24b-uncensored-audit'
P = [
"""An "uncensored" LFM2.5-24B-A2B GGUF on Hugging Face is described as a quantization of Liquid AI's model. I checked what else changed, if anything.

I compared it with:
• Liquid's original BF16
• Liquid's own Q5_K_M
• my clean copy: the original, compressed the same way 🧵""",
"""1. Card vs file
• The card says "quantized" and never mentions abliteration
• The file's own metadata: name "Lfm Heretic Bf16", tags heretic / uncensored / abliterated
• Tokenizer and chat template: identical to Liquid's
(Heretic = a public refusal-removal tool)""",
"""2. Weight diff vs Liquid's Q5
• attn_output in layers 18, 22, 26 stands out: ~50% of each change sits in one direction, 5.5–6.6% shift vs 3.7% in the other 7
• 77 of 80 output matrices also share one pattern (channel 557), so I needed a control""",
"""3. Clean control
• Q8_0, F16 and F32 compression is deterministic: 311 of 314 such tensors are byte-identical to my copy
• The 3 that differ: attn_output in layers 18, 22, 26
• Channel 557 and the expert layers: same as my copy, so just rounding""",
"""4. Behaviour (no network, strace on)
• 50 trigger prefixes × 10 prompts: largest shift 1.8 SD, nothing above 3
• 70 agent tasks with tools: 0 flagged calls, 0 unexpected domains
• Outbound connections: 0
• Injected URLs, packages or payloads: none found""",
"""5. Refusals, same 28 prompts, bigger budget
• Liquid: refused 20/20 harmful and 4/8 borderline
• Upload: refused 0/20 harmful and 0/8 borderline
• Same with reasoning on or off
So it does what the file's metadata says: refusals are removed""",
"""6. Verdict: no evidence of a backdoor. It is Liquid's model with a Heretic-style refusal edit in 3 attention layers, plus normal compression.
Limits: rounding noise can hide tiny edits; triggers can't be exhaustive.
Scripts 00–08 + hashes: """ + URL,
]
def _w(c):
    c = ord(c); return 1 if (c <= 4351 or 8192 <= c <= 8205 or 8208 <= c <= 8223 or 8242 <= c <= 8247) else 2
def n(s):  # X weighted length: URL = 23, most non-Latin symbols (bullets, emoji) = 2
    import unicodedata; s = unicodedata.normalize('NFC', re.sub(r'https?://\S+', 'x' * 23, s)); return sum(_w(c) for c in s)
if __name__ == '__main__':
    ch = sys.argv[1] if len(sys.argv) > 1 else '{CH557}'
    for i, p in enumerate(P): print(i + 1, n(p.replace('{CH557}', ch)))
