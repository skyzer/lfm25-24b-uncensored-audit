# Dump every GGUF metadata key except the big tokenizer arrays (repro/scripts/meta_dump.py)
import sys, json, gguf
r = gguf.GGUFReader(sys.argv[1]); out = {}
for k, f in r.fields.items():
    if k.startswith('tokenizer.ggml.') and k not in ('tokenizer.ggml.model', 'tokenizer.ggml.pre'): out[k] = f'<array len {len(f.data)}>'; continue
    if k == 'tokenizer.chat_template': out[k] = f'<{len(bytes(f.parts[-1]))} bytes>'; continue
    try:
        v = f.contents()
    except Exception:
        v = [p.tolist() for p in f.parts[-1:]]
    out[k] = v
print(json.dumps(out, indent=1, default=str))
