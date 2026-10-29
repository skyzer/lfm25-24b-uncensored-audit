# Write a llama-quantize --tensor-type-file that reproduces abenzerps' exact per-tensor quant types,
# and (mode "check") verify that a produced GGUF has exactly the same per-tensor types as abenzerps.
import sys, re, gguf
mode, ref = sys.argv[1], sys.argv[2]
rt = {t.name: t.tensor_type.name for t in gguf.GGUFReader(ref).tensors}
if mode == 'write':
    with open(sys.argv[3], 'w') as f:
        for n, ty in rt.items():
            if ty in ('F32',): continue
            f.write('^' + re.escape(n).replace('\\_', '_') + '$=' + ty.lower() + '\n')
    print(len(rt), 'tensors,', sum(1 for v in rt.values() if v != 'F32'), 'rules')
else:
    ct = {t.name: t.tensor_type.name for t in gguf.GGUFReader(sys.argv[3]).tensors}
    bad = [(n, rt.get(n), ct.get(n)) for n in set(rt) | set(ct) if rt.get(n) != ct.get(n)]
    print(sys.argv[3], 'type mismatches vs abz:', len(bad)); [print(' ', b) for b in bad[:20]]
    sys.exit(1 if bad else 0)
