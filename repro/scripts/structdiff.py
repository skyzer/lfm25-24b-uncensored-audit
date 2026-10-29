import sys, gguf, numpy as np, hashlib, json
def load(p):
    r = gguf.GGUFReader(p); kv = {}
    for k,f in r.fields.items():
        if k.startswith('GGUF.'): continue
        if f.types and f.types[0]==gguf.GGUFValueType.ARRAY:
            v = f.contents()
            kv[k] = ('ARR', len(v), hashlib.sha256(json.dumps(v, default=str).encode()).hexdigest()[:16], v)
        else: kv[k] = ('VAL', f.contents())
    return r, kv
a, akv = load(sys.argv[1]); b, bkv = load(sys.argv[2])
print('### KV only in A:', sorted(set(akv)-set(bkv)))
print('### KV only in B:', sorted(set(bkv)-set(akv)))
for k in sorted(set(akv)&set(bkv)):
    x, y = akv[k], bkv[k]
    if x[0]=='ARR':
        if x[2]!=y[2]:
            print('DIFF ARR', k, x[1], y[1])
            xv, yv = x[3], y[3]
            n = min(len(xv), len(yv)); d=[i for i in range(n) if xv[i]!=yv[i]]
            print('   n_diff', len(d), 'first', [(i, xv[i], yv[i]) for i in d[:10]])
    elif x[1]!=y[1]:
        s1, s2 = str(x[1]), str(y[1])
        print('DIFF', k, repr(s1[:120]), '|', repr(s2[:120]), '(len', len(s1), len(s2), ')')
    else: print('same', k)
at = {t.name:(tuple(t.shape), t.tensor_type.name) for t in a.tensors}
bt = {t.name:(tuple(t.shape), t.tensor_type.name) for t in b.tensors}
print('### tensors only in A', sorted(set(at)-set(bt)))
print('### tensors only in B', sorted(set(bt)-set(at)))
print('### shape mismatches', [(n, at[n][0], bt[n][0]) for n in at if n in bt and at[n][0]!=bt[n][0]])
if len(sys.argv)>3:
    print('### type mismatches'); [print(' ', n, at[n][1], bt[n][1]) for n in sorted(at) if n in bt and at[n][1]!=bt[n][1]]
