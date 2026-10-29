# Byte equality of every tensor between two GGUFs (raw quantised bytes). Usage: byteeq.py A.gguf B.gguf
import sys, collections, numpy as np, gguf
A = {t.name: t for t in gguf.GGUFReader(sys.argv[1]).tensors}; B = {t.name: t for t in gguf.GGUFReader(sys.argv[2]).tensors}
c = collections.Counter(); diff = []
for n, t in A.items():
    u = B[n]; same = t.tensor_type == u.tensor_type and np.array_equal(np.asarray(t.data).view(np.uint8), np.asarray(u.data).view(np.uint8))
    c[(t.tensor_type.name, same)] += 1
    if not same and t.tensor_type.name in ("Q8_0", "F16", "F32", "BF16"): diff.append(n)
for k in sorted(c): print(k, c[k])
print("deterministic-type tensors that differ:", len(diff)); print("\n".join(diff))
