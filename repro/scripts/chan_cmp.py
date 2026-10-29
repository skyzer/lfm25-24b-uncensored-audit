# Requant-control comparison for LFM2.5-24B-A2B, one pass over all tensors.
# Files: O = Liquid BF16 (the original), A = abenzerps Q5_K_M, L = Liquid's official Q5_K_M,
#        CI = my control: Liquid BF16 re-quantised with abenzerps' exact per-tensor types + my imatrix,
#        CN = same control without imatrix.
# Per tensor:
#   * byte equality A vs CI / CN (only meaningful when the quant type matches; Q8_0/F16/F32 do not use the imatrix,
#     so an untouched tensor must be byte-identical to an honest requant)
#   * rel delta of every file vs the original, and ratio ||A-O|| / ||C-O|| (1.0 = pure quantisation noise)
#   * share of the delta energy that sits in output channel 557 (the "e_557" pattern) for the pairs used before
#     (A-L), the control pair (C-L), and vs the original (A-O, C-O, L-O)
#   * the exact wdiff_pair.py statistics (output-space top-1 energy, s1/s2, top direction u) for A-L and C-L
#   * top-1 energy / s1/s2 / direction of A-C (what is left after subtracting the honest control)
# Optional env CHAN_FILTER=<regex> restricts the run to matching tensor names.
# Usage: chan_cmp.py O.gguf A.gguf L.gguf CI.gguf CN.gguf out.jsonl [shard nshards]
import sys, os, re, json, gc, numpy as np, gguf
from gguf.quants import dequantize
fO, fA, fL, fCI, fCN, out = sys.argv[1:7]; shard, nsh = (int(sys.argv[7]), int(sys.argv[8])) if len(sys.argv) > 8 else (0, 1)
R = {k: {t.name: t for t in gguf.GGUFReader(f).tensors} for k, f in dict(O=fO, A=fA, L=fL, CI=fCI, CN=fCN).items()}
WR = re.compile(r'(attn_output|shortconv\.out_proj|ffn_down|ffn_down_exps|ffn_down_shexp)\.weight$')
CH = 557
def deq(t): return dequantize(t.data, t.tensor_type).astype(np.float32).reshape([int(x) for x in t.shape[::-1]])
def raw(t): return np.asarray(t.data).view(np.uint8)
def gram_stats(d):
    D = d.astype(np.float64).reshape(-1, *d.shape[-2:]); G = sum(x @ x.T for x in D)
    ev, U = np.linalg.eigh(G); ev = np.clip(ev[::-1], 0, None); u = U[:, ::-1][:, 0]
    return dict(top1_energy=float(ev[0] / max(ev.sum(), 1e-300)), s1_over_s2=float(np.sqrt(ev[0] / max(ev[1], 1e-300))),
                u_argmax=int(np.argmax(np.abs(u))), u_abs_cos_e557=float(abs(u[CH])) if u.shape[0] > CH else None), u
def ch_share(d):
    if d.ndim < 2 or d.shape[-2] <= CH: return None
    e = (d.astype(np.float64) ** 2); tot = e.sum()
    return float(e[..., CH, :].sum() / tot) if tot > 0 else None
FILT = re.compile(os.environ.get('CHAN_FILTER', '.'))  # optional: restrict to a subset of tensors (regex on name)
names = [n for n in R['O'] if FILT.search(n)][shard::nsh]; dirs = {}
with open(out, 'w') as f:
    for n in names:
        tO, tA, tL, tCI, tCN = (R[k][n] for k in ['O', 'A', 'L', 'CI', 'CN'])
        o, a, l, ci, cn = deq(tO), deq(tA), deq(tL), deq(tCI), deq(tCN)
        r = dict(name=n, shape=list(o.shape), writer=bool(WR.search(n)), type_A=tA.tensor_type.name, type_L=tL.tensor_type.name,
                 type_CI=tCI.tensor_type.name, type_CN=tCN.tensor_type.name,
                 bytes_eq_A_CI=bool(tA.tensor_type == tCI.tensor_type and np.array_equal(raw(tA), raw(tCI))),
                 bytes_eq_A_CN=bool(tA.tensor_type == tCN.tensor_type and np.array_equal(raw(tA), raw(tCN))),
                 exact_eq_A_L=bool(np.array_equal(a, l)), exact_eq_A_O=bool(np.array_equal(a, o)))
        no = float(np.linalg.norm(o)) or 1e-30
        for k, x in dict(A=a, L=l, CI=ci, CN=cn).items(): r[f'rel_{k}_O'] = float(np.linalg.norm(x - o) / no)
        r['rel_A_CI'] = float(np.linalg.norm(a - ci) / no); r['rel_A_CN'] = float(np.linalg.norm(a - cn) / no)
        r['ratio_A_vs_CI'] = r['rel_A_O'] / max(r['rel_CI_O'], 1e-30); r['ratio_A_vs_CN'] = r['rel_A_O'] / max(r['rel_CN_O'], 1e-30)
        r['ratio_CI_vs_CN'] = r['rel_CI_O'] / max(r['rel_CN_O'], 1e-30); r['ratio_L_vs_CN'] = r['rel_L_O'] / max(r['rel_CN_O'], 1e-30)
        if o.ndim >= 2 and n not in ('token_embd.weight', 'output.weight'):
            for p, (x1, x2) in {'A-L': (a, l), 'CI-L': (ci, l), 'CN-L': (cn, l), 'A-O': (a, o), 'CI-O': (ci, o), 'CN-O': (cn, o), 'L-O': (l, o), 'A-CI': (a, ci), 'A-CN': (a, cn)}.items():
                d = x1 - x2
                r[f'ch557_{p}'] = ch_share(d)
                if p in ('A-L', 'CI-L', 'CN-L', 'A-CI', 'A-CN', 'A-O', 'CI-O') and r['writer']:
                    if np.any(d):
                        s, u = gram_stats(d); r.update({f'{k}_{p}': v for k, v in s.items()})
                        if p in ('A-CN', 'A-O'): dirs[f'{p}|{n}'] = u.astype(np.float32)
                del d
            if o.shape[-2] > CH:
                rn = np.sqrt((o.astype(np.float64) ** 2).reshape(-1, o.shape[-2], o.shape[-1]).sum(axis=(0, 2)))
                r['row557_norm_over_median'] = float(rn[CH] / np.median(rn)); r['row_norm_argmax'] = int(rn.argmax())
                # channel 557 of the INPUT side (columns) for reader matrices of the residual stream
            if o.shape[-1] > CH and not r['writer']:
                cnrm = np.sqrt((o.astype(np.float64) ** 2).reshape(-1, o.shape[-1]).sum(axis=0)); r['col557_norm_over_median'] = float(cnrm[CH] / np.median(cnrm))
        f.write(json.dumps(r) + '\n'); f.flush(); print(n, flush=True)
        del o, a, l, ci, cn; gc.collect()
np.savez(out.replace('.jsonl', '_dirs.npz'), **dirs)
