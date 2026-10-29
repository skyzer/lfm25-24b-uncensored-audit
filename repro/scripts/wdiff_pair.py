# Quant-vs-quant diff WITHOUT an unquantised original (used for LFM2.5: Liquid's official Q5_K_M vs the candidate Q5_K_M).
# Internal control: Heretic only edits writer matrices (attn/conv out_proj, ffn_down incl. experts). Read-side matrices
# (q/k/v, in_proj, ffn_up/gate, router) see only re-quantisation noise, so their delta structure is the noise reference.
# Per tensor: rel delta, exact/bytes equality; for matrices: output-space top-1 energy and s1/s2 of the delta (Gram summed over experts),
# and for writer tensors the share of delta energy explained by a rank-1 projection (I - a u u^T) W with shared u.
# Usage: wdiff_pair.py orig.gguf cand.gguf out_prefix
import sys, re, json, gguf, numpy as np, gc
from gguf.quants import dequantize
O={t.name:t for t in gguf.GGUFReader(sys.argv[1]).tensors}; A={t.name:t for t in gguf.GGUFReader(sys.argv[2]).tensors}; out=sys.argv[3]
def deq(t): return dequantize(t.data,t.tensor_type).astype(np.float32).reshape([int(x) for x in t.shape[::-1]])
WR=re.compile(r'(attn_output|shortconv\.out_proj|ffn_down|ffn_down_exps|ffn_down_shexp)\.weight$')
rows=[]; dirs={}
for n in O:
    if n not in A: continue
    o=deq(O[n]); a=deq(A[n]); da=a-o; no=float(np.linalg.norm(o))
    r=dict(name=n,type_orig=O[n].tensor_type.name,type_cand=A[n].tensor_type.name,shape=list(o.shape),writer=bool(WR.search(n)),
           rel=float(np.linalg.norm(da)/max(no,1e-30)),exact_equal=bool(np.array_equal(a,o)))
    if o.ndim==3:
        pe=np.linalg.norm(da.reshape(da.shape[0],-1),axis=1)/np.maximum(np.linalg.norm(o.reshape(o.shape[0],-1),axis=1),1e-30)
        r['expert_rel_min']=float(pe.min()); r['expert_rel_max']=float(pe.max()); r['expert_rel_argmax']=int(pe.argmax())
    if o.ndim>=2 and n!='token_embd.weight' and n!='output.weight':
        D=da.astype(np.float64).reshape(-1,*da.shape[-2:]); Oo=o.astype(np.float64).reshape(-1,*o.shape[-2:])
        G=sum(x@x.T for x in D); ev,U=np.linalg.eigh(G); ev=np.clip(ev[::-1],0,None); U=U[:,::-1]; u=U[:,0]
        r['top1_energy']=float(ev[0]/max(ev.sum(),1e-300)); r['s1_over_s2']=float(np.sqrt(ev[0]/max(ev[1],1e-300))); r['out_dim']=int(G.shape[0])
        if r['writer']:
            num=sum(float((u@x)@(u@w)) for x,w in zip(D,Oo)); den=sum(float((u@w)@(u@w)) for w in Oo); alpha=-num/den; r['alpha']=alpha
            res2=sum(float(np.linalg.norm(x+alpha*np.outer(u,u@w))**2) for x,w in zip(D,Oo)); r['rank1_explained']=float(1-res2/float((D**2).sum()))
            dirs[n]=u.astype(np.float32)
        del D,Oo
    if n=='token_embd.weight':
        q=np.linalg.norm(da,axis=1)/np.maximum(np.linalg.norm(o,axis=1),1e-30); r['emb_row_rel_max']=float(q.max()); r['emb_row_rel_median']=float(np.median(q))
        r['emb_top_rows']=[(int(i),float(q[i])) for i in np.argsort(-q)[:20]]
    rows.append(r); print(json.dumps({k:v for k,v in r.items() if k!='emb_top_rows'}),flush=True)
    del o,a,da; gc.collect()
json.dump(dict(tensors=rows,only_in_cand=sorted(set(A)-set(O)),only_in_orig=sorted(set(O)-set(A))),open(out+'.json','w'),indent=1); np.savez(out+'_dirs.npz',**dirs)
