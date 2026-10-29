# Writer-vs-reader s1/s2 and cross-layer consistency of writer edit directions (lfm25 abz vs liquid Q5).
import json,sys,numpy as np,statistics as st,collections
d=json.load(open(sys.argv[1])); Z=np.load(sys.argv[2]); out={}
T=[t for t in d["tensors"] if not t.get("exact_equal") and t.get("s1_over_s2") is not None]
W=[t for t in T if t["writer"]]; R=[t for t in T if not t["writer"]]
def summ(xs,k): v=[x[k] for x in xs if x.get(k) is not None]; return dict(n=len(v),median=st.median(v),p90=float(np.quantile(v,.9)),max=max(v),min=min(v)) if v else None
out["n_tensors_total"]=len(d["tensors"]); out["n_exact_equal"]=sum(bool(t.get("exact_equal")) for t in d["tensors"])
out["only_in_cand"]=d["only_in_cand"]; out["only_in_orig"]=d["only_in_orig"]
out["writer"]={k:summ(W,k) for k in ["s1_over_s2","top1_energy","rel"]}; out["reader"]={k:summ(R,k) for k in ["s1_over_s2","top1_energy","rel"]}
out["writer_by_kind"]={}
for kind in sorted({t["name"].split(".",2)[2] for t in W}):
    xs=[t for t in W if t["name"].split(".",2)[2]==kind]; out["writer_by_kind"][kind]=dict(n=len(xs),s1s2=summ(xs,"s1_over_s2"),top1=summ(xs,"top1_energy"))
out["writer_top10_s1s2"]=[(t["name"],round(t["s1_over_s2"],2),round(t["top1_energy"],4)) for t in sorted(W,key=lambda t:-t["s1_over_s2"])[:10]]
out["reader_top5_s1s2"]=[(t["name"],round(t["s1_over_s2"],2)) for t in sorted(R,key=lambda t:-t["s1_over_s2"])[:5]]
ks=[k for k in Z.keys() if Z[k].shape==(2048,)]; U=np.stack([Z[k]/np.linalg.norm(Z[k]) for k in ks]).astype(np.float64)
C=np.abs(U@U.T); iu=np.triu_indices(len(ks),1); pc=C[iu]
rng=np.random.default_rng(0); v=rng.standard_normal((4000,2048)); v/=np.linalg.norm(v,axis=1,keepdims=True); rc=np.abs(v[:2000]@v[2000:].T).ravel()
out["xlayer"]=dict(n_dirs=len(ks),pairwise_abs_cos=dict(median=float(np.median(pc)),p90=float(np.quantile(pc,.9)),max=float(pc.max()),min=float(pc.min()),frac_over_rand_p99=float((pc>np.quantile(rc,.99)).mean())),random_p99=float(np.quantile(rc,.99)),random_median=float(np.median(rc)))
s=np.linalg.svd(U,compute_uv=False); out["xlayer"]["stack_svd_energy_top1"]=float(s[0]**2/(s**2).sum()); out["xlayer"]["stack_svd_energy_top5"]=float((s[:5]**2).sum()/(s**2).sum())
_,_,Vt=np.linalg.svd(U,full_matrices=False); g=Vt[0]; cg=np.abs(U@g)
out["xlayer"]["cos_to_common_dir"]=dict(median=float(np.median(cg)),min=float(cg.min()),max=float(cg.max()),n_over_0p5=int((cg>0.5).sum()))
out["xlayer"]["per_dir"]=[(k,round(float(c),3)) for k,c in zip(ks,cg)]
# for reference: same stat expected for a pure rank-1 shared edit is ~1.0 (LFM3 Vortecks: pairwise median 0.996)
json.dump(out,open(sys.argv[3],"w"),indent=1); x=out["xlayer"]; print(json.dumps({k:out[k] for k in ["writer","reader"]})); print(json.dumps({k:v for k,v in x.items() if k!="per_dir"})); print(out["writer_top10_s1s2"]); print(out["reader_top5_s1s2"]); print("only",out["only_in_cand"],out["only_in_orig"],"eq",out["n_exact_equal"],"/",out["n_tensors_total"])
