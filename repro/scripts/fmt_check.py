# GGUF file-format sanity + suspicious-content scan. Usage: fmt_check.py out.json file1.gguf [file2.gguf ...]
import sys, os, re, json, hashlib, struct, gguf, numpy as np
SUS = {
 'url': r'https?://|www\.', 'ip': r'\b\d{1,3}(?:\.\d{1,3}){3}\b', 'py_exec': r'\b(?:__import__|subprocess|os\.system|eval\s*\(|exec\s*\(|popen|getattr\s*\(|__class__|__globals__|__builtins__|__subclasses__)',
 'shell': r'(?:curl|wget|nc |bash -c|/dev/tcp|powershell|chmod \+x)', 'b64blob': r'[A-Za-z0-9+/]{200,}={0,2}',
 'zero_width': '[\u200b\u200c\u200d\u2060\ufeff]', 'jinja_io': r'\b(?:import|include|extends|from)\s+["\']', 'jinja_dunder': r'\.__\w+__',
}
def scan(s):
    hits={}
    for k,r in SUS.items():
        m=re.findall(r,s,re.I)
        if m: hits[k]=[str(x)[:80] for x in m[:5]]
    return hits
out={}
for p in sys.argv[2:]:
    r = gguf.GGUFReader(p); size=os.path.getsize(p)
    with open(p,'rb') as f: magic=f.read(4); ver=struct.unpack('<I',f.read(4))[0]; nt,nkv=struct.unpack('<QQ',f.read(16))
    align = r.fields['general.alignment'].contents() if 'general.alignment' in r.fields else 32
    ends=[]; bad_align=[]; ivs=[]
    for t in r.tensors:
        st=t.data_offset; en=st+t.n_bytes; ivs.append((st,en,t.name))
        if (st - r.data_offset) % align: bad_align.append(t.name)
    ivs.sort(); overlaps=[(a[2],b[2]) for a,b in zip(ivs,ivs[1:]) if b[0] < a[1]]
    last_end=max(e for _,e,_ in ivs); tail=size-last_end
    gaps=sum(max(0,b[0]-a[1]) for a,b in zip(ivs,ivs[1:])); maxgap=max([b[0]-a[1] for a,b in zip(ivs,ivs[1:])]+[0])
    kv={}; sus={}; nonstd=[]
    STD=('general.','tokenizer.','quantize.','split.')
    arch = r.fields['general.architecture'].contents()
    for k,fld in r.fields.items():
        if k.startswith('GGUF.'): continue
        if not (k.startswith(STD) or k.startswith(arch+'.')): nonstd.append(k)
        if fld.types and fld.types[0]==gguf.GGUFValueType.ARRAY:
            v=fld.contents(); kv[k]={'array_len':len(v),'sha16':hashlib.sha256(json.dumps(v,default=str,ensure_ascii=False).encode()).hexdigest()[:16]}
            if k not in ('tokenizer.ggml.tokens','tokenizer.ggml.merges') and v and isinstance(v[0],str):
                h=scan('\n'.join(v))
                if h: sus[k]=h
        else:
            v=fld.contents(); kv[k]=v if not isinstance(v,str) or len(v)<300 else {'str_len':len(v),'sha16':hashlib.sha256(v.encode()).hexdigest()[:16]}
            if isinstance(v,str):
                h=scan(v)
                if h: sus[k]=h
    tmpl = r.fields['tokenizer.chat_template'].contents() if 'tokenizer.chat_template' in r.fields else None
    if tmpl: open(os.path.splitext(p)[0]+'.chat_template.jinja','w').write(tmpl)
    # tokens: count tokens containing urls / zero-width / control chars (vocab-level injection check)
    toks = r.fields['tokenizer.ggml.tokens'].contents() if 'tokenizer.ggml.tokens' in r.fields else []
    nonfinite=[]
    for t in r.tensors:
        if t.tensor_type.name in ('F32','F16','BF16') and t.n_elements < 5e7:
            a=np.asarray(t.data)
            if t.tensor_type.name=='BF16': a=(a.astype(np.uint32)<<16).view(np.float32)
            if not np.isfinite(a.astype(np.float32)).all(): nonfinite.append(t.name)
    out[os.path.basename(p)] = dict(size=size, magic=magic.decode('latin1'), version=ver, n_tensors=nt, n_kv=nkv, alignment=align,
        data_offset=r.data_offset, tensors_misaligned=bad_align, tensor_overlaps=overlaps, bytes_after_last_tensor=tail, inter_tensor_gap_total=gaps, max_gap=maxgap,
        nonstandard_kv=nonstd, suspicious_strings=sus, chat_template_len=len(tmpl) if tmpl else 0, chat_template_scan=scan(tmpl) if tmpl else None,
        vocab=len(toks), types=dict(__import__('collections').Counter(t.tensor_type.name for t in r.tensors)), nonfinite_small_float_tensors=nonfinite, kv=kv)
    print(p, 'ok', 'tail', tail, 'overlaps', len(overlaps), 'misaligned', len(bad_align), 'nonstd', nonstd, 'sus', list(sus), flush=True)
json.dump(out, open(sys.argv[1],'w'), indent=1, default=str, ensure_ascii=False)
