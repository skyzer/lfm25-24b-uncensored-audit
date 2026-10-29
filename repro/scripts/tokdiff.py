# Tokenizer + chat-template diff between two GGUFs. Usage: tokdiff.py A.gguf B.gguf
import sys, gguf, difflib
def kv(p):
    r=gguf.GGUFReader(p); return {k:f.contents() for k,f in r.fields.items() if k.startswith('tokenizer.')}
A,B=kv(sys.argv[1]),kv(sys.argv[2])
for k in sorted(set(A)|set(B)):
    a,b=A.get(k),B.get(k)
    if k=='tokenizer.chat_template':
        if a==b: print('chat_template: IDENTICAL', len(a or ''))
        else:
            print('chat_template: DIFFERENT', len(a or ''), len(b or ''))
            for l in difflib.unified_diff((a or '').splitlines(),(b or '').splitlines(),'A','B',lineterm='',n=1): print('   ',l)
        continue
    if a==b: print(k, 'IDENTICAL', f'(len {len(a)})' if isinstance(a,list) else repr(a)[:60])
    else:
        if isinstance(a,list) and isinstance(b,list):
            d=[i for i in range(min(len(a),len(b))) if a[i]!=b[i]]
            print(k,'DIFF len',len(a),len(b),'n_diff',len(d),'first',[(i,a[i],b[i]) for i in d[:8]])
        else: print(k,'DIFF',repr(a)[:100],'|',repr(b)[:100])
