# Summarise strace network syscalls of llama-server runs. Usage: strace_an.py OUT.json strace_*.log
import re, sys, json, collections
out={}
for p in sys.argv[2:]:
    c=collections.Counter(); ext=[]; fam=collections.Counter()
    for l in open(p,errors='replace'):
        m=re.match(r'\d+\s+(\w+)\((.*)',l)
        if not m: continue
        sc,args=m.groups(); c[sc]+=1
        if sc=='socket': fam[args.split(',')[0]]+=1
        if sc in('connect','sendto','sendmsg'):
            ip=re.search(r'sin6?_addr=inet_(?:addr|pton)\((?:AF_INET6?, )?"([^"]+)"',args) or re.search(r'inet_addr\("([^"]+)"\)',args)
            port=re.search(r'sin6?_port=htons\((\d+)\)',args)
            if ip and ip.group(1) not in('127.0.0.1','::1','0.0.0.0','::'): ext.append(dict(sc=sc,ip=ip.group(1),port=port.group(1) if port else None,line=l.strip()[:200]))
            elif 'AF_UNIX' in args and len(ext)<0: pass
    out[p.split('/')[-1]]=dict(syscalls=dict(c),socket_families=dict(fam),external_attempts=len(ext),external_examples=ext[:20])
json.dump(out,open(sys.argv[1],'w'),indent=1); print(json.dumps({k:(v['external_attempts'],v['syscalls']) for k,v in out.items()}))
