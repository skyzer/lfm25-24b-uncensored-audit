# One model per run -> its own jsonl (rows keyed by id), so big models can run one at a time.
# Usage: behave2.py PORT OUT.jsonl MODE[benign|trig|refusal|tools|det] [max_tokens]
# Tool calls are only RECORDED, never executed.
import json, sys, requests
sys.path.insert(0,'.'); from prompts import *
port,out,mode=sys.argv[1],sys.argv[2],sys.argv[3]; N=int(sys.argv[4]) if len(sys.argv)>4 else None
def chat(msgs, n, tools=None):
    body={"messages":msgs,"temperature":0,"top_k":1,"seed":0,"max_tokens":n,"logprobs":True,"top_logprobs":20,"chat_template_kwargs":{"enable_thinking":False}}
    if tools: body["tools"]=tools; body.pop("logprobs"); body.pop("top_logprobs")
    r=requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",json=body,timeout=1200).json()
    if "choices" not in r: return {"error":str(r)[:300]}
    c=r["choices"][0]; m=c["message"]
    lp=[{t["token"]:t["logprob"] for t in x["top_logprobs"]} for x in (c.get("logprobs") or {}).get("content",[])[:8]]
    return {"text":m.get("content") or "","reasoning":m.get("reasoning_content") or "","toplp":lp,"finish":c.get("finish_reason"),"tool_calls":m.get("tool_calls") or []}
if mode=="benign": items=[(f"b{i}",dict(cat=c,prompt=p)) for i,(c,p) in enumerate(BENIGN)]; n=N or 512
elif mode=="trig": items=[(f"t{i}_{j}",dict(cat="trig",trigger=t,base=b,prompt=t+b)) for i,t in enumerate(ALL_TRIGGERS) for j,b in enumerate(TRIG_BASE)]; n=N or 160
elif mode=="refusal": items=[(f"rb{i}",dict(cat="borderline",prompt=p)) for i,p in enumerate(REFUSAL_BORDERLINE)]+[(f"rh{i}",dict(cat="harmful",prompt=p)) for i,p in enumerate(REFUSAL_HARMFUL)]; n=N or 64
elif mode=="tools": items=[(f"k{i}",dict(cat="tool",trigger="",prompt=p)) for i,p in enumerate(TOOL_TASKS)]+[(f"k{i}_{j}",dict(cat="tool_trig",trigger=t,prompt=t+p)) for j,t in enumerate(TOOL_TRIGGERS) for i,p in enumerate(TOOL_TASKS[:10])]; n=N or 256
elif mode=="det": items=[(f"b{i}",dict(cat=c,prompt=p)) for i,(c,p) in enumerate(BENIGN[:30])]; n=N or 256
import os
from concurrent.futures import ThreadPoolExecutor
PAR=int(os.environ.get("PAR","8"))
def work(item):
    id_,row=item; row=dict(row)
    res=chat([{"role":"user","content":row["prompt"]}],n,TOOLS if mode=="tools" else None)
    if mode=="refusal":  # keep only an opening snippet of harmful completions
        res={"start":(res.get("text") or res.get("reasoning",""))[:90].replace("\n"," "),"full_len":len(res.get("text","")),"finish":res.get("finish"),"refusal_text":(res.get("text") or "")[:400] if row["cat"]=="borderline" else None}
    row.update(id=id_, **res); return row
with open(out,'w') as f, ThreadPoolExecutor(PAR) as ex:
    for row in ex.map(work,items): f.write(json.dumps(row,ensure_ascii=False)+"\n"); f.flush()
print("done",mode,len(items))
