# Summarise refusal2.py outputs. Usage: refusal_summary.py results/refusal2_*.jsonl
# refused = regex refusal phrase in the first 300 chars of the final answer; capped = hit the token limit with no final answer.
import json, sys, os, re, collections
out = {}
for p in sys.argv[1:]:
    key = re.sub(r'^refusal2_|\.jsonl$', '', os.path.basename(p)); rows = [json.loads(l) for l in open(p)]
    s = {}
    for cat in ('borderline', 'harmful'):
        r = [x for x in rows if x['cat'] == cat]
        s[cat] = dict(n=len(r), refused=sum(bool(x['refused']) for x in r), answered=sum(x['refused'] is False for x in r),
                      capped_no_answer=sum(bool(x['capped']) for x in r), hit_token_limit=sum(x['finish'] == 'length' for x in r))
    s['median_tokens'] = sorted(x['n_tokens'] or 0 for x in rows)[len(rows) // 2] if rows else None
    out[key] = s
print(json.dumps(out, indent=1))
