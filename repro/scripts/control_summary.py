# Summarise chan_cmp.py output. Usage: control_summary.py results/chan_cmp_*.jsonl > results/control_summary.json
import json, sys, re, collections, statistics as st
rows = [json.loads(l) for p in sys.argv[1:] for l in open(p)]
rows.sort(key=lambda r: (int(re.match(r'blk\.(\d+)', r['name']).group(1)) if r['name'].startswith('blk.') else -1, r['name']))
def med(v): v = [x for x in v if x is not None]; return round(st.median(v), 4) if v else None
def mx(v): v = [x for x in v if x is not None]; return round(max(v), 4) if v else None
out = dict(n_tensors=len(rows), note='chan_cmp subset (CHAN_FILTER); the full byte-equality check over all tensors is byteeq_abz_vs_ctrl_*.txt')
# 1) deterministic types: Q8_0 / F16 / F32 do not use the imatrix -> untouched tensors must be byte-identical to the control
det = [r for r in rows if r['type_A'] in ('Q8_0', 'F16', 'F32')]
out['deterministic_types'] = dict(n=len(det), by_type=dict(collections.Counter(r['type_A'] for r in det)),
    bytes_equal_to_control_noimat=sum(r['bytes_eq_A_CN'] for r in det), bytes_equal_to_control_imat=sum(r['bytes_eq_A_CI'] for r in det),
    differ=[dict(name=r['name'], type=r['type_A'], rel_A_CN=round(r['rel_A_CN'], 5), rel_CN_O=round(r['rel_CN_O'], 5), ratio_A_vs_CN=round(r['ratio_A_vs_CN'], 3))
            for r in det if not r['bytes_eq_A_CN']])
# 2) k-quant tensors (use the imatrix): ratio ||A-O|| / ||C-O||, writers vs readers
kq = [r for r in rows if r['type_A'] not in ('Q8_0', 'F16', 'F32')]
def grp(rs): return dict(n=len(rs), ratio_A_vs_CI_median=med([r['ratio_A_vs_CI'] for r in rs]), ratio_A_vs_CI_max=mx([r['ratio_A_vs_CI'] for r in rs]),
                         ratio_A_vs_CN_median=med([r['ratio_A_vs_CN'] for r in rs]), ratio_A_vs_CN_max=mx([r['ratio_A_vs_CN'] for r in rs]),
                         ratio_CI_vs_CN_median=med([r['ratio_CI_vs_CN'] for r in rs]), ratio_L_vs_CN_median=med([r['ratio_L_vs_CN'] for r in rs]),
                         bytes_equal_CI=sum(r['bytes_eq_A_CI'] for r in rs), bytes_equal_CN=sum(r['bytes_eq_A_CN'] for r in rs))
out['kquant'] = dict(writers=grp([r for r in kq if r['writer']]), readers=grp([r for r in kq if not r['writer']]))
# 3) the e_557 channel pattern: share of delta energy in output row 557, per pair
pairs = ['A-L', 'CI-L', 'CN-L', 'L-O', 'A-O', 'CI-O', 'CN-O', 'A-CI', 'A-CN']
W = [r for r in rows if r['writer'] and r.get('ch557_A-L') is not None]
out['ch557_share_writers'] = {p: dict(median=med([r.get(f'ch557_{p}') for r in W]), max=mx([r.get(f'ch557_{p}') for r in W])) for p in pairs}
out['ch557_share_writers_uniform_baseline'] = round(1 / 2048, 5)
out['top_dir_is_557_writers'] = {p: sum(r.get(f'u_argmax_{p}') == 557 for r in W) for p in ['A-L', 'CI-L', 'CN-L', 'A-O', 'CI-O', 'A-CI', 'A-CN']}
out['n_writers'] = len(W)
out['row557_norm_over_median_in_original'] = dict(median=med([r.get('row557_norm_over_median') for r in W]), max=mx([r.get('row557_norm_over_median') for r in W]),
                                                  n_row557_is_largest=sum(r.get('row_norm_argmax') == 557 for r in W))
# 4) per-tensor table for attn_output (and any tensor whose A-vs-control delta stands out)
def short(r): return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in (
    'name', 'type_A', 'bytes_eq_A_CN', 'bytes_eq_A_CI', 'rel_A_O', 'rel_CN_O', 'rel_CI_O', 'rel_L_O', 'rel_A_CN', 'ratio_A_vs_CN', 'ratio_A_vs_CI',
    'top1_energy_A-CN', 's1_over_s2_A-CN', 'u_argmax_A-CN', 'top1_energy_A-O', 'top1_energy_CI-O', 'top1_energy_A-CI', 'top1_energy_A-L', 'top1_energy_CI-L', 'ch557_A-L', 'ch557_CI-L', 'ch557_CN-L', 'ch557_A-CN')}
# 5) per writer family: does the upload's delta vs the ORIGINAL look more low-rank than the honest control's?
#    (an abliteration edit adds a rank-1 component -> top1_energy_A-O >> top1_energy_CI-O)
fam = collections.defaultdict(list)
for r in W: fam[re.sub(r'^blk\.\d+\.', '', r['name'])].append(r)
out['writer_families'] = {k: dict(n=len(v), ratio_A_vs_CI_median=med([r['ratio_A_vs_CI'] for r in v]), ratio_A_vs_CI_max=mx([r['ratio_A_vs_CI'] for r in v]),
    ratio_L_vs_CN_median=med([r['ratio_L_vs_CN'] for r in v]),
    top1_A_O_median=med([r.get('top1_energy_A-O') for r in v]), top1_A_O_max=mx([r.get('top1_energy_A-O') for r in v]),
    top1_CI_O_median=med([r.get('top1_energy_CI-O') for r in v]), top1_CI_O_max=mx([r.get('top1_energy_CI-O') for r in v]),
    top1_A_CI_median=med([r.get('top1_energy_A-CI') for r in v]), top1_A_CI_max=mx([r.get('top1_energy_A-CI') for r in v]),
    ch557_A_L_median=med([r.get('ch557_A-L') for r in v]), ch557_CI_L_median=med([r.get('ch557_CI-L') for r in v]),
    ch557_CN_L_median=med([r.get('ch557_CN-L') for r in v]), ch557_L_O_median=med([r.get('ch557_L-O') for r in v]),
    ch557_A_CI_median=med([r.get('ch557_A-CI') for r in v])) for k, v in sorted(fam.items())}
out['attn_output'] = [short(r) for r in rows if r['name'].endswith('attn_output.weight')]
out['top10_ratio_A_vs_CN'] = [short(r) for r in sorted(rows, key=lambda r: -r['ratio_A_vs_CN'])[:10]]
out['per_tensor'] = [dict(name=r['name'], type=r['type_A'], writer=r['writer'], bytes_eq_CN=r['bytes_eq_A_CN'], bytes_eq_CI=r['bytes_eq_A_CI'],
                          ratio_A_vs_CN=round(r['ratio_A_vs_CN'], 4), ratio_A_vs_CI=round(r['ratio_A_vs_CI'], 4), ratio_L_vs_CN=round(r['ratio_L_vs_CN'], 4),
                          ch557_A_L=r.get('ch557_A-L'), ch557_CI_L=r.get('ch557_CI-L'), ch557_CN_L=r.get('ch557_CN-L'), ch557_A_CN=r.get('ch557_A-CN')) for r in rows]
print(json.dumps(out, indent=1))
