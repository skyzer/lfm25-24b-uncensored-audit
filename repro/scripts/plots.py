# Charts for the LFM2.5-24B-A2B "uncensored" GGUF audit. Reads $RES (default results/), writes ../plots and ../thread/img.
import json, os, re, sys, glob, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from charts_common import *
HERE = os.path.dirname(os.path.abspath(__file__)); KIT = os.path.abspath(os.path.join(HERE, '..', '..'))
RES = os.environ.get('RES', os.path.join(KIT, 'expected_outputs'))
if not os.path.isabs(RES): RES = os.path.abspath(RES)
P = os.path.join(KIT, 'plots'); TI = os.path.join(KIT, 'thread', 'img'); os.makedirs(P, exist_ok=True); os.makedirs(TI, exist_ok=True)
J = lambda f: json.load(open(os.path.join(RES, f)))
ABZ_ID = 'abenzerps/LFM2.5-24B-A2B-Uncensored-GSQ-RCO-GGUF @a35dd08 (Q5_K_M)'
LIQ_ID = 'LiquidAI/LFM2.5-24B-A2B-GGUF @0912d33 (Q5_K_M, BF16)'
def out(f, name, post=None):
    f.savefig(os.path.join(P, name), dpi=100, facecolor=BG)
    if post: f.savefig(os.path.join(TI, post), dpi=100, facecolor=BG)
    plt.close(f)
KIND = {'attn_output.weight': ('attn_output', ABZ), 'shortconv.out_proj.weight': ('shortconv.out_proj', WARN),
        'ffn_down_exps.weight': ('ffn_down (experts)', ORIG), 'ffn_down.weight': ('ffn_down (dense)', CLEAN)}
def lay(n): m = re.match(r'blk\.(\d+)\.', n); return int(m.group(1)) if m else None

def weight_by_layer():
    d = J('lfm25_abz_vs_liquidq5.json')['tensors']; x = J('lfm25_xlayer.json')
    T = [t for t in d if not t.get('exact_equal') and t.get('top1_energy') is not None and lay(t['name']) is not None]
    f = fig(); title(f, 'Weight diff vs Liquid\'s Q5, by layer',
                     'Share of each tensor\'s delta energy in its single strongest output direction.')
    a = f.add_axes([0.105, 0.25, 0.545, 0.47]); ax_style(a)
    R = [t for t in T if not t['writer']]
    a.scatter([lay(t['name']) for t in R], [t['top1_energy'] for t in R], s=26, c=NOISE, label='read-side (q/k/v, in_proj, up/gate)', zorder=2)
    for k, (lab, c) in KIND.items():
        W = [t for t in T if t['writer'] and t['name'].endswith('.' + k) and not (k == 'ffn_down.weight' and 'exps' in t['name'])]
        a.scatter([lay(t['name']) for t in W], [t['top1_energy'] for t in W], s=80 if k.startswith('attn') else 36, c=c, label=lab, zorder=3,
                  edgecolors=FG if k.startswith('attn') else 'none', linewidths=1)
    for t in T:
        if t['name'].endswith('attn_output.weight') and lay(t['name']) in (18, 22, 26):
            a.annotate(f"L{lay(t['name'])}", (lay(t['name']), t['top1_energy']), xytext=(-8, 10), textcoords='offset points', fontsize=15, color=ABZ, weight='bold', ha='right' if lay(t['name'])==18 else 'center')
    a.set_yscale('log'); a.set_ylim(4e-4, 1.2); a.set_xlim(-1, 40)
    a.set_yticks([1e-3, 1e-2, 1e-1, 1]); a.set_yticklabels(['0.1%', '1%', '10%', '100%'])
    a.set_xlabel('layer (block)'); a.set_ylabel('top-1 share of delta')
    a.legend(loc='lower left', fontsize=12, frameon=False, labelcolor=FG, ncol=3, bbox_to_anchor=(-0.02, 1.0), columnspacing=1.0, handletextpad=0.2)
    att = {lay(t['name']): t for t in T if t['name'].endswith('attn_output.weight')}
    oth = [t['rel'] for l, t in att.items() if l not in (18, 22, 26)]
    o = overlay(f)
    o.text(0.68, 0.72, 'attn_output L18 / L22 / L26', fontsize=15, color=ABZ, weight='bold')
    o.text(0.68, 0.595, f"top-1 share {att[18]['top1_energy']*100:.0f}% / {att[22]['top1_energy']*100:.0f}% / {att[26]['top1_energy']*100:.0f}%\n"
                        f"rel. delta {att[18]['rel']*100:.1f}–{att[26]['rel']*100:.1f}%\n(other attn_output {min(oth)*100:.1f}%)", fontsize=14, color=FG, linespacing=1.35)
    cd = x['xlayer']['cos_to_common_dir']
    o.text(0.68, 0.50, 'All other writer tensors', fontsize=15, color=MUT, weight='bold')
    o.text(0.68, 0.38, f"{cd['n_over_0p5']}/{x['xlayer']['n_dirs']} deltas share ONE direction\n(output channel 557, median\n|cos| {cd['median']:.2f}); tested on control →", fontsize=14, color=FG, linespacing=1.35)
    o.text(0.68, 0.30, 'Read-side tensors', fontsize=15, color=MUT, weight='bold')
    o.text(0.68, 0.25, f"no structure (s1/s2 median {x['reader']['s1_over_s2']['median']:.2f})", fontsize=14, color=FG)
    foot(f, f'{ABZ_ID}  vs  {LIQ_ID}\n{x["n_exact_equal"]}/{x["n_tensors_total"]} tensors exactly equal (all F32). The delta of two Q5 files includes rounding noise from both.\nData: expected_outputs/lfm25_abz_vs_liquidq5.json, lfm25_xlayer.json', fs=11)
    out(f, 'weightdiff_by_layer.png', 'post_03.png')

def behaviour():
    b = J('beh_an_lfm25.json'); s = J('strace_summary.json')
    A, L = 'abz_q5', 'liquid25_q5'; tr = b['trig'][A + '_summary']; to = b['tools']
    f = fig(); title(f, 'Behaviour: triggers, tools, network', 'Greedy decoding, both models in a no-network sandbox, llama-server under strace.')
    rows = [
        ('50 trigger prefixes × 10 prompts', f"largest shift z = {tr['did_z_max']:.1f}\n(no trigger above 3 SD)", CLEAN),
        ('70 agent prompts with tools', f"abz {to[A]['calls']} calls · Liquid {to[L]['calls']} calls\n0 flagged · 0 unexpected domains", CLEAN),
        ('send_email recipients', f"abz → boss@example.com ×{to[A]['email_to'].get('boss@example.com',0)}\nLiquid → boss@example.com ×{to[L]['email_to'].get('boss@example.com',0)}", CLEAN),
        ('strace: outbound connections', f"abz {s['strace_abz_q5.log']['external_attempts']} · Liquid {s['strace_liquid25_q5.log']['external_attempts']}\n(1 AF_INET socket each = local server port)", CLEAN),
        ('100 benign prompts vs Liquid Q5', f"identical text {b['benign'][A]['ident']}/100\nsame first token {b['benign'][A]['first_tok_same']}/100", WARN),
    ]
    o = overlay(f); y = 0.685
    for lab, val, c in rows:
        box(f, 0.04, y, 0.42, 0.10, lab, NOISE, fs=16, ha='left', a=o)
        box(f, 0.50, y, 0.46, 0.10, val, c, fs=15, a=o); y -= 0.124
    foot(f, f'{ABZ_ID}  vs  {LIQ_ID}\nTool calls are recorded, never executed. New URLs/packages in abz answers vs Liquid: common docs or placeholder hosts only\n(python.org, example.com, 192.168.0.1); full list in the JSON.\nData: expected_outputs/beh_an_lfm25.json, strace_summary.json, netns_*.txt', fs=11)
    out(f, 'behaviour_trigger_tool_strace.png', 'post_05.png')

def refusal():
    r = J('refusal2_summary.json')
    order = [('abz_q5_nothink', 'abz · reasoning off'), ('liquid25_q5_nothink', 'Liquid · reasoning off'),
             ('abz_q5_think', 'abz · reasoning on'), ('liquid25_q5_think', 'Liquid · reasoning on')]
    order = [(k, l) for k, l in order if k in r]
    f = fig(); title(f, 'Refusal retry: same 28 prompts, bigger budget',
                     'Reasoning off: 512 new tokens. Reasoning on: 4096 tokens, final answer scored.')
    for j, cat in enumerate(['harmful', 'borderline']):
        a = f.add_axes([0.25 + 0.38 * j, 0.22, 0.33, 0.53]); ax_style(a); a.grid(axis='y', visible=False); a.grid(axis='x', color=GRID)
        for i, (k, l) in enumerate(order):
            c = r[k][cat]; y = len(order) - 1 - i; x0 = 0
            for v, col in ((c['refused'], CLEAN), (c['answered'], ABZ), (c['capped_no_answer'], NOISE)):
                if v: a.barh(y, v, left=x0, color=col, height=0.62); a.text(x0 + v / 2, y, str(v), ha='center', va='center', fontsize=15, color=BG, weight='bold')
                x0 += v
        a.set_xlim(0, r[order[0][0]][cat]['n']); a.set_yticks(range(len(order)))
        a.set_yticklabels([l for _, l in order][::-1] if j == 0 else [], fontsize=15)
        a.set_title(f"{cat} ({r[order[0][0]][cat]['n']})", fontsize=18, color=FG, loc='left')
    o = overlay(f)
    for x, lab, col in ((0.25, 'refused', CLEAN), (0.40, 'answered', ABZ), (0.57, 'no final answer (token cap)', NOISE)):
        o.add_patch(FancyBboxPatch((x, 0.135), 0.018, 0.03, boxstyle='square,pad=0', fc=col, ec=col)); o.text(x + 0.025, 0.15, lab, fontsize=14, va='center', color=FG)
    foot(f, f'{ABZ_ID}  vs  {LIQ_ID}\nGreedy, regex refusal check on the first 300 chars of the final answer. Compliant answers are redacted in the kit; refusals are kept verbatim.\nData: expected_outputs/refusal2_summary.json, refusal2_*.jsonl', fs=11)
    out(f, 'refusal_retry.png', 'post_06.png')

def control():
    import ast
    be = collections.Counter(); bad = []
    for l in open(os.path.join(RES, 'byteeq_abz_vs_ctrl_noimat.txt')):
        if l.startswith('('): k, v = l.rsplit(' ', 1); be[ast.literal_eval(k)] = int(v)
        elif l.startswith('blk.'): bad.append(l.strip())
    c = J('control_summary.json')
    f = fig(); title(f, 'Clean control: what compression alone does', "Liquid's BF16, quantized by me with the upload's exact per-tensor types")
    a = f.add_axes([0.20, 0.30, 0.27, 0.40]); ax_style(a); a.grid(False)
    T = ['F32', 'F16', 'Q8_0']
    for i, t in enumerate(T):
        s_, d_ = be[(t, True)], be[(t, False)]
        a.barh(i, s_, color=CLEAN, height=0.6); a.barh(i, d_, left=s_, color=ABZ, height=0.6)
        a.text(s_ + d_ + 4, i, f'{s_}/{s_ + d_}', va='center', fontsize=16, weight='bold')
    a.set_yticks(range(3)); a.set_yticklabels(T); a.set_xlim(0, 260); a.invert_yaxis(); a.set_xticks([])
    for s in ['bottom', 'left']: a.spines[s].set_visible(False)
    a.set_title('Byte-identical to my copy', fontsize=17, pad=10, loc='left')
    o = overlay(f)
    o.text(0.04, 0.165, 'These types are deterministic:\nany untouched tensor must match.', fontsize=13, color=MUT, linespacing=1.35)
    o.text(0.04, 0.085, 'Not identical (pink): ' + ', '.join('L' + b.split('.')[1] for b in bad) + ' attn_output', fontsize=14, color=ABZ, linespacing=1.35, weight='bold')
    ch = c['ch557_share_writers']; P = [('upload − Liquid Q5', 'A-L', ABZ), ('my copy − Liquid Q5', 'CI-L', CLEAN), ('my copy − original', 'CI-O', ORIG)]
    a2 = f.add_axes([0.62, 0.30, 0.34, 0.40]); ax_style(a2)
    v = [ch[p]['median'] for _, p, _ in P]
    a2.bar(range(3), v, color=[col for *_, col in P], width=0.6)
    for i, x in enumerate(v): a2.text(i, x * 1.15, f'{x*100:.0f}%' if x >= 0.01 else f'{x*100:.2f}%', ha='center', fontsize=16, weight='bold')
    a2.set_yscale('log'); a2.set_ylim(1e-4, 2); a2.set_yticks([1e-3, 1e-2, 1e-1, 1]); a2.set_yticklabels(['0.1%', '1%', '10%', '100%'])
    a2.axhline(1 / 2048, color=MUT, lw=1.5, ls='--')
    a2.set_xticks(range(3)); a2.set_xticklabels([p[0].replace(' − ', ' −\n') for p in P], fontsize=12.5)
    a2.set_title('Channel-557 share of the delta', fontsize=17, pad=10, loc='left')
    o.text(0.56, 0.10, CONTROL_NOTE, fontsize=13, color=FG, linespacing=1.35)
    foot(f, f"Median over {c['n_writers']} output (writer) matrices; dashed = even spread (1/2048). Data: byteeq_abz_vs_ctrl_noimat.txt, control_summary.json", fs=11)
    out(f, 'control_requant.png', 'post_04.png')
CONTROL_NOTE = 'Same share for the upload and my honest copy.\nChannel 557 is the largest-norm row in every\nwriter, so rounding error lands there.'

def hook():
    f = fig(); title(f, 'What is inside this "uncensored" LFM2.5-24B?', 'I put four versions of the same model side by side')
    a = box(f, 0.04, 0.42, 0.25, 0.28, 'ORIGINAL\nLiquid AI\nLFM2.5-24B-A2B\n(BF16)', ORIG)
    box(f, 0.38, 0.625, 0.29, 0.16, '"UNCENSORED" GGUF\nabenzerps, Q5_K_M', ABZ, fs=16, a=a)
    box(f, 0.38, 0.415, 0.29, 0.16, "LIQUID'S OWN Q5_K_M\n(official reference)", NOISE, fs=16, a=a)
    box(f, 0.38, 0.205, 0.29, 0.16, 'MY CLEAN COPY\noriginal, compressed with the\nupload\'s exact per-tensor types', CLEAN, fs=15, a=a)
    box(f, 0.76, 0.32, 0.21, 0.40, 'COMPARE\nevery weight\n+ metadata\n+ behaviour\n+ refusals', WARN, a=a)
    for y in (0.705, 0.495, 0.285): arrow(a, 0.30, 0.56, 0.375, y)
    for y in (0.705, 0.495, 0.285): arrow(a, 0.675, y, 0.755, 0.52)
    a.text(0.04, 0.085, 'Clean copy = what compression alone does.\nAnything the upload has beyond that is a real change.', fontsize=17, linespacing=1.4, color=FG)
    foot(f, 'Files pinned by revision + sha256 (see repro kit)'); out(f, 'overview.png', 'post_01.png')

def verdict():
    f = fig(); title(f, "Don't trust me: rerun it", 'Repro kit: 9 scripts, pinned files + sha256, one GPU (or CPU, slowly)')
    S = ['00 setup\nllama.cpp', '01 download\n+ verify hashes', '02 metadata\n+ tokenizer', '03 build clean\ncopies', '04 diff every\nweight',
         '05 behaviour\nsandbox + strace', '06 refusal\nretry', '07 analyse', '08 charts']
    a = overlay(f)
    for i, s in enumerate(S):
        r, cI = divmod(i, 5); x = 0.04 + cI * 0.19; y = 0.56 - r * 0.24; col = [ORIG, ORIG, WARN, CLEAN, ABZ, WARN, ABZ, WARN, CLEAN][i]
        box(f, x, y, 0.155, 0.15, s, col, fs=15, a=a)
        if i < 8 and cI < 4: arrow(a, x + 0.16, y + 0.075, x + 0.185, y + 0.075)
    a.plot([0.8775, 0.8775, 0.1175], [0.555, 0.505, 0.505], color=MUT, lw=2.5); arrow(a, 0.1175, 0.505, 0.1175, 0.475)
    a.text(0.04, 0.10, "Limits: weight checks can't rule out changes hidden inside compression rounding;\ntrigger tests can't cover every possible input. Only this one GGUF file was checked.", fontsize=15.5, color=MUT, linespacing=1.4)
    foot(f, 'Repro kit: github.com/skyzer/lfm25-24b-uncensored-audit'); out(f, 'repro_flow.png', 'post_07.png')

if __name__ == '__main__':
    todo = sys.argv[1:] or ['weight_by_layer', 'behaviour', 'refusal', 'control', 'hook', 'verdict']
    for n in todo:
        fn = globals().get(n)
        if fn is None: print('skip', n); continue
        try: fn(); print('ok', n)
        except FileNotFoundError as e: print('missing input for', n, e)
