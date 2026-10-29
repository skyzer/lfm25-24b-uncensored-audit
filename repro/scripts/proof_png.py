# Proof image: what the model card at the pinned revision says vs what the GGUF's own metadata says.
# Inputs: expected_outputs/card_abenzerps_a35dd08_README.md, expected_outputs/meta_abz_q5.json (from meta_dump.py)
import json, re, sys
from charts_common import *
E = sys.argv[1] if len(sys.argv) > 1 else '../../expected_outputs/'; OUT = sys.argv[2:] or ['../../plots/proof_metadata_vs_card.png']
card = open(E + 'card_abenzerps_a35dd08_README.md', encoding='utf-8').read(); meta = json.load(open(E + 'meta_abz_q5.json'))
fm = card.split('---')[1]
rel = re.search(r'base_model_relation:\s*(\S+)', fm).group(1); base = re.search(r'base_model:\s*(\S+)', fm).group(1)
tags = re.findall(r'^- (\S+)$', fm.split('tags:')[1].split('base_model')[0], re.M)
body = card.split('---', 2)[2]
hits = {w: len(re.findall(w, card, re.I)) for w in ['heretic', 'abliterat', 'refusal', 'decensor']}
desc = re.search(r'Mixed-precision \*\*GSQ-RCO\*\* GGUF quantizations of', body) is not None
f = fig(); title(f, 'The model card vs the file\'s own metadata', 'abenzerps LFM2.5-24B-A2B "Uncensored" GSQ-RCO Q5_K_M, same pinned revision')
a = overlay(f)
box(f, 0.04, 0.17, 0.44, 0.63, None, ORIG, a=a)
box(f, 0.52, 0.17, 0.44, 0.63, None, ABZ, a=a)
a.text(0.06, 0.755, 'MODEL CARD (README.md)', fontsize=17, weight='bold', color=ORIG, va='top')
a.text(0.54, 0.755, 'INSIDE THE GGUF FILE (gguf-py)', fontsize=17, weight='bold', color=ABZ, va='top')
L = [('base_model:', base), ('base_model_relation:', rel),
     ('tags:', 'quantized, mixed-precision, gsq, rco' if all(t in tags for t in ('quantized', 'mixed-precision', 'gsq', 'rco')) else ', '.join(tags)),
     ('text:', '"Mixed-precision GSQ-RCO GGUF quantizations\n of LiquidAI/LFM2.5-24B-A2B"' if desc else '-'),
     ('word count: heretic / abliterat / refusal', f"{hits['heretic']} / {hits['abliterat']} / {hits['refusal']}")]
y = 0.69
for k, v in L:
    a.text(0.06, y, k, fontsize=12, color=MUT, va='top', family='DejaVu Sans Mono')
    a.text(0.06, y - 0.03, v, fontsize=13.5, color=FG, va='top', family='DejaVu Sans Mono', weight='bold' if k.startswith('base_model_relation') else 'normal')
    y -= 0.077 if '\n' not in v else 0.11
tg = meta['general.tags']
R = [('general.name', repr(meta['general.name'])), ('general.tags', ', '.join(tg[:4]) + ',\n' + ', '.join(tg[4:7])),
     ('quantize.imatrix.file', '.../' + meta['quantize.imatrix.file'].split('/')[-1].replace('Uncensored-', 'Uncensored-\n    ')),
     ('quantize.imatrix.dataset', '.../' + meta['quantize.imatrix.dataset'].split('/')[-1]),
     ('imatrix chunks / entries', f"{meta['quantize.imatrix.chunks_count']} / {meta['quantize.imatrix.entries_count']}"),
     ('general.file_type', f"{meta['general.file_type']} (Q5_K_M)")]
y = 0.69
for k, v in R:
    a.text(0.54, y, k, fontsize=12, color=MUT, va='top', family='DejaVu Sans Mono')
    a.text(0.54, y - 0.03, v, fontsize=13.5, color=FG, va='top', family='DejaVu Sans Mono', weight='bold' if k in ('general.name', 'general.tags') else 'normal')
    y -= 0.077 if '\n' not in v else 0.11
a.text(0.04, 0.125, '"Heretic" is a public automatic abliteration tool (it removes the refusal direction from the weights).', fontsize=13, color=WARN)
foot(f, 'Card + GGUF: huggingface.co/abenzerps/LFM2.5-24B-A2B-Uncensored-GSQ-RCO-GGUF @ revision a35dd08746427b39342943fefad690f53136a681\n'
        'README.md sha256 b2aadcd261ce…a9bc · LFM2.5-24B-A2B-GSQ-RCO-Q5_K_M.gguf sha256 3761486d7ca9…5d59\n'
        'Base: huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF @ 0912d33141f77e10705345a6732bfd83f0252fdc', fs=10)
for o in OUT: f.savefig(o, dpi=100, facecolor=BG)
print('card relation', rel, 'tags', tags, 'hits', hits)
