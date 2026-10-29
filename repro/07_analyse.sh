#!/usr/bin/env bash
# Behaviour analysis (Liquid's official Q5 = reference) + strace summary.
set -euo pipefail; . venv/bin/activate
python scripts/an_beh2.py results/beh liquid25_q5 abz_q5 liquid25_q5 results/beh_an_lfm25.json
python scripts/strace_an.py results/strace_summary.json logs/strace_abz_q5.log logs/strace_liquid25_q5.log
cat logs/netns_*.txt
