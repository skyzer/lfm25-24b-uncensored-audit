#!/usr/bin/env bash
# Run ONE llama-server + the probe battery inside a network namespace with NO external network (bwrap --unshare-net),
# with strace logging every socket/connect/bind/sendto/sendmsg/listen syscall of the server.
# Usage (from repro/): scripts/sandbox_run.sh NAME MODEL.gguf "EXTRA_SERVER_ARGS" "MODES"
set -u
NAME=$1; MODEL=$2; EXTRA=${3:-}; MODES=${4:-"benign trig tools det"}
exec bwrap --dev-bind / / --unshare-net --die-with-parent -- bash -c '
set -u; NAME='"$NAME"'; MODEL='"$MODEL"'; EXTRA="'"$EXTRA"'"; MODES="'"$MODES"'"
B=llama.cpp/build/bin; PY=$PWD/venv/bin/python; mkdir -p results/beh logs
{ echo "== netns check $(date -u +%FT%TZ)"; ip -br addr; curl -s -m4 -o /dev/null -w "%{http_code}" https://huggingface.co && echo " EXTERNAL_NET_REACHABLE(!)" || echo " external net blocked (expected)"; } > logs/netns_$NAME.txt 2>&1
strace -f -qq --seccomp-bpf -e trace=socket,connect,bind,sendto,sendmsg,listen -o logs/strace_$NAME.log \
  $B/llama-server -m $MODEL --port 18100 --host 127.0.0.1 -c ${CTX:-16384} -np ${NP:-8} --jinja $EXTRA > logs/server_$NAME.log 2>&1 &
SP=$!
for i in $(seq 1 600); do curl -sf 127.0.0.1:18100/health >/dev/null && break; sleep 2; done
cd scripts
for m in $MODES; do $PY behave2.py 18100 ../results/beh/${NAME}_$m.jsonl $m >> ../logs/beh_$NAME.log 2>&1; done
pkill -P $SP; kill $SP; wait $SP 2>/dev/null; echo SANDBOX_DONE >> ../logs/beh_$NAME.log'
