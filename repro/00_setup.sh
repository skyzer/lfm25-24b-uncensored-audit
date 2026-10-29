#!/usr/bin/env bash
# Build llama.cpp at the commit I used, create a venv, install python deps.
set -euo pipefail
[ -d llama.cpp ] || git clone https://github.com/ggml-org/llama.cpp
git -C llama.cpp checkout 19cce6919992fdffd92e0ac152caaf1e2b8f254e
CUDA=${CUDA:-ON}   # CUDA=OFF for CPU-only
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=$CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build -j --target llama-quantize llama-imatrix llama-server
python3 -m venv venv && . venv/bin/activate
pip install -q -r requirements.txt && pip install -q -e llama.cpp/gguf-py
