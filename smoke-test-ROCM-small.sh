#!/usr/bin/env bash
# smoke-test-ROCM-small.sh — staged HIP/turbo KV cache validation for gfx1103
#
# Model:  ggml-org/gemma-4-E2B-it-Q8_0.gguf  (~2B, fits fully on GPU)
# Phases: 1 = baseline q8_0/q8_0 KV + FA, -ngl 99  (proves HIP backend + full GPU load)
#         2 = f16 K + turbo4 V + FA, -ngl 99        (lightest turbo, step-1 ladder)
#         3 = q8_0 K + turbo3 V + FA, -ngl 99       (production target)
#
# Usage:  ./smoke-test-ROCM-small.sh [1|2|3]   (default: 1)
#
# Hardware context (780M iGPU — UMA, reports ~24 GiB VRAM shared from system RAM):
#  - The 780M is UMA: VRAM is a window into system RAM, not dedicated memory.
#  - 2B model weights ~2 GiB at Q8_0; fits fully on GPU with -ngl 99.
#  - HIP backend (gfx1103) dispatches compute directly; GGML_HIP_GRAPHS=OFF baked in.

set -euo pipefail

BIN=/home/none/git/llama-cpp-turboquant/build-x64-linux-hip-rocm713-gfx1103-release/bin/llama-server
MODEL=/mnt/512_ssd_internal/vai-llm-tei/llm-models/ggml-org/gemma-4-E2B-it-Q8_0.gguf
PORT=8080
PHASE=${1:-1}

die() { echo "ERROR: $*" >&2; exit 1; }

start_server() {
    local extra_args=("$@")
    echo "--- starting server (phase $PHASE) ---"
    "$BIN" \
        -m "$MODEL" \
        --port "$PORT" \
        --no-warmup \
        -ngl 99 \
        "${extra_args[@]}" &
    SERVER_PID=$!
}

wait_ready() {
    local retries=20
    echo "waiting for server on :$PORT ..."
    while (( retries-- > 0 )); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            echo "server ready"
            return 0
        fi
        sleep 1
    done
    die "server did not become ready within 20 s"
}

check_inference() {
    echo "--- inference check ---"
    curl -sf http://127.0.0.1:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"smoke","messages":[{"role":"user","content":"Reply with one word: ready"}],"max_tokens":8,"temperature":0}' \
        | python3 -c "import sys,json; r=json.load(sys.stdin); print('response:', r['choices'][0]['message']['content'])"
}

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=
    fi
}

trap stop_server EXIT

case "$PHASE" in
1)
    echo "=== phase 1: baseline — f16/f16 KV, no turbo (proves HIP backend loads) ==="
    start_server --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on
    wait_ready
    check_inference
    echo "PASS: HIP baseline OK"
    ;;
2)
    echo "=== phase 2: step-1 ladder — f16 K + turbo4 V (lightest turbo; FA required) ==="
    start_server --cache-type-k f16 --cache-type-v turbo4 --flash-attn on
    wait_ready
    check_inference
    echo "PASS: turbo4 V OK"
    ;;
3)
    echo "=== phase 3: recommended default — q8_0 K + turbo3 V ==="
    start_server --cache-type-k q8_0 --cache-type-v turbo3 --flash-attn on
    wait_ready
    check_inference
    echo "PASS: turbo3 V OK"
    ;;
*)
    die "unknown phase '$PHASE' — pass 1, 2, or 3"
    ;;
esac
