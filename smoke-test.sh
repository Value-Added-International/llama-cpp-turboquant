#!/usr/bin/env bash
# smoke-test.sh — staged HIP/turbo KV cache validation for gfx1103
#
# Known issue: this build was compiled with GGML_HIP_GRAPHS enabled (default).
# HIP graph capture is incompatible with the FA kernel on gfx1103 — it aborts
# on the 2nd inference request (or during warmup). Workaround: --no-warmup delays
# the crash past startup. Permanent fix: rebuild with -DGGML_HIP_GRAPHS=OFF.
#
# Run phases:  1 = baseline (f16/f16, no turbo — proves HIP loads correctly)
#              2 = step-1 ladder (f16 K + turbo4 V — lightest turbo, FA required)
#              3 = recommended default (q8_0 K + turbo3 V — production target)
# Usage: ./smoke-test.sh [1|2|3]   (default: 1)

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
    start_server
    wait_ready
    check_inference
    echo "PASS: HIP baseline OK"
    ;;
2)
    echo "=== phase 2: step-1 ladder — f16 K + turbo4 V (lightest turbo; FA auto-enabled) ==="
    echo "NOTE: if this aborts on the 2nd request, rebuild with -DGGML_HIP_GRAPHS=OFF"
    start_server --cache-type-k f16 --cache-type-v turbo4 --flash-attn on
    wait_ready
    check_inference
    echo "PASS: turbo4 V OK"
    ;;
3)
    echo "=== phase 3: recommended default — q8_0 K + turbo3 V ==="
    echo "NOTE: if this aborts on the 2nd request, rebuild with -DGGML_HIP_GRAPHS=OFF"
    start_server --cache-type-k q8_0 --cache-type-v turbo3 --flash-attn on
    wait_ready
    check_inference
    echo "PASS: turbo3 V OK"
    ;;
*)
    die "unknown phase '$PHASE' — pass 1, 2, or 3"
    ;;
esac
