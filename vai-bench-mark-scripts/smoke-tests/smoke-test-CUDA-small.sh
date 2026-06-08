#!/usr/bin/env bash
# smoke-test-CUDA-small.sh — NVIDIA CUDA RTX 4060 small-model validation
#
# Model:  ggml-org/gemma-4-E2B-it-Q8_0.gguf  (~2B, fits fully on GPU)
# Phases: 1 = baseline q8_0/q8_0 KV + FA, -ngl 99  (proves CUDA backend + full GPU load)
#         2 = f16 K + turbo4 V + FA, -ngl 99        (lightest turbo, step-1 ladder)
#         3 = q8_0 K + turbo3 V + FA, -ngl 99       (production target)
#
# Usage:  ./smoke-test-CUDA-small.sh [1|2|3|all]   (default: all)
#
# Hardware context (RTX 4060 Laptop — 8 GiB GDDR6 discrete VRAM):
#  - Discrete GPU: VRAM is separate from system RAM; 8 GiB hard limit.
#  - 2B model weights ~2 GiB at Q8_0; fits fully on GPU with -ngl 99.
#  - FA must be ON whenever any quantized KV type is used (q8_0, turbo3, etc.).
#  - GGML_CUDA_FA_ALL_QUANTS=ON is baked into the binary (all quant types use FA).

set -euo pipefail

BIN=/home/none/git/llama-cpp-turboquant/build-x64-linux-cuda-nvidia-RTX4060-release/bin/llama-server
MODEL=/mnt/512_ssd_internal/vai-llm-tei/llm-models/ggml-org/gemma-4-E2B-it-Q8_0.gguf
PORT=8084
PHASE=${1:-all}

READY_TIMEOUT=60

PASS_COUNT=0
FAIL_COUNT=0
SERVER_PID=

die() { echo "ERROR: $*" >&2; exit 1; }

phase_header() {
    local num="$1"; shift
    local desc="$1"; shift
    local args=("$@")
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    printf  "║  PHASE %-2s : %-53s║\n" "$num" "$desc"
    echo    "╠══════════════════════════════════════════════════════════════════╣"
    printf  "║  model  : %-54s║\n" "$(basename "$MODEL")"
    printf  "║  binary : %-54s║\n" "$(basename "$BIN")"
    printf  "║  port   : %-54s║\n" "$PORT"
    printf  "║  ngl    : %-54s║\n" "see flags (RTX 4060, 8 GiB discrete VRAM)"
    echo    "║  flags  :                                                        ║"
    for arg in "${args[@]}"; do
        printf "║    %-62s║\n" "$arg"
    done
    echo "╚══════════════════════════════════════════════════════════════════╝"
}

start_server() {
    local extra_args=("$@")
    echo "starting server …"
    "$BIN" \
        -m "$MODEL" \
        --port "$PORT" \
        --no-warmup \
        -np 1 \
        "${extra_args[@]}" &
    SERVER_PID=$!
}

wait_ready() {
    local retries=$READY_TIMEOUT
    echo "waiting for /health on :$PORT (up to ${retries}s) …"
    while (( retries-- > 0 )); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            echo "server ready"
            return 0
        fi
        sleep 1
    done
    die "server did not become ready within ${READY_TIMEOUT}s"
}

check_inference() {
    local prompt="${1:-Reply with exactly one word: ready}"
    local max_tokens="${2:-24}"
    local label="${3:-inference}"
    echo "--- $label ---"
    local payload
    payload=$(python3 -c "
import json, sys
print(json.dumps({
    'model': 'smoke',
    'messages': [{'role': 'user', 'content': sys.argv[1]}],
    'max_tokens': int(sys.argv[2]),
    'temperature': 0,
}))" "$prompt" "$max_tokens")

    local raw
    raw=$(curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$payload") || die "curl failed for $label"

    echo "$raw" | python3 -c "
import sys, json
r = json.load(sys.stdin)
content = r['choices'][0]['message']['content'].strip()
if not content:
    print('WARN: empty response')
else:
    print('response :', content[:160])
timings = r.get('timings', {})
if timings:
    tps = timings.get('predicted_per_second', 0)
    print(f'throughput: {tps:.2f} tok/s')
"
}

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=
    fi
}

record_pass() {
    echo "PASS: $*"
    (( PASS_COUNT++ )) || true
}

record_fail() {
    echo "FAIL: $*"
    (( FAIL_COUNT++ )) || true
}

run_phase() {
    local num="$1"
    local desc flags prompt max_tokens

    case "$num" in
    1)
        desc="baseline — q8_0/q8_0 KV + FA, full GPU offload"
        flags=(-ngl 99 --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on -c 2048)
        prompt="Reply with exactly one word: ready"
        max_tokens=16
        ;;
    2)
        desc="turbo4 V + FA, full GPU offload (lightest turbo)"
        flags=(-ngl 99 --cache-type-k f16 --cache-type-v turbo4 --flash-attn on -c 2048)
        prompt="Count from 1 to 5, one number per line."
        max_tokens=32
        ;;
    3)
        desc="q8_0 K + turbo3 V + FA, full GPU offload (production)"
        flags=(-ngl 99 --cache-type-k q8_0 --cache-type-v turbo3 --flash-attn on -c 4096)
        prompt="Count from 1 to 5, one number per line."
        max_tokens=32
        ;;
    *)
        die "unknown phase '$num'"
        ;;
    esac

    phase_header "$num" "$desc" "${flags[@]}"

    stop_server
    start_server "${flags[@]}"

    local ok=0
    if wait_ready && check_inference "$prompt" "$max_tokens" "phase $num check"; then
        record_pass "phase $num — $desc"
    else
        record_fail "phase $num — $desc"
        ok=1
    fi

    stop_server
    return $ok
}

trap stop_server EXIT

[[ -f "$BIN"   ]] || die "binary not found: $BIN"
[[ -f "$MODEL" ]] || die "model not found: $MODEL"

case "$PHASE" in
all)  phases=(1 2 3) ;;
[1-3]) phases=("$PHASE") ;;
*)    die "usage: $0 [1|2|3|all]" ;;
esac

echo "smoke-test-CUDA-small: running phases: ${phases[*]}"
echo "model : $MODEL"
echo "binary: $BIN"

for p in "${phases[@]}"; do
    run_phase "$p" || true
done

echo ""
echo "══════════════════════════════════════════"
echo "  Results: ${PASS_COUNT} passed, ${FAIL_COUNT} failed"
echo "══════════════════════════════════════════"

(( FAIL_COUNT == 0 ))
