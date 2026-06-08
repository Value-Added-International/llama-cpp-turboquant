#!/usr/bin/env bash
# smoke-test-VULKAN-780-small.sh — Vulkan AMD 780M iGPU small-model validation
#
# Model:  meta-llama/Llama-2-7B-chat-hf (7B, q4_k_m quantization)
# Phases: 1 = q8_0/q8_0 KV + FA, full GPU offload (-ngl 99)
#         2 = q8_0/turbo3 KV + FA, full GPU offload (turbo KV compression)
#         3 = q8_0/turbo3 KV + FA, larger context  (4K token stress test)
#
# Usage:  ./smoke-test-VULKAN-780-small.sh [1|2|3|all]   (default: all)
#
# Hardware context (780M iGPU — UMA, reports ~24 GiB VRAM shared from system RAM):
#  - The 780M is UMA: VRAM is a window into system RAM, not dedicated memory.
#  - 7B model weights ~4 GiB at Q4_K_M; fits fully on Vulkan GPU with -ngl 99.
#  - FA must be ON whenever any quantized KV type is used (q8_0, turbo3, etc.).
#  - -ngl 99 offloads all layers; Vulkan dispatches the compute shaders on GPU.
#  - -np 1 recommended to avoid memory pressure with parallel sequences.

set -euo pipefail

BIN=/home/none/git/llama-cpp-turboquant/build-x64-linux-vulkan-AMD-780-release/bin/llama-server
MODEL=/mnt/512_ssd_internal/vai-llm-tei/llm-models/meta-llama/Llama-2-7B-chat-hf/ggml-model-Q4_K_M.gguf
PORT=8082
PHASE=${1:-all}

READY_TIMEOUT=120

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
    printf  "║  ngl    : %-54s║\n" "see flags (780M UMA, ~24 GiB reported VRAM)"
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
        desc="turbo3 KV + FA, full GPU offload"
        flags=(-ngl 99 --cache-type-k q8_0 --cache-type-v turbo3 --flash-attn on -c 2048)
        prompt="Count from 1 to 5, one number per line."
        max_tokens=32
        ;;
    3)
        desc="turbo3 KV + FA, full GPU offload, 4K context"
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

echo "smoke-test-VULKAN-780-small: running phases: ${phases[*]}"
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
