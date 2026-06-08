#!/usr/bin/env bash
# smoke-test-CUDA-large-mtp.sh — NVIDIA CUDA RTX 4060 large-model MTP validation
#
# Model:  unsloth/Qwen3.6-27B-UD-Q4_K_XL.gguf  (MTP heads embedded, 65+1 layers)
# Phases: 1 = baseline q8_0/q8_0 FA, no MTP, -ngl 20              (proves CUDA backend + FA)
#         2 = MTP only, no turbo KV, -ngl 20, draft on CPU         (proves draft-mtp path)
#         3 = MTP + turbo3 V, -ngl 20, draft on CPU                (turbo KV + speculative)
#         4 = MTP + turbo3 V, -ngl 20, 8K ctx, draft on CPU        (stress test)
#
# Usage:  ./smoke-test-CUDA-large-mtp.sh [1|2|3|4|all]   (default: all)
#
# Hardware context (RTX 4060 Laptop — 8 GiB GDDR6 discrete VRAM):
#  - 8 GiB GDDR6, discrete: no shared-RAM overflow like the 780M iGPU.
#  - 27B model weights ~26 GiB at Q4_K_XL; cannot fit fully on GPU.
#  - -ngl 20 offloads ~20 layers to GPU; remainder runs on CPU (system RAM).
#  - Increasing -ngl beyond ~6 risks OOM (8 GiB VRAM for weights alone).
#    Tune carefully: each additional layer is ~400 MiB for this model.
#  - Turbo KV types (turbo3/turbo4) reduce KV cache VRAM by ~2-4.5×,
#    leaving more headroom to increase -ngl without OOM.
#  - FA must be ON whenever any quantized KV type is used (q8_0, turbo3, etc.).
#  - --spec-type draft-mtp requires MTP heads embedded in the GGUF (Qwen3.6 unsloth build).
#  - --spec-draft-ngl 0: MTP draft context runs on CPU, not GPU. The draft head is one
#    lightweight layer; CPU latency is negligible. Without this flag the MTP context
#    tries to allocate ~495 MiB on the GPU after the main model is loaded, which OOMs
#    on an 8 GiB card with -ngl 20 main layers active.
#  - -np 1 required; MTP does not support parallel sequences (--parallel > 1).

set -euo pipefail

BIN=/home/none/git/llama-cpp-turboquant/build-x64-linux-cuda-nvidia-RTX4060-release/bin/llama-server
MODEL=/mnt/512_ssd_internal/vai-llm-tei/llm-models/unsloth/Qwen3.6-27B-UD-Q4_K_XL.gguf
PORT=8085
PHASE=${1:-all}

READY_TIMEOUT=180

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
    da = timings.get('draft_n_accepted')
    dt = timings.get('draft_n')
    if da is not None and dt:
        print(f'draft acceptance: {da}/{dt} ({da/dt*100:.0f}%)')
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
        desc="baseline — q8_0/q8_0 KV + FA, no MTP, -ngl 20"
        flags=(-ngl 20 --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on -c 2048)
        prompt="Reply with exactly one word: ready"
        max_tokens=16
        ;;
    2)
        desc="MTP draft-mtp, q8_0/q8_0 KV, -ngl 20 (draft on CPU)"
        flags=(-ngl 20 --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on
               --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-ngl 0 -c 2048)
        prompt="Count from 1 to 5, one number per line."
        max_tokens=48
        ;;
    3)
        desc="MTP + turbo3 V — production target, -ngl 20 (draft on CPU)"
        flags=(-ngl 20 --cache-type-k q8_0 --cache-type-v turbo3 --flash-attn on
               --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-ngl 0 -c 2048)
        prompt="Count from 1 to 5, one number per line."
        max_tokens=48
        ;;
    4)
        desc="MTP + turbo3 V, 8K context, -ngl 20 — stress test (draft on CPU)"
        flags=(-ngl 20 --cache-type-k q8_0 --cache-type-v turbo3 --flash-attn on
               --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-ngl 0 -c 8192)
        prompt="Count from 1 to 5, one number per line."
        max_tokens=48
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
all)  phases=(1 2 3 4) ;;
[1-4]) phases=("$PHASE") ;;
*)    die "usage: $0 [1|2|3|4|all]" ;;
esac

echo "smoke-test-CUDA-large-mtp: running phases: ${phases[*]}"
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
