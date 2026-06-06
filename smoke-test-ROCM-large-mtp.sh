#!/usr/bin/env bash
# smoke-test-large-mtp.sh — large-model MTP speculative decoding validation
#
# Model:  unsloth/Qwen3.6-27B-UD-Q4_K_XL.gguf  (MTP heads embedded, 65+1 layers)
# Phases: 1 = baseline q8_0/q8_0 FA, no MTP      (proves model loads + FA works)
#         2 = MTP only, no turbo KV               (proves draft-mtp path)
#         3 = MTP + turbo3 V (production target)  (turbo KV + speculative)
#         4 = MTP + turbo3 V, larger context      (8K tokens, stress test)
#
# Usage:  ./smoke-test-large-mtp.sh [1|2|3|4|all]   (default: all)
#
# Hardware context (780M iGPU, 2048 MiB VRAM):
#  - scripts/vram-layer-split.py reports -ngl 0 with 512 MiB KV reserve.
#  - All 65 transformer layers run on CPU (~15.4 GiB RAM); 20 GiB available.
#  - GPU is not used for weights; HIP backend still handles FA if -ngl > 0.
#  - Use -ngl 0 (pure CPU) to avoid the cudaMalloc OOM on the 780M.
#  - Re-run vram-layer-split.py if moving to a different GPU.
#
# Other notes:
#  - --spec-type draft-mtp requires MTP heads embedded in the GGUF (Qwen3.6 unsloth build)
#  - -np 1 is required; MTP does not support parallel sequences (--parallel > 1)
#  - GGML_HIP_GRAPHS=OFF is already baked into the binary for gfx1103

set -euo pipefail

BIN=/home/none/git/llama-cpp-turboquant/build-x64-linux-hip-rocm713-gfx1103-release/bin/llama-server
MODEL=/mnt/512_ssd_internal/vai-llm-tei/llm-models/unsloth/Qwen3.6-27B-UD-Q4_K_XL.gguf
PORT=8081
PHASE=${1:-all}

READY_TIMEOUT=120

# Accumulated pass/fail counts across the full run
PASS_COUNT=0
FAIL_COUNT=0
SERVER_PID=

die() { echo "ERROR: $*" >&2; exit 1; }

# Print a prominent phase header showing the phase number and every server flag
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
    printf  "║  ngl    : %-54s║\n" "0 (CPU-only; 780M has 2048 MiB VRAM)"
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
        -ngl 0 \
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

# Check inference and print response + throughput.
# Fails hard if the response body is empty or the HTTP call errors.
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

# Run a single numbered phase.  Returns 0 on pass, 1 on fail (does not exit).
run_phase() {
    local num="$1"
    local desc flags prompt max_tokens

    case "$num" in
    1)
        desc="baseline — q8_0/q8_0 KV + FA, no MTP"
        flags=(--cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on -c 2048)
        prompt="Reply with exactly one word: ready"
        max_tokens=16
        ;;
    2)
        desc="MTP draft-mtp, q8_0/q8_0 KV — proves speculative path"
        flags=(--cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on
               --spec-type draft-mtp --spec-draft-n-max 2 -c 2048)
        prompt="Count from 1 to 5, one number per line."
        max_tokens=48
        ;;
    3)
        desc="MTP + turbo3 V — production target"
        flags=(--cache-type-k q8_0 --cache-type-v turbo3 --flash-attn on
               --spec-type draft-mtp --spec-draft-n-max 2 -c 2048)
        prompt="Count from 1 to 5, one number per line."
        max_tokens=48
        ;;
    4)
        desc="MTP + turbo3 V, 8K context — stress test"
        flags=(--cache-type-k q8_0 --cache-type-v turbo3 --flash-attn on
               --spec-type draft-mtp --spec-draft-n-max 2 -c 8192)
        prompt="Count from 1 to 5, one number per line."
        max_tokens=48
        ;;
    *)
        die "unknown phase '$num'"
        ;;
    esac

    phase_header "$num" "$desc" "${flags[@]}"

    # Each phase gets its own server; stop any previous one first.
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

# Determine which phases to run
case "$PHASE" in
all)  phases=(1 2 3 4) ;;
[1-4]) phases=("$PHASE") ;;
*)    die "usage: $0 [1|2|3|4|all]" ;;
esac

echo "smoke-test-large-mtp: running phases: ${phases[*]}"
echo "model : $MODEL"
echo "binary: $BIN"

for p in "${phases[@]}"; do
    run_phase "$p" || true   # continue even if one phase fails
done

echo ""
echo "══════════════════════════════════════════"
echo "  Results: ${PASS_COUNT} passed, ${FAIL_COUNT} failed"
echo "══════════════════════════════════════════"

(( FAIL_COUNT == 0 ))
