# llama-bench and llama-batched-bench — TurboQuant+ Fork Reference

**Updated:** 2026-06-06  
**Binary location:** `build-x64-linux-cuda-nvidia-RTX4060-release/bin/`  
**Build script:** `./va-make-bench-only.sh [cuda|rocm|vulkan]`

---

## Overview

This fork extends both benchmark tools with two new capabilities added in the same change:

| Capability | llama-bench | llama-batched-bench |
|---|---|---|
| TurboQuant KV types (`turbo2/3/4`) | `-ctk`/`-ctv` flags (pre-existing) | `-ctk`/`-ctv` flags (pre-existing) |
| MTP speculative decoding | **new** `--mtp-n-max N` | **new** `--spec-type draft-mtp` |
| Draft acceptance reporting | reported in `mtp` column | `D_ACC %` + `SPEC_UP` columns |

Both tools require a model with **embedded MTP heads** to use the MTP path (e.g. unsloth Qwen3 UD builds). Standard GGUF models like bartowski Q4_K_M exports will warn and fall back to plain generation automatically — no crash.

---

## llama-bench

### What it measures

`llama-bench` measures raw throughput in tokens/second across repeated runs and reports mean ± stddev. It sweeps any parameter that is given multiple comma-separated values, producing one row per combination. Results print as Markdown tables by default; JSON, JSONL, CSV, and SQL are also available.

Two benchmark types are always available:

- **Prompt processing (PP):** `-p N` — fills the KV cache with N random tokens and times how long decoding takes.
- **Token generation (TG):** `-n N` — generates N tokens one at a time from a single-token seed.
- **Combined PP+TG:** `-pg PP,TG` — times both back-to-back as one sample.

### New flag: `--mtp-n-max N`

```
--mtp-n-max <n>    enable MTP speculative decoding; 0 = disabled (default: 0)
                   requires model with embedded MTP heads (e.g. unsloth Qwen3 UD builds)
```

When `N > 0`, the TG benchmark loop is replaced by a **draft→verify→accept cycle**:

1. The speculative engine drafts up to `N` candidate tokens using the embedded MTP head.
2. The main model decodes `[id_last, draft_0, …, draft_N-1]` in a single batched call.
3. Accepted tokens advance the KV cache; rejected tokens are discarded.
4. The wall-clock time for `n_gen` accepted tokens is recorded — identical timing slot to the plain TG path, so numbers are directly comparable.

Multiple values are accepted: `--mtp-n-max 0,1,2,4` produces four rows — baseline and three MTP depths — in one run.

The `mtp` column appears automatically in the Markdown table whenever `--mtp-n-max` is specified with any non-default value.

### Key flags (full set)

```
-m, --model <path>           model GGUF path (multi-value: -m a.gguf,b.gguf)
-p, --n-prompt <n>           tokens to prefill  (default: 512)
-n, --n-gen <n>              tokens to generate (default: 128)
-pg <pp,tg>                  combined PP+TG pair
-d, --n-depth <n>            pre-fill depth for TG context (default: 0)
-b, --batch-size <n>         logical batch size   (default: 2048)
-ub, --ubatch-size <n>       physical micro-batch (default: 512)
-ctk, --cache-type-k <t>     KV-cache K dtype: f16, bf16, q8_0, q4_0, turbo2, turbo3, turbo4
-ctv, --cache-type-v <t>     KV-cache V dtype: same options
-fa, --flash-attn <0|1>      flash attention (required for quantized KV types)
-ngl, --n-gpu-layers <n>     layers to offload to GPU (default: 99)
-t, --threads <n>            CPU threads
-r, --repetitions <n>        benchmark repetitions (default: 5)
--mtp-n-max <n>              MTP draft depth; 0 = off (default: 0)
-o, --output <fmt>           csv | json | jsonl | md | sql  (default: md)
--no-warmup                  skip warmup run
--progress                   show per-run progress to stderr
-v, --verbose                verbose llama.cpp log output
```

### Output columns (Markdown)

| Column | Description |
|---|---|
| `model` | model type string from GGUF metadata |
| `size` | weight file size (MiB / GiB) |
| `params` | parameter count (M / B) |
| `backend` | active compute backend(s) |
| `ngl` | GPU layers offloaded |
| `type_k` / `type_v` | KV cache dtypes |
| `fa` | flash attention on/off |
| `mtp` | MTP draft depth (shown only when varied) |
| `test` | `pp512`, `tg128`, `pp512+tg128` etc. |
| `t/s` | throughput avg ± stddev |

---

## llama-batched-bench

### What it measures

`llama-batched-bench` measures throughput under **concurrent parallel sequences** — the scenario a server faces when handling multiple simultaneous requests. It sweeps three dimensions simultaneously:

- `-npp n0,n1,…` — prompt lengths per sequence
- `-ntg n0,n1,…` — generation lengths per sequence  
- `-npl n0,n1,…` — number of parallel sequences (batch width)

and for each combination it reports:

| Column | Meaning |
|---|---|
| `PP` | prompt tokens per sequence |
| `TG` | generation tokens per sequence |
| `B` | parallel sequences (batch width) |
| `N_KV` | total KV slots consumed |
| `T_PP s` | prompt phase wall time (seconds) |
| `S_PP t/s` | prompt throughput (tokens/s) |
| `T_TG s` | generation phase wall time |
| `S_TG t/s` | generation throughput |
| `T s` | total wall time |
| `S t/s` | combined throughput |

When MTP is active (see below), two extra columns appear:

| Column | Meaning |
|---|---|
| `D_ACC %` | draft acceptance rate — accepted/drafted × 100 |
| `SPEC_UP` | speculative speedup — effective tokens/verify-step |

### MTP support: what changed

Previously `llama-batched-bench` used `LLAMA_EXAMPLE_BENCH` which excluded the entire speculative-decoding parameter group from the argument parser. The tool is now registered as `LLAMA_EXAMPLE_SPECULATIVE`, which unlocks:

```
--spec-type <type>           none | draft-mtp | draft-simple | ngram-simple | …
--spec-draft-n-max N         max draft tokens per speculate step (default: 16)
--spec-draft-ngl N           GPU layers for draft model (0 = CPU)
--spec-draft-model <path>    explicit draft model path (auto-detected from GGUF if omitted)
--spec-draft-type-k <t>      draft KV-cache K dtype
--spec-draft-type-v <t>      draft KV-cache V dtype
```

The MTP path is only active for `--npl 1` (single sequence) runs — MTP does not support parallel sequences. Multi-sequence rows (`npl > 1`) automatically fall through to the standard decode loop. The tool prints a warning if `--npl` contains values > 1 while `--spec-type draft-mtp` is set.

### Key flags (full set)

```
-m, --model <path>           model GGUF path
-c, --ctx-size N             context window size
-b, --batch-size N           logical batch size
-ub, --ubatch-size N         physical micro-batch size
-ctk, --cache-type-k <t>     KV K dtype: f16, q8_0, turbo2, turbo3, turbo4, …
-ctv, --cache-type-v <t>     KV V dtype: same
-fa, --flash-attn on|off     flash attention
-ngl N                       GPU layers to offload
-npp n0,n1,…                 prompt lengths to sweep
-ntg n0,n1,…                 generation lengths to sweep
-npl n0,n1,…                 parallel sequence counts to sweep
-pps                         share prompt across sequences (server cache-hit scenario)
-tgs                         separate TG per sequence (interleave=false pattern)
--kv-unified / --no-kv-unified  unified KV buffer mode
--spec-type draft-mtp        enable MTP speculative generation
--spec-draft-n-max N         draft depth (default: 16)
--spec-draft-ngl N           draft model GPU layers (0 = CPU only)
--output-format md|jsonl     output format (default: md)
```

---

## Building bench-only

After the initial full build with `va-make-CUDA-nvidia.sh`, use the fast incremental target:

```bash
# NVIDIA CUDA RTX 4060 (default)
./va-make-bench-only.sh

# AMD HIP gfx1103
./va-make-bench-only.sh rocm

# Vulkan AMD 780M
./va-make-bench-only.sh vulkan
```

This invokes `cmake --build … --target llama-bench --target llama-batched-bench` — only the two bench objects recompile, not the server or any other tools.

---

## Variables used in all examples below

```bash
BIN_DIR=build-x64-linux-cuda-nvidia-RTX4060-release/bin
BENCH=$BIN_DIR/llama-bench
BBENCH=$BIN_DIR/llama-batched-bench

# Standard 9B model — no MTP heads
MODEL_STD=/mnt/512_ssd_internal/vai-llm-tei/llm-models/bartowski/Tesslate_OmniCoder-9B-Q4_K_M.gguf

# MTP-capable model — Qwen3 unsloth UD build with embedded MTP heads
MODEL_MTP=/mnt/512_ssd_internal/vai-llm-tei/llm-models/unsloth/Qwen3.6-27B-UD-Q4_K_XL.gguf
```

---

## Example 1 — Baseline KV-type ladder (llama-bench)

Compare f16, q8_0, turbo3, turbo4 V-cache with q8_0 K-cache in a single run.  
Produces one table with a `type_v` column showing all four variants.

```bash
$BENCH \
  -m "$MODEL_STD" \
  -ngl 99 \
  -fa 1 \
  -ctk q8_0 \
  -ctv f16,q8_0,turbo3,turbo4 \
  -p 512 \
  -n 128 \
  -r 3
```

### Script: `bench-kv-ladder.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

BIN_DIR=build-x64-linux-cuda-nvidia-RTX4060-release/bin
MODEL=/mnt/512_ssd_internal/vai-llm-tei/llm-models/bartowski/Tesslate_OmniCoder-9B-Q4_K_M.gguf
OUT=results/kv-ladder-$(date +%Y%m%d-%H%M%S).md

mkdir -p results

echo "# KV-type ladder — $(date)" | tee "$OUT"

$BIN_DIR/llama-bench \
  -m "$MODEL" \
  -ngl 99 \
  -fa 1 \
  -ctk q8_0 \
  -ctv f16,q8_0,turbo3,turbo4 \
  -p 512 \
  -n 128 \
  -r 3 \
  -o md \
  | tee -a "$OUT"

echo "Saved: $OUT"
```

---

## Example 2 — MTP depth sweep (llama-bench)

Compare baseline TG (no MTP) vs MTP depth 1, 2, 4 on a model with embedded heads.  
The `mtp` column appears automatically; all four rows use the same context/KV config.

```bash
$BENCH \
  -m "$MODEL_MTP" \
  -ngl 20 \
  -fa 1 \
  -ctk q8_0 \
  -ctv turbo3 \
  -n 128 \
  --mtp-n-max 0,1,2,4 \
  -r 3
```

### Script: `bench-mtp-sweep.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

BIN_DIR=build-x64-linux-cuda-nvidia-RTX4060-release/bin
MODEL=/mnt/512_ssd_internal/vai-llm-tei/llm-models/unsloth/Qwen3.6-27B-UD-Q4_K_XL.gguf
OUT=results/mtp-sweep-$(date +%Y%m%d-%H%M%S).md

mkdir -p results

echo "# MTP depth sweep — $(date)" | tee "$OUT"

$BIN_DIR/llama-bench \
  -m "$MODEL" \
  -ngl 20 \
  -fa 1 \
  -ctk q8_0 \
  -ctv turbo3 \
  -n 128 \
  --mtp-n-max 0,1,2,4 \
  -r 3 \
  -o md \
  | tee -a "$OUT"

echo "Saved: $OUT"
```

---

## Example 3 — PP+TG combined with context depth (llama-bench)

Use `-pg` to benchmark prompt+generation together, at multiple context depths.  
This exercises the KV cache at realistic fill levels.

```bash
$BENCH \
  -m "$MODEL_STD" \
  -ngl 99 \
  -fa 1 \
  -ctk q8_0 \
  -ctv turbo3 \
  -d 0,512,2048 \
  -pg 256,128 \
  -r 3
```

### Script: `bench-pg-depth.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

BIN_DIR=build-x64-linux-cuda-nvidia-RTX4060-release/bin
MODEL=/mnt/512_ssd_internal/vai-llm-tei/llm-models/bartowski/Tesslate_OmniCoder-9B-Q4_K_M.gguf
OUT=results/pg-depth-$(date +%Y%m%d-%H%M%S).md

mkdir -p results

echo "# PP+TG at context depth — $(date)" | tee "$OUT"

$BIN_DIR/llama-bench \
  -m "$MODEL" \
  -ngl 99 \
  -fa 1 \
  -ctk q8_0 \
  -ctv turbo3 \
  -d 0,512,2048 \
  -pg 256,128 \
  -r 3 \
  -o md \
  | tee -a "$OUT"

echo "Saved: $OUT"
```

---

## Example 4 — Multi-format JSONL output for CI logging (llama-bench)

Pipe JSONL to a file for downstream processing (plotting, regression detection).  
stderr gets Markdown for human review; stdout gets JSONL for machine parsing.

```bash
$BENCH \
  -m "$MODEL_STD" \
  -ngl 99 \
  -fa 1 \
  -ctk q8_0 \
  -ctv f16,turbo3,turbo4 \
  -p 128,512 \
  -n 64,128 \
  -r 3 \
  -o jsonl \
  -oe md
```

### Script: `bench-ci-log.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

BIN_DIR=build-x64-linux-cuda-nvidia-RTX4060-release/bin
MODEL=/mnt/512_ssd_internal/vai-llm-tei/llm-models/bartowski/Tesslate_OmniCoder-9B-Q4_K_M.gguf
TS=$(date +%Y%m%d-%H%M%S)
JSONL_OUT=results/bench-ci-$TS.jsonl
MD_OUT=results/bench-ci-$TS.md

mkdir -p results

echo "# CI bench run — $TS" > "$MD_OUT"

$BIN_DIR/llama-bench \
  -m "$MODEL" \
  -ngl 99 \
  -fa 1 \
  -ctk q8_0 \
  -ctv f16,turbo3,turbo4 \
  -p 128,512 \
  -n 64,128 \
  -r 3 \
  -o jsonl \
  -oe md \
  > "$JSONL_OUT" \
  2>> "$MD_OUT"

echo "JSONL: $JSONL_OUT"
echo "MD:    $MD_OUT"
echo "Rows:  $(wc -l < "$JSONL_OUT")"
```

---

## Example 5 — Parallel-sequence throughput sweep (llama-batched-bench)

Sweep prompt and generation sizes across 1, 2, 4, 8 parallel sequences.  
No MTP; baseline to establish server throughput ceiling before tuning KV types.

```bash
$BBENCH \
  -m "$MODEL_STD" \
  -ngl 99 \
  -fa on \
  -ctk q8_0 \
  -ctv f16 \
  -c 8192 \
  -b 2048 \
  -ub 512 \
  -npp 128,256,512 \
  -ntg 64,128 \
  -npl 1,2,4,8
```

### Script: `bbench-parallel-baseline.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

BIN_DIR=build-x64-linux-cuda-nvidia-RTX4060-release/bin
MODEL=/mnt/512_ssd_internal/vai-llm-tei/llm-models/bartowski/Tesslate_OmniCoder-9B-Q4_K_M.gguf
OUT=results/bbench-parallel-$(date +%Y%m%d-%H%M%S).md

mkdir -p results

echo "# Batched-bench parallel baseline — $(date)" | tee "$OUT"

$BIN_DIR/llama-batched-bench \
  -m "$MODEL" \
  -ngl 99 \
  -fa on \
  -ctk q8_0 \
  -ctv f16 \
  -c 8192 \
  -b 2048 \
  -ub 512 \
  -npp 128,256,512 \
  -ntg 64,128 \
  -npl 1,2,4,8 \
  | tee -a "$OUT"

echo "Saved: $OUT"
```

---

## Example 6 — TurboQuant KV sweep in batched mode (llama-batched-bench)

Compare f16, turbo3, turbo4 V-cache at several batch widths.  
Run with `-pps` (shared prompt) to simulate a cached system-prompt scenario.

```bash
$BBENCH \
  -m "$MODEL_STD" \
  -ngl 99 \
  -fa on \
  -ctk q8_0 \
  -ctv f16,turbo3,turbo4 \
  -c 8192 \
  -b 2048 \
  -ub 512 \
  -npp 256,512 \
  -ntg 128,256 \
  -npl 1,2,4 \
  -pps
```

### Script: `bbench-turbo-kv.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

BIN_DIR=build-x64-linux-cuda-nvidia-RTX4060-release/bin
MODEL=/mnt/512_ssd_internal/vai-llm-tei/llm-models/bartowski/Tesslate_OmniCoder-9B-Q4_K_M.gguf
OUT=results/bbench-turbo-kv-$(date +%Y%m%d-%H%M%S).md

mkdir -p results

echo "# Batched-bench TurboQuant KV sweep — $(date)" | tee "$OUT"

for KV_V in f16 turbo3 turbo4; do
  echo "## ctv=$KV_V" | tee -a "$OUT"
  $BIN_DIR/llama-batched-bench \
    -m "$MODEL" \
    -ngl 99 \
    -fa on \
    -ctk q8_0 \
    -ctv "$KV_V" \
    -c 8192 \
    -b 2048 \
    -ub 512 \
    -npp 256,512 \
    -ntg 128,256 \
    -npl 1,2,4 \
    -pps \
    | tee -a "$OUT"
  echo "" | tee -a "$OUT"
done

echo "Saved: $OUT"
```

---

## Example 7 — MTP speculative batched bench (llama-batched-bench)

Single-sequence MTP run on the Qwen3 UD model.  
Outputs `D_ACC %` (draft acceptance rate) and `SPEC_UP` (effective speedup multiplier).  
Draft context runs on CPU (`--spec-draft-ngl 0`) to avoid OOM on 8 GiB VRAM with ngl 20.

```bash
$BBENCH \
  -m "$MODEL_MTP" \
  -ngl 20 \
  -fa on \
  -ctk q8_0 \
  -ctv turbo3 \
  -c 4096 \
  -b 2048 \
  -ub 512 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --spec-draft-ngl 0 \
  -npp 128,256,512 \
  -ntg 64,128,256 \
  -npl 1
```

### Script: `bbench-mtp.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

BIN_DIR=build-x64-linux-cuda-nvidia-RTX4060-release/bin
MODEL=/mnt/512_ssd_internal/vai-llm-tei/llm-models/unsloth/Qwen3.6-27B-UD-Q4_K_XL.gguf
OUT=results/bbench-mtp-$(date +%Y%m%d-%H%M%S).md

mkdir -p results

echo "# Batched-bench MTP speculative — $(date)" | tee "$OUT"

for N_MAX in 1 2 4; do
  echo "## spec-draft-n-max=$N_MAX" | tee -a "$OUT"
  $BIN_DIR/llama-batched-bench \
    -m "$MODEL" \
    -ngl 20 \
    -fa on \
    -ctk q8_0 \
    -ctv turbo3 \
    -c 4096 \
    -b 2048 \
    -ub 512 \
    --spec-type draft-mtp \
    --spec-draft-n-max "$N_MAX" \
    --spec-draft-ngl 0 \
    -npp 128,256,512 \
    -ntg 64,128,256 \
    -npl 1 \
    | tee -a "$OUT"
  echo "" | tee -a "$OUT"
done

echo "Saved: $OUT"
```

---

## Example 8 — Full matrix: MTP × TurboQuant KV combined (llama-batched-bench JSONL)

The most comprehensive single run: sweeps three KV-V types against three MTP draft depths,
outputting JSONL for post-processing. This is the regression baseline to commit after any
codec or attention kernel change.

```bash
$BBENCH \
  -m "$MODEL_MTP" \
  -ngl 20 \
  -fa on \
  -ctk q8_0 \
  -ctv turbo3 \
  -c 4096 \
  -b 2048 \
  -ub 512 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --spec-draft-ngl 0 \
  -npp 128,256,512 \
  -ntg 128,256 \
  -npl 1 \
  --output-format jsonl
```

### Script: `bbench-full-matrix.sh`

```bash
#!/usr/bin/env bash
# Full MTP × TurboQuant KV regression matrix
# Loops over KV-V types; each inner run varies MTP depth.
# All output goes to a timestamped JSONL for plotting/diffing.
set -euo pipefail

BIN_DIR=build-x64-linux-cuda-nvidia-RTX4060-release/bin
MODEL=/mnt/512_ssd_internal/vai-llm-tei/llm-models/unsloth/Qwen3.6-27B-UD-Q4_K_XL.gguf
TS=$(date +%Y%m%d-%H%M%S)
JSONL=results/full-matrix-$TS.jsonl
MD=results/full-matrix-$TS.md

mkdir -p results

echo "# Full MTP × TurboQuant matrix — $TS" | tee "$MD"
echo "Model: $MODEL" | tee -a "$MD"
echo "" | tee -a "$MD"

for KV_V in f16 turbo3 turbo4; do
  for N_MAX in 0 1 2 4; do
    TAG="ctv=${KV_V} mtp=${N_MAX}"
    echo "## $TAG" | tee -a "$MD"

    MTP_FLAGS=()
    if (( N_MAX > 0 )); then
      MTP_FLAGS=(--spec-type draft-mtp --spec-draft-n-max "$N_MAX" --spec-draft-ngl 0)
    fi

    $BIN_DIR/llama-batched-bench \
      -m "$MODEL" \
      -ngl 20 \
      -fa on \
      -ctk q8_0 \
      -ctv "$KV_V" \
      -c 4096 \
      -b 2048 \
      -ub 512 \
      "${MTP_FLAGS[@]}" \
      -npp 128,256,512 \
      -ntg 128,256 \
      -npl 1 \
      --output-format jsonl \
      | while IFS= read -r line; do
          # tag each JSONL row with the sweep parameters
          printf '%s\n' "${line%\}}, \"ctv\": \"$KV_V\", \"mtp_n_max\": $N_MAX}" >> "$JSONL"
          # also print markdown-friendly summary to the MD file
          printf '%s\n' "$line" >> "$MD"
        done

    echo "" | tee -a "$MD"
  done
done

echo "" | tee -a "$MD"
echo "JSONL rows: $(wc -l < "$JSONL")"
echo "JSONL: $JSONL"
echo "MD:    $MD"
```

---

## Notes on model compatibility

| Model type | TurboQuant KV | MTP |
|---|---|---|
| Any standard GGUF (bartowski, TheBloke, etc.) | ✅ `-ctk`/`-ctv` turbo2/3/4 | ❌ no MTP heads — graceful fallback |
| unsloth `*-UD-*` Qwen3 builds | ✅ | ✅ embedded MTP heads auto-detected |
| Other models with `--spec-draft-model` | ✅ | ✅ if explicit draft model provided |

MTP draft heads are typically a ~100 MB sidecar embedded in the unsloth UD GGUF. They are auto-detected by `common_params_handle_models` when `--spec-type draft-mtp` is specified — no manual path required.

## Notes on hardware constraints (RTX 4060 Laptop, 8 GiB GDDR6)

- For the 27B Qwen3 model, use `-ngl 20` and `--spec-draft-ngl 0` (draft on CPU) to avoid OOM.
- Each additional GPU layer is ~400 MiB for the 27B model. With MTP active, the draft context adds ~50–100 MiB on GPU if `--spec-draft-ngl > 0`.
- TurboQuant KV types reduce KV cache VRAM by 2–4.5×, enabling higher `-ngl` values at the same peak VRAM.
- Flash attention (`-fa on`) is required whenever any quantized KV type is used (`q8_0`, `turbo2/3/4`).
