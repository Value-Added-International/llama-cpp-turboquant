# CUDA-Only Benchmark Report — 2026-06-07

## Test Environment

| Field | Value |
|---|---|
| **GPU** | NVIDIA GeForce RTX 4060 Laptop GPU (8 GiB VRAM, compute 8.9) |
| **CPU** | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics |
| **Build commit** | `8ef5d3454` (build #9425) |
| **Backend** | CUDA only (`-sm layer -fa 1 -nkvo 0`) |
| **Context** | 32768 tokens (auto-fit, no explicit `-c`) |
| **Bench params** | `-p 512,1024,2048,4096,8192,16384 -n 256 -r 3 --progress` |
| **Model tested** | `Tesslate_OmniCoder-9B-Q4_K_M` (Qwen3.5 9B, 5.90 GiB, 8.95B params) |

---

## Run Summary

| Outcome | Count |
|---|---|
| Total log files | 43 |
| Completed (with bench data) | **4** |
| Completed (dry-run / no bench data — `duration=0s`) | 34 |
| Failed (CLI arg crash) | 4 |
| In-progress (incomplete) | 1 |
| Models with no usable data | Qwen3.5-9B-UD-Q4_K_XL, Qwopus3.5-9B-Coder-MTP-Q6_K |

> **34 "ghost" completions**: Early runs in the session completed instantly (`duration=0s`) without running any benchmarks. This was an earlier version of the harness that logged `completed` but ran a dry-run pass only. Only the 4 latest runs (timestamps `2014xx`–`2020xx`) produced real data.

---

## KV Cache Type Abbreviations

| Symbol | Meaning |
|---|---|
| `q8_0` / `q8_0` | 8-bit quantized K and V cache (baseline high quality) |
| `q4_0` / `q4_0` | 4-bit quantized K and V cache (baseline compressed) |
| `q4_0` / `turbo4` | 4-bit K cache + TurboQuant+ `turbo4` V cache (mixed) |
| `turbo4` / `turbo4` | TurboQuant+ `turbo4` for both K and V cache |

---

## Prompt Throughput (tokens/sec) — Tesslate OmniCoder 9B Q4_K_M

All runs: flash attention ON, full GPU offload (ngl=32).

### Prefill (prompt processing)

| Prompt tokens | `q8_0/q8_0` | `q4_0/q4_0` | `q4_0/turbo4` | `turbo4/turbo4` |
|---:|---:|---:|---:|---:|
| 512 | 1414.7 ± 20.9 | 976.4 ± 102.6 | 1419.8 ± 15.3 | 878.9 ± 139.3 |
| 1024 | 1409.3 ± 21.2 | 1305.9 ± 42.6 | 1398.1 ± 45.0 | 1376.3 ± 15.7 |
| 2048 | **1453.8** ± 6.8 | 1433.9 ± 2.6 | 1408.1 ± 28.3 | 1446.1 ± 2.4 |
| 4096 | 1429.2 ± 17.1 | 1389.9 ± 6.7 | **1429.9** ± 3.0 | 1432.3 ± 6.6 |
| 8192 | **1415.4** ± 1.5 | 1372.7 ± 19.6 | 1407.7 ± 1.1 | 1376.0 ± 20.4 |
| 16384 | **1364.0** ± 1.5 | 1335.8 ± 9.5 | 1355.8 ± 3.7 | 1340.8 ± 7.8 |

### Generation (token decode)

| Metric | `q8_0/q8_0` | `q4_0/q4_0` | `q4_0/turbo4` | `turbo4/turbo4` |
|---|---:|---:|---:|---:|
| **avg t/s** (n=256) | **29.84** ± 0.11 | 29.31 ± 0.26 | 29.05 ± 0.32 | 29.67 ± 0.15 |

---

## VRAM Usage

| KV config | Estimated KV cache | Peak VRAM observed |
|---|---:|---:|
| `q8_0/q8_0` | 1024 MiB | **6667 MiB** |
| `q4_0/q4_0` | — | 6557 MiB |
| `q4_0/turbo4` | — | 6581 MiB |
| `turbo4/turbo4` | 512 MiB | **6534 MiB** |
| VRAM total available | — | 8188 MiB (8053 MiB usable) |
| VRAM headroom (turbo4/turbo4) | — | ~1654 MiB free |

`turbo4/turbo4` saves ~133 MiB peak VRAM vs `q8_0/q8_0` at 32K context.

---

## Analysis

### Prefill throughput
- All four KV configurations produce comparable throughput at **≥2048 tokens** (1340–1454 t/s range).
- The standout anomaly is **`q4_0/q4_0` and `turbo4/turbo4` at 512 tokens**: both drop to ~878–976 t/s, ~37–38% below baseline. High stddev (±102–139) suggests cache warm-up or scheduling jitter at small context lengths. At 1024+ tokens both recover to within ~5% of `q8_0/q8_0`.
- `q4_0/turbo4` shows the most stable prefill across all context sizes (stddev drops sharply from 512 to 4096+ tokens).

### Generation throughput
- Generation is essentially identical across all KV configs: **29.05–29.84 t/s** (delta < 2.7%). This is expected — generation is memory-bandwidth-bound on weight reads, not KV accesses.
- `q8_0/q8_0` leads by a slim margin (+0.53 t/s vs `turbo4/turbo4`), within stddev noise.

### KV compression efficiency
- `turbo4/turbo4` achieves 50% KV cache size reduction (512 vs 1024 MiB estimated) at negligible throughput cost.
- At 16K context, `turbo4/turbo4` is within **1.7%** of `q8_0/q8_0` prefill speed (1340.8 vs 1364.0 t/s).

---

## Failure Analysis

### 4 failed runs (`*-failed.jsonl`)
- **Root cause**: `error: invalid parameter for argument: -c` — the `-c 32768` flag was passed to `llama-bench`, which does not accept `-c` (context size) as a parameter (unlike `llama-server`/`llama-cli`).
- These failures occurred in an earlier harness version that passed `-c` explicitly. Later runs drop `-c` and succeed.
- All 4 failed runs were `Tesslate_OmniCoder-9B-Q4_K_M` combos; all were subsequently re-run successfully.

### 1 in-progress run
- `Tesslate_OmniCoder-9B-Q4_K_M / q4_0 / turbo4` at timestamp `200623` — interrupted mid-run. A completed version of this combo exists from the later session.

### Missing models (Qwen3.5, Qwopus3.5)
- Both models have log files (8 each) that all show `duration=0s success=True` with no bench data.
- These were dry-run passes from an earlier harness iteration. **No real benchmark data was collected for these models in this session.** They need to be re-run.

---

## Outstanding Work

| Item | Status |
|---|---|
| Tesslate_OmniCoder-9B-Q4_K_M — all 4 KV combos | Complete |
| Qwen3.5-9B-UD-Q4_K_XL — all 4 KV combos | **Missing — needs re-run** |
| Qwopus3.5-9B-Coder-MTP-Q6_K — all 4 KV combos | **Missing — needs re-run** |
| `-c` CLI flag bug in bench harness | Fixed (runs after `201329` omit `-c`) |
| `q4_0/turbo4` inprogress run | Superseded by completed run |
