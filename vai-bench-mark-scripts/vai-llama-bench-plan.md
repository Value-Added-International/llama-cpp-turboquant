# VAI llama-bench Test Series — Implementation Plan

**Date:** 2026-06-07
**Branch:** rocm10_aocc
**Author:** none (plan recorded by Claude Code, implementation by human-directed AI assistance)

---

## Objective

Produce a repeatable, crash-tolerant benchmark suite that measures token-generation throughput and VRAM consumption for three Gemma 4 QAT models across three llama.cpp backends (CUDA, ROCm/HIP, Vulkan). The primary research question is:

> **What is the maximum practical context window per backend × model, and what is the token-generation rate at each context size under various KV-cache quantizations?**

---

## Files to Create

| File | Purpose |
|---|---|
| `vai-bench-mark-scripts/vai-llama-bench-runner.py` | Main executable (shebang `#!/usr/bin/env python3`, `chmod +x`) |
| `vai-bench-mark-scripts/vai-bench-config.json` | Config: models, backends, CUDA params, context sweeps |

---

## Source Data Used in Planning

### Bench reference (`vai-model-12B-31B.md`)
- Base bench command: `./llama-bench -m model.gguf -ngl 99 -fa 1 -ctk turbo4 -ctv q4_0 -p 512,2048,4096,8192 -n 512`
- Three llama-bench binaries: CUDA (RTX4060), HIP/ROCm (gfx1103), Vulkan (AMD 780M)
- Three models: gemma-4-12B, gemma-4-26B-A4B (MoE), gemma-4-31B
- Log path: `lamma-bench-logs/llama-bench-[MODEL]-[Backend]-[timestamp].jsonl`

### Model cards (all three share the same recommended settings)
- `temperature = 1.0`, `top_p = 0.95`, `top_k = 64`
- Context window: 256K tokens (12B, 26B A4B, 31B)
- 12B: 48 layers (dense), 26B A4B: 30 layers (MoE, 3.8B active), 31B: 60 layers (dense)
- Thinking mode and function calling are model capabilities — not direct llama-bench parameters
- No image/video/audio processing required (text-only thinking mode + tool calling)

### Hardware
- **CUDA:** NVIDIA GeForce RTX 4060 Laptop GPU — 8053 MiB VRAM (nvidia-smi confirmed 8188 MiB total, 6926 MiB free at idle)
- **ROCm/HIP:** AMD Radeon 780M iGPU (gfx1103) — 8 GiB VRAM shared from system RAM (rocm-smi confirmed)
- **Vulkan:** same AMD 780M via Vulkan backend

**Critical constraint:** All three models exceed 8 GB raw; none will fit fully in VRAM on any backend. Layer offload (`-ngl`) must be tuned per model × backend to maximise GPU utilisation without OOM.

---

## Part 1 — Configuration File (`vai-bench-config.json`)

```json
{
  "bench_params": {
    "kv_types": [
      {"ctk": "turbo4", "ctv": "q4_0"},
      {"ctk": "turbo3", "ctv": "q4_0"},
      {"ctk": "q8_0",   "ctv": "q8_0"}
    ],
    "prompts":  [512, 2048, 4096, 8192],
    "n_gen":    512,
    "flash_attn": 1,
    "repetitions": 3
  },
  "backends": {
    "cuda": {
      "binary": "/home/none/git/llama-cpp-turboquant/build-x64-linux-cuda-nvidia-RTX4060-release/bin/llama-bench",
      "enabled": true,
      "split_mode": "layer",
      "vram_monitor": "nvidia-smi"
    },
    "rocm": {
      "binary": "/home/none/git/llama-cpp-turboquant/build-x64-linux-hip-rocm713-gfx1103-release/bin/llama-bench",
      "enabled": true,
      "split_mode": "layer",
      "vram_monitor": "rocm-smi"
    },
    "vulkan": {
      "binary": "/home/none/git/llama-cpp-turboquant/build-x64-linux-vulkan-AMD-780-release/bin/llama-bench",
      "enabled": true,
      "split_mode": "layer",
      "vram_monitor": "rocm-smi"
    }
  },
  "models": [
    {
      "name": "gemma-4-12B-it-qat",
      "path": "/mnt/512_ssd_internal/...",
      "total_layers": 48,
      "ngl_cuda_candidates": [20, 25, 30, 35, 40, 45, 48],
      "ngl_rocm_candidates": [10, 15, 20, 25, 30],
      "ngl_vulkan_candidates": [5, 10, 15, 20]
    }
  ]
}
```

**`ngl_*_candidates`:** Ordered list of layer-offload counts to try per model × backend. The resolver picks the highest that fits within available VRAM minus a 512 MiB safety margin.

---

## Part 2 — Script Architecture (`vai-llama-bench-runner.py`)

### Module layout (single file)

```
vai-llama-bench-runner.py
├── VRAMMonitor class         — nvidia-smi / rocm-smi abstraction
├── Config Loader             — load and validate vai-bench-config.json
├── NGL Resolver              — pick best layer-offload count from candidates
├── VRAM Telemetry Thread     — background thread sampling VRAM every N seconds
├── Watchdog Runner           — inner loop: Popen + timeout + crash retry
├── Bench Orchestrator        — outer loop: backends → models → kv_types
└── main() + argparse         — CLI entry point
```

---

## Part 3 — Step-by-Step Logic

### Step 1 — Startup / Validation
1. Parse CLI: `--config path`, `--backend cuda|rocm|vulkan|all`, `--model name|all`, `--dry-run`
2. Load JSON config
3. Verify all binary paths exist and are executable
4. Verify all `.gguf` model files exist
5. Create `log_dir` if absent

### Step 2 — VRAM Discovery (`VRAMMonitor`)

| Backend | Tool | Command |
|---|---|---|
| CUDA (Nvidia) | `nvidia-smi` | `nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader,nounits` |
| ROCm / Vulkan (AMD) | `rocm-smi` | `rocm-smi --showmeminfo vram --json` → parse used/total bytes |

Returns `free_mib` and `used_mib` as integers.

### Step 3 — NGL Resolver

```
free_vram = query_free_vram_mib(backend)
safety_margin = 512  # MiB
usable = free_vram - safety_margin

# Heuristic: vram_per_layer ≈ (gguf_file_size_MiB / total_layers) * 1.15
# Select highest candidate where (candidate * vram_per_layer) <= usable
```

The chosen `ngl` is logged. If no candidate fits, the run is skipped with a `SKIP_OOM` event.

### Step 4 — VRAM Telemetry Thread

A `threading.Thread` runs for the lifetime of each llama-bench subprocess. Every `vram_sample_interval_s` (default 5 s) it appends a JSON line to the log:

```json
{"event": "vram_sample", "ts": 1749280000.1, "backend": "cuda",
 "vram_used_mib": 6843, "vram_free_mib": 1210, "phase": "prefill_8192"}
```

The main thread updates a shared `phase` string at transitions: `model_load`, `prefill_NNN`, `gen_NNN`, `unload`.

### Step 5 — Watchdog Runner (inner loop)

```python
def run_with_watchdog(cmd, log_path, timeout_s, max_retries):
    for attempt in range(max_retries + 1):
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh)
        start = time.monotonic()
        while proc.poll() is None:
            if time.monotonic() - start > timeout_s:
                proc.kill()
                log_event("TIMEOUT", attempt=attempt)
                break
            time.sleep(2)
        if proc.returncode == 0:
            return True
        log_event("CRASH", returncode=proc.returncode, attempt=attempt)
        if attempt < max_retries:
            time.sleep(10)
    return False
```

- `timeout_s = 1800` (30 min) — covers largest model on slowest backend
- On crash: log event, pause 10 s, retry up to `max_retries` (default 2)
- After all retries exhausted: mark FAILED, continue outer loop

### Step 6 — Bench Orchestrator (outer loop)

```
for backend in enabled_backends:
  for model in models:
    free_vram = VRAMMonitor.query_free(backend)
    ngl = resolve_ngl(model, backend, free_vram)
    log_event("run_config", ngl=ngl, free_vram=free_vram)

    for kv in bench_params.kv_types:
      log_path = f"{log_dir}/llama-bench-{model.name}-{BACKEND}-{ctk}-{ctv}-{timestamp}.jsonl"
      cmd = build_cmd(backend, model, ngl, kv, bench_params)

      telemetry = VRAMTelemetryThread(log_path, backend)
      telemetry.start()
      success = run_with_watchdog(cmd, log_path, timeout_s, max_retries)
      telemetry.stop()

      log_event("run_end", success=success)
```

### Step 7 — llama-bench Command Construction

```
{binary} \
  -m {model.path} \
  -ngl {ngl} \
  -sm {split_mode} \
  -fa {flash_attn} \
  -ctk {ctk} \
  -ctv {ctv} \
  -p {','.join(prompts)} \
  -n {n_gen} \
  -r {repetitions} \
  -o jsonl \
  --progress
```

For CUDA: `-sm layer` (required to split across GPU/CPU layers).
ROCm/Vulkan: `-sm layer` as default; can be overridden per backend in config.

### Step 8 — Log File Format (JSONL, one file per run)

```jsonl
{"event": "run_start",    "model": "gemma-4-12B-it-qat", "backend": "cuda", "ngl": 35, "ctk": "turbo4", "ctv": "q4_0", "ts": 1749280000}
{"event": "vram_pre_load","vram_free_mib": 6926, "vram_used_mib": 1128, "ts": 1749280001}
{"event": "vram_sample",  "phase": "running", "vram_used_mib": 7340, "vram_free_mib": 713, "ts": 1749280010}
... (llama-bench native JSONL result lines) ...
{"event": "run_end",      "success": true, "duration_s": 743, "ts": 1749280750}
```

---

## Part 4 — Design Decisions

| Decision | Rationale |
|---|---|
| Shebang `#!/usr/bin/env python3` + `chmod +x` | Runs as `./vai-llama-bench-runner.py` without `python3 script.py` |
| `subprocess.Popen` + 2-second poll loop | Enables timeout kill and crash detection without blocking the telemetry thread |
| Background `threading.Thread` for VRAM | Samples independently of llama-bench I/O; doesn't interfere with stdout capture |
| Per-model `ngl_*_candidates` list in config | Different models have different per-layer VRAM cost; list gives manual override capability |
| JSONL output (`-o jsonl`) | llama-bench native; easily `jq`-parsed; all events in one stream per run |
| `--progress` flag | Gives live stderr progress even when stdout is redirected to log |
| Retry up to `max_retries` with 10 s pause | Handles transient driver crashes or VRAM fragmentation without human intervention |
| 512 MiB safety margin in NGL resolver | Prevents OOM; GPU needs scratch space beyond static weight allocation |
| Separate log per (model × backend × kv_type) | Crash isolation — one bad run doesn't corrupt others |
| `--dry-run` flag | Prints all commands without executing; useful for review before a long run |

---

## Part 5 — Expected Output Layout

```
lamma-bench-logs/
├── llama-bench-gemma-4-12B-it-qat-CUDA-turbo4-q4_0-20260607-143022.jsonl
├── llama-bench-gemma-4-12B-it-qat-CUDA-turbo3-q4_0-20260607-145811.jsonl
├── llama-bench-gemma-4-12B-it-qat-CUDA-q8_0-q8_0-20260607-152244.jsonl
├── llama-bench-gemma-4-12B-it-qat-ROCM-turbo4-q4_0-20260607-160000.jsonl
│   ... (3 models × 3 backends × 3 kv_types = 27 log files total)
└── llama-bench-gemma-4-31B-it-qat-VULKAN-q8_0-q8_0-20260608-020000.jsonl
```

Total: **27 log files** (3 models × 3 backends × 3 KV quantization combos).
Estimated wall time: 6–18 hours depending on which backends are enabled and model load times.

---

## Part 6 — Usage

```bash
# Run all backends, all models
./vai-bench-mark-scripts/vai-llama-bench-runner.py

# Run only CUDA backend
./vai-bench-mark-scripts/vai-llama-bench-runner.py --backend cuda

# Run only one model
./vai-bench-mark-scripts/vai-llama-bench-runner.py --model gemma-4-12B-it-qat

# Preview commands without running
./vai-bench-mark-scripts/vai-llama-bench-runner.py --dry-run

# Custom config location
./vai-bench-mark-scripts/vai-llama-bench-runner.py --config /path/to/custom-config.json
```
