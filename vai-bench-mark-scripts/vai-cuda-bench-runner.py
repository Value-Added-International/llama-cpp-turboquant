#!/usr/bin/env python3
"""
VAI llama-bench CUDA-ONLY runner — 2026-06-07
Benchmark suite for NVIDIA RTX 4060 dGPU — code-completion, tool-use, and MCP server workloads.
Tests turbo4/turbo4, q8_0/q8_0, q4_0/turbo4, q4_0/q4_0 KV quantizations to maximize context
window on 8GiB VRAM. Supports standard models and MTP (multi-token prediction) models.

Usage:
    ./vai-cuda-bench-runner.py [--config path] [--model name|all]
                               [--dry-run] [--show-last-cmd] [--list-models]

  --show-last-cmd   Read the most recent CUDA-ONLY log and print the exact llama-bench
                    command, with each flag on its own line. Verifies -nkvo flag state.

Log file naming
---------------
  CUDA-ONLY-{UUIDv7}-{model}-{ctk}-{ctv}-{ts}-{status}.jsonl

  Status suffix written at end of run:
    -completed   run finished successfully
    -skipped     VRAM pre-flight blocked the run (model+KV won't fit)
    -failed      all retries exhausted or timeout hit

VRAM pre-flight check
---------------------
  Before each run the runner estimates model VRAM + KV cache size at the configured
  context size. If the total exceeds available free VRAM minus the safety margin the
  run is skipped and logged as -skipped rather than letting the process OOM-crash.

  The RTX 4060 has 8 GiB GDDR6. At idle ~650 MiB is consumed by the OS/driver.
  Remaining ~7400 MiB is available. Safety margin (default 400 MiB) is reserved for
  CUDA runtime, cuBLAS workspace, and activation buffers during forward passes.

KV-cache quantization — context-window impact on 8GiB VRAM (qwen35 9B models)
===============================================================================
  ctk/ctv         | Bytes/token/layer | Max ctx (9B Q4_K_M) | Notes
  ----------------|-------------------|---------------------|-------------------------
  turbo4 / turbo4 | ~0.5 + 0.5 = 1.0 | ~32K                | Best ctx / acceptable quality
  q4_0   / turbo4 | ~0.5 + 0.5 = 1.0 | ~32K                | Asymmetric; K4 + V turbo4
  q4_0   / q4_0   | ~0.5 + 0.5 = 1.0 | ~32K                | Standard 4-bit; wider compat
  q8_0   / q8_0   | ~1.0 + 1.0 = 2.0 | ~16K                | Quality baseline, half the ctx

  MTP model (Q6_K, 7.1GiB): requires NGL≤28; turbo4 allows 8K-16K; q8_0 may OOM at 16K.

NOT REGISTERED (will GGML_ABORT — do not use):
  turbo4/q4_0, turbo4/q4_1, turbo3/q4_0, turbo3/q4_1, turbo2/q4_0, turbo2/q4_1
  turbo*/bf16 variants and their reverses
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# UUIDv7 — installed via: pipx install uuid-v7
# ---------------------------------------------------------------------------

_UUID_V7_SITE = Path.home() / ".local/share/pipx/venvs/uuid-v7/lib"


def _find_uuid_v7_site() -> Path | None:
    if _UUID_V7_SITE.exists():
        for py_dir in sorted(_UUID_V7_SITE.iterdir(), reverse=True):
            sp = py_dir / "site-packages"
            if sp.exists():
                return sp
    return None


_uuid_v7_sp = _find_uuid_v7_site()
if _uuid_v7_sp and str(_uuid_v7_sp) not in sys.path:
    sys.path.insert(0, str(_uuid_v7_sp))

try:
    from uuid_v7.base import uuid7 as _uuid7
    def make_run_uuid() -> str:
        return str(_uuid7())
except ImportError:
    import uuid as _uuid
    def make_run_uuid() -> str:  # type: ignore[misc]
        return str(_uuid.uuid4())
    print("WARNING: uuid_v7 not importable; falling back to UUIDv4. "
          "Run: pipx install uuid-v7", file=sys.stderr)


# ---------------------------------------------------------------------------
# VRAM monitoring — nvidia-smi only (CUDA-ONLY build)
# ---------------------------------------------------------------------------

class VRAMMonitor:
    """nvidia-smi VRAM queries for NVIDIA dGPU."""

    def snapshot(self) -> dict:
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=memory.free,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).decode().strip()
            free, used, total = [int(x.strip()) for x in out.split(",")]
            return {"free_mib": free, "used_mib": used, "total_mib": total}
        except Exception as exc:
            return {"free_mib": 0, "used_mib": 0, "total_mib": 0, "error": str(exc)}

    def free_mib(self) -> int:
        return self.snapshot().get("free_mib", 0)


# ---------------------------------------------------------------------------
# VRAM pre-flight check — estimate model + KV cache fit
# ---------------------------------------------------------------------------

def estimate_kv_cache_mib(
    n_ctx: int,
    n_layers: int,
    kv_head_count: int,
    head_dim: int,
    ctk: str,
    ctv: str,
) -> float:
    """Estimate KV cache VRAM in MiB for given context and quant types."""
    BYTES_PER_ELEM = {
        "f16":    2.0,
        "bf16":   2.0,
        "q8_0":   1.0,
        "q4_0":   0.5,
        "q4_1":   0.5,
        "turbo4": 0.5,
        "turbo3": 0.375,
        "turbo2": 0.25,
    }
    bk = BYTES_PER_ELEM.get(ctk, 2.0)
    bv = BYTES_PER_ELEM.get(ctv, 2.0)
    k_bytes = n_ctx * n_layers * kv_head_count * head_dim * bk
    v_bytes = n_ctx * n_layers * kv_head_count * head_dim * bv
    return (k_bytes + v_bytes) / (1024 * 1024)


def vram_preflight(
    model: dict,
    ngl: int,
    n_ctx: int,
    ctk: str,
    ctv: str,
    free_mib: int,
    safety_margin_mib: int = 400,
) -> dict:
    """
    Estimate whether model-at-ngl + KV-at-ctx fits in free_mib VRAM.

    Returns:
      safe     : bool
      comment  : str
      estimate : dict  — breakdown of the VRAM budget
    """
    total_layers  = model.get("total_layers", 1)
    kv_heads      = model.get("kv_head_count", 4)   # qwen35 GQA: 4 KV heads
    head_dim      = model.get("head_dim", 128)

    # Model weight VRAM: proportional to NGL fraction, plus 5% for CUDA buffers
    try:
        model_file_mib = os.path.getsize(model["path"]) // (1024 * 1024)
    except OSError:
        model_file_mib = 0

    ngl_fraction   = ngl / total_layers if total_layers > 0 else 1.0
    model_vram_mib = model_file_mib * ngl_fraction * 1.05

    kv_mib = estimate_kv_cache_mib(n_ctx, total_layers, kv_heads, head_dim, ctk, ctv)

    total_needed = model_vram_mib + kv_mib + safety_margin_mib
    safe         = total_needed <= free_mib

    comment = (
        f"model({ngl}/{total_layers} layers)={model_vram_mib:.0f}MiB + "
        f"KV(ctx={n_ctx} {ctk}/{ctv})={kv_mib:.0f}MiB + "
        f"safety={safety_margin_mib}MiB = {total_needed:.0f}MiB "
        f"{'<= ' if safe else '> '}{free_mib}MiB free"
    )

    return {
        "safe":            safe,
        "comment":         comment,
        "estimate": {
            "model_vram_mib":   round(model_vram_mib),
            "kv_cache_mib":     round(kv_mib),
            "safety_mib":       safety_margin_mib,
            "total_needed_mib": round(total_needed),
            "free_mib":         free_mib,
        },
    }


# ---------------------------------------------------------------------------
# NGL resolver — pick highest layer count that fits in available VRAM
# ---------------------------------------------------------------------------

def resolve_ngl(model: dict, free_mib: int, safety_margin_mib: int = 400) -> int:
    candidates = model.get("ngl_cuda_candidates", [])
    if not candidates:
        return 0

    try:
        file_size_mib = os.path.getsize(model["path"]) // (1024 * 1024)
    except OSError:
        return candidates[0]

    total_layers  = model.get("total_layers", 1)
    usable_mib    = free_mib - safety_margin_mib

    for ngl in candidates:
        fraction      = ngl / total_layers
        model_vram    = file_size_mib * fraction * 1.05
        if model_vram <= usable_mib:
            return ngl
    return 0


# ---------------------------------------------------------------------------
# Log filename helpers
# ---------------------------------------------------------------------------

LOG_TAG = "CUDA-ONLY"


def make_log_stem(run_uuid: str, model_name: str, ctk: str, ctv: str, ts: str) -> str:
    """
    Build log filename stem: CUDA-ONLY-{UUIDv7}-{model}-{ctk}-{ctv}-{ts}
    """
    return f"{LOG_TAG}-{run_uuid}-{model_name}-{ctk}-{ctv}-{ts}"


def log_path_for(log_dir: Path, stem: str, status: str) -> Path:
    return log_dir / f"{stem}-{status}.jsonl"


def finalise_log(log_dir: Path, stem: str, status: str,
                 current_path: Path | None) -> Path:
    final = log_path_for(log_dir, stem, status)
    if current_path and current_path.exists() and current_path != final:
        current_path.rename(final)
    return final


# ---------------------------------------------------------------------------
# VRAM telemetry thread
# ---------------------------------------------------------------------------

class VRAMTelemetryThread(threading.Thread):
    def __init__(self, log_path: Path, monitor: VRAMMonitor, interval_s: int = 5):
        super().__init__(daemon=True)
        self.log_path      = log_path
        self.monitor       = monitor
        self.interval_s    = interval_s
        self._stop_event   = threading.Event()
        self.phase         = "running"
        self._phase_lock   = threading.Lock()
        self.peak_used_mib = 0

    def set_phase(self, phase: str):
        with self._phase_lock:
            self.phase = phase

    def stop(self):
        self._stop_event.set()
        self.join(timeout=self.interval_s + 2)

    def run(self):
        while not self._stop_event.is_set():
            snap = self.monitor.snapshot()
            used = snap.get("used_mib", 0)
            if used > self.peak_used_mib:
                self.peak_used_mib = used
            with self._phase_lock:
                current_phase = self.phase
            entry = {
                "event":          "vram_sample",
                "backend":        "cuda",
                "log_tag":        LOG_TAG,
                "ts":             time.time(),
                "phase":          current_phase,
                "vram_used_mib":  used,
                "vram_free_mib":  snap.get("free_mib", 0),
                "vram_total_mib": snap.get("total_mib", 0),
            }
            try:
                with open(self.log_path, "a") as fh:
                    fh.write(json.dumps(entry) + "\n")
            except OSError:
                pass
            self._stop_event.wait(timeout=self.interval_s)


# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------

def log_event(log_path: Path, event: str, **kwargs):
    entry = {"event": event, "backend": "cuda", "log_tag": LOG_TAG,
             "ts": time.time(), **kwargs}
    with open(log_path, "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    tag     = f"[{event.upper()}]"
    details = "  ".join(f"{k}={v}" for k, v in kwargs.items()
                        if k not in ("cmd", "traceback"))
    print(f"  {tag} {details}", flush=True)


# ---------------------------------------------------------------------------
# Tee reader thread
# ---------------------------------------------------------------------------

class TeeReaderThread(threading.Thread):
    def __init__(self, pipe, log_path: Path):
        super().__init__(daemon=True)
        self.pipe          = pipe
        self.log_path      = log_path
        self.bench_results: list[dict] = []

    def run(self):
        try:
            with open(self.log_path, "a", buffering=1) as log_fh:
                for raw_line in self.pipe:
                    line = raw_line.decode("utf-8", errors="replace")
                    log_fh.write(line)
                    log_fh.flush()
                    print(f"    | {line}", end="", flush=True)
                    stripped = line.strip()
                    if stripped.startswith("{") and "avg_ts" in stripped:
                        try:
                            self.bench_results.append(json.loads(stripped))
                        except json.JSONDecodeError:
                            pass
        except Exception as exc:
            try:
                with open(self.log_path, "a") as fh:
                    fh.write(json.dumps({"event": "tee_reader_error",
                                         "backend": "cuda",
                                         "log_tag": LOG_TAG,
                                         "ts": time.time(),
                                         "error": str(exc)}) + "\n")
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Result summary printer
# ---------------------------------------------------------------------------

def print_result_summary(bench_results: list[dict], peak_vram_mib: int,
                         log_path: Path):
    if not bench_results:
        print("  [SUMMARY] No llama-bench result lines captured.", flush=True)
        return

    prompt_rows, gen_rows = [], []
    for r in bench_results:
        n_prompt = r.get("n_prompt", 0)
        n_gen    = r.get("n_gen", 0)
        avg_ts   = r.get("avg_ts", 0.0)
        stddev   = r.get("stddev_ts", 0.0)
        type_k   = r.get("type_k", "?")
        type_v   = r.get("type_v", "?")
        if n_prompt > 0 and n_gen == 0:
            prompt_rows.append((n_prompt, avg_ts, stddev, type_k, type_v))
        elif n_gen > 0 and n_prompt == 0:
            gen_rows.append((n_gen, avg_ts, stddev, type_k, type_v))

    sep = "  " + "-" * 62
    print(f"\n{sep}", flush=True)
    print(f"  [CUDA-ONLY] PERFORMANCE SUMMARY  —  peak VRAM used: {peak_vram_mib} MiB",
          flush=True)
    print(sep, flush=True)

    summary_entries = []

    if prompt_rows:
        print(f"  PROMPT PROCESSING (prefill speed):", flush=True)
        print(f"  {'tokens':>8}  {'tok/s avg':>12}  {'±stddev':>10}  ctk/ctv",
              flush=True)
        print(f"  {'-------':>8}  {'-----------':>12}  {'---------':>10}  -------",
              flush=True)
        for n_prompt, avg_ts, stddev, ctk, ctv in prompt_rows:
            print(f"  {n_prompt:>8}  {avg_ts:>12.2f}  {stddev:>10.2f}  {ctk}/{ctv}",
                  flush=True)
            summary_entries.append({"type": "prompt", "n_prompt": n_prompt,
                                    "avg_ts": avg_ts, "stddev_ts": stddev,
                                    "ctk": ctk, "ctv": ctv})

    if gen_rows:
        print(f"\n  TOKEN GENERATION (decode speed):", flush=True)
        print(f"  {'tokens':>8}  {'tok/s avg':>12}  {'±stddev':>10}  ctk/ctv",
              flush=True)
        print(f"  {'-------':>8}  {'-----------':>12}  {'---------':>10}  -------",
              flush=True)
        for n_gen, avg_ts, stddev, ctk, ctv in gen_rows:
            print(f"  {n_gen:>8}  {avg_ts:>12.2f}  {stddev:>10.2f}  {ctk}/{ctv}",
                  flush=True)
            summary_entries.append({"type": "gen", "n_gen": n_gen,
                                    "avg_ts": avg_ts, "stddev_ts": stddev,
                                    "ctk": ctk, "ctv": ctv})

    print(f"  peak VRAM: {peak_vram_mib} MiB", flush=True)
    print(sep, flush=True)

    summary_event = {
        "event":          "run_perf_summary",
        "backend":        "cuda",
        "log_tag":        LOG_TAG,
        "ts":             time.time(),
        "peak_vram_mib":  peak_vram_mib,
        "results":        summary_entries,
    }
    try:
        with open(log_path, "a") as fh:
            fh.write(json.dumps(summary_event) + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Watchdog runner
# ---------------------------------------------------------------------------

def run_with_watchdog(
    cmd: list,
    log_path: Path,
    timeout_s: int,
    max_retries: int,
    telemetry: VRAMTelemetryThread,
    dry_run: bool = False,
) -> tuple[bool, list[dict]]:
    if dry_run:
        log_event(log_path, "dry_run_cmd", cmd=" ".join(cmd))
        return True, []

    all_bench_results: list[dict] = []

    for attempt in range(max_retries + 1):
        log_event(log_path, "attempt_start", attempt=attempt, cmd=" ".join(cmd))

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            log_event(log_path, "launch_error", error=str(exc), attempt=attempt)
            return False, all_bench_results

        tee = TeeReaderThread(proc.stdout, log_path)
        tee.start()

        start     = time.monotonic()
        timed_out = False
        while proc.poll() is None:
            if time.monotonic() - start > timeout_s:
                proc.kill()
                proc.wait()
                timed_out = True
                log_event(log_path, "timeout",
                          elapsed_s=int(time.monotonic() - start),
                          attempt=attempt)
                break
            time.sleep(2)

        tee.join(timeout=10)
        all_bench_results.extend(tee.bench_results)

        rc = proc.returncode
        if not timed_out and rc == 0:
            log_event(log_path, "attempt_success", attempt=attempt,
                      duration_s=int(time.monotonic() - start))
            return True, all_bench_results

        if not timed_out:
            log_event(log_path, "crash", returncode=rc, attempt=attempt,
                      duration_s=int(time.monotonic() - start))

        if attempt < max_retries:
            log_event(log_path, "retry_wait", wait_s=10, next_attempt=attempt + 1)
            time.sleep(10)

    return False, all_bench_results


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def build_cmd(binary: str, model: dict, ngl: int, n_ctx: int,
              kv: dict, bench_params: dict) -> list:
    n_gen = bench_params["n_gen"]
    # Only include prompt sizes that fit within the resolved context window.
    # llama-bench sets n_ctx = n_prompt + n_gen internally; no -c flag exists.
    max_prompt = n_ctx - n_gen
    valid_prompts = [p for p in bench_params["prompts"] if p <= max_prompt]
    if not valid_prompts:
        valid_prompts = [min(bench_params["prompts"])]
    prompts_str = ",".join(str(p) for p in valid_prompts)
    nkvo        = str(bench_params.get("kv_offload", 0))
    cmd = [
        binary,
        "-m",    model["path"],
        "-ngl",  str(ngl),
        "-sm",   "layer",
        "-fa",   str(bench_params["flash_attn"]),
        "-nkvo", nkvo,
        "-ctk",  kv["ctk"],
        "-ctv",  kv["ctv"],
        "-p",    prompts_str,
        "-n",    str(n_gen),
        "-r",    str(bench_params["repetitions"]),
        "-o",    "jsonl",
        "--progress",
    ]
    # MTP models: benchmark with and without speculation, quantize the draft KV cache
    if model.get("is_mtp"):
        mtp_n_max  = str(model.get("mtp_n_max", "0,2"))
        draft_ctk  = model.get("mtp_draft_ctk", "q4_0")
        draft_ctv  = model.get("mtp_draft_ctv", "q4_0")
        cmd += [
            "--mtp-n-max", mtp_n_max,
            "-ctkd",       draft_ctk,
            "-ctvd",       draft_ctv,
        ]
    return cmd


# ---------------------------------------------------------------------------
# Context size selector — pick the largest ctx that fits in free VRAM
# ---------------------------------------------------------------------------

def resolve_ctx(
    model: dict,
    ngl: int,
    ctk: str,
    ctv: str,
    free_mib: int,
    safety_margin_mib: int = 400,
) -> int:
    """
    Return the largest power-of-two context size (from ctx_candidates) that
    fits in free_mib VRAM given the model at ngl and KV types ctk/ctv.
    Falls back to the smallest candidate if nothing larger fits.
    """
    candidates = model.get("ctx_candidates", [32768, 16384, 8192, 4096])
    kv_heads   = model.get("kv_head_count", 4)
    head_dim   = model.get("head_dim", 128)
    n_layers   = model.get("total_layers", 1)

    try:
        file_mib = os.path.getsize(model["path"]) // (1024 * 1024)
    except OSError:
        file_mib = 0

    total_layers  = n_layers
    ngl_fraction  = ngl / total_layers if total_layers else 1.0
    model_vram    = file_mib * ngl_fraction * 1.05
    usable        = free_mib - safety_margin_mib - model_vram

    for ctx in sorted(candidates, reverse=True):
        kv_mib = estimate_kv_cache_mib(ctx, n_layers, kv_heads, head_dim, ctk, ctv)
        if kv_mib <= usable:
            return ctx

    return candidates[-1]  # smallest available; VRAM pre-flight will skip if still too large


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_benchmarks(config: dict, selected_models: list, dry_run: bool):
    bench_params = config["bench_params"]
    backend_cfg  = config["backends"]["cuda"]
    models_cfg   = config["models"]
    log_cfg      = config["logging"]
    watchdog_cfg = config["watchdog"]

    log_dir       = Path(log_cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    vram_interval  = log_cfg.get("vram_sample_interval_s", 5)
    timeout_s      = watchdog_cfg.get("timeout_s", 1800)
    max_retries    = watchdog_cfg.get("max_retries", 2)
    safety_mib     = config.get("vram_safety_margin_mib", 400)

    binary   = backend_cfg["binary"]
    monitor  = VRAMMonitor()

    print(f"\n{'='*66}", flush=True)
    print(f"CUDA-ONLY BENCHMARK — RTX 4060 8GiB", flush=True)
    print(f"Binary: {binary}", flush=True)
    print(f"{'='*66}", flush=True)

    if not os.path.isfile(binary):
        print(f"  [FATAL] llama-bench binary not found: {binary}", flush=True)
        sys.exit(1)

    active_models = [
        m for m in models_cfg
        if selected_models == ["all"] or m["name"] in selected_models
    ]

    total_runs = len(active_models) * len(bench_params["kv_types"])
    run_num    = 0

    for model in active_models:
        model_name = model["name"]

        print(f"\n  MODEL: {model_name}", flush=True)

        if not os.path.isfile(model["path"]) and not dry_run:
            print(f"  [SKIP] model file not found: {model['path']}", flush=True)
            continue

        for kv in bench_params["kv_types"]:
            run_num += 1
            ctk = kv["ctk"]
            ctv = kv["ctv"]
            ts  = datetime.now().strftime("%Y%m%d-%H%M%S")

            # ── Live VRAM query — re-measure before every run ────────────
            # Any prior run may have left residual VRAM allocated. Always
            # query nvidia-smi fresh here so NGL and ctx are computed against
            # the actual current free VRAM, not a stale snapshot.
            live_snap = monitor.snapshot()
            free_mib_now = live_snap["free_mib"]
            print(f"\n  [{run_num}/{total_runs}] ctk={ctk} ctv={ctv}", flush=True)
            print(f"  [VRAM-LIVE] free={free_mib_now} MiB  "
                  f"used={live_snap['used_mib']} MiB  "
                  f"total={live_snap['total_mib']} MiB", flush=True)

            ngl = resolve_ngl(model, free_mib_now, safety_mib)
            if ngl == 0:
                print(f"  [SKIP] no NGL candidate fits in current free VRAM "
                      f"({free_mib_now} MiB)", flush=True)
                run_uuid = make_run_uuid()
                stem     = make_log_stem(run_uuid, model_name, ctk, ctv, ts)
                skip_path = log_path_for(log_dir, stem, "skipped")
                log_event(skip_path, "run_skipped",
                          model    = model_name,
                          ctk      = ctk,
                          ctv      = ctv,
                          run_uuid = run_uuid,
                          comment  = f"no NGL candidate fits: free_mib={free_mib_now}")
                continue

            print(f"  [NGL] resolved={ngl} / {model['total_layers']} layers  "
                  f"(free_mib={free_mib_now})", flush=True)

            run_uuid = make_run_uuid()
            stem     = make_log_stem(run_uuid, model_name, ctk, ctv, ts)

            # Pick the largest context that fits for this KV type
            n_ctx = resolve_ctx(model, ngl, ctk, ctv, free_mib_now, safety_mib)

            print(f"  [CTX] resolved={n_ctx}  ctk={ctk}/ctv={ctv}", flush=True)

            # VRAM pre-flight
            pf = vram_preflight(model, ngl, n_ctx, ctk, ctv, free_mib_now, safety_mib)
            status_str = "SAFE  " if pf["safe"] else "UNSAFE"
            print(f"  [PREFLIGHT {status_str}] {pf['comment']}", flush=True)

            if not pf["safe"]:
                skip_path = log_path_for(log_dir, stem, "skipped")
                print(f"  [SKIP] VRAM pre-flight blocked: {skip_path.name}", flush=True)
                log_event(skip_path, "run_skipped",
                          model      = model_name,
                          ngl        = ngl,
                          n_ctx      = n_ctx,
                          ctk        = ctk,
                          ctv        = ctv,
                          run_uuid   = run_uuid,
                          comment    = pf["comment"],
                          estimate   = pf["estimate"])
                continue

            # ── Run ──────────────────────────────────────────────────────
            inprogress_path = log_path_for(log_dir, stem, "inprogress")
            cmd = build_cmd(binary, model, ngl, n_ctx, kv, bench_params)

            log_event(inprogress_path, "run_start",
                      model    = model_name,
                      ngl      = ngl,
                      n_ctx    = n_ctx,
                      ctk      = ctk,
                      ctv      = ctv,
                      run_uuid = run_uuid,
                      cmd      = " ".join(cmd))

            log_event(inprogress_path, "preflight_passed",
                      n_ctx    = n_ctx,
                      ctk      = ctk,
                      ctv      = ctv,
                      estimate = pf["estimate"])

            pre_run_snap = monitor.snapshot()
            log_event(inprogress_path, "vram_pre_run",
                      vram_free_mib  = pre_run_snap["free_mib"],
                      vram_used_mib  = pre_run_snap["used_mib"],
                      vram_total_mib = pre_run_snap["total_mib"])

            telemetry = VRAMTelemetryThread(inprogress_path, monitor, vram_interval)
            telemetry.start()

            print(f"\n  --- llama-bench output ---", flush=True)
            run_start = time.time()
            success, bench_results = run_with_watchdog(
                cmd, inprogress_path, timeout_s, max_retries,
                telemetry=telemetry, dry_run=dry_run,
            )
            duration = time.time() - run_start
            print(f"  --- end llama-bench output ---", flush=True)

            telemetry.stop()
            peak_vram = telemetry.peak_used_mib

            post_snap = monitor.snapshot()
            log_event(inprogress_path, "vram_post_run",
                      vram_free_mib  = post_snap["free_mib"],
                      vram_used_mib  = post_snap["used_mib"],
                      vram_total_mib = post_snap["total_mib"])

            final_status = "completed" if success else "failed"
            log_event(inprogress_path, "run_end",
                      success       = success,
                      duration_s    = int(duration),
                      peak_vram_mib = peak_vram,
                      status        = final_status)

            print_result_summary(bench_results, peak_vram, inprogress_path)

            final_path = finalise_log(log_dir, stem, final_status, inprogress_path)

            status_tag = "OK" if success else "FAILED"
            print(f"\n  [{status_tag}] duration={int(duration)}s  "
                  f"log={final_path.name}", flush=True)

    print(f"\n{'='*66}", flush=True)
    print(f"All runs complete. Logs in: {log_dir}", flush=True)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_config(config: dict) -> list:
    errors = []
    for field in ("bench_params", "backends", "models", "logging", "watchdog"):
        if field not in config:
            errors.append(f"Missing top-level key: '{field}'")
    if "cuda" not in config.get("backends", {}):
        errors.append("Missing 'cuda' entry under 'backends'")
    for model in config.get("models", []):
        for field in ("name", "path", "total_layers"):
            if field not in model:
                errors.append(f"Model missing field '{field}': {model.get('name', '?')}")
    return errors


def show_last_cmd(log_dir: Path):
    logs = sorted(log_dir.glob("CUDA-ONLY-*.jsonl"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        # Fall back to any jsonl
        logs = sorted(log_dir.glob("*.jsonl"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        print(f"No log files found in {log_dir}", file=sys.stderr)
        sys.exit(1)

    latest       = logs[0]
    cmd_str      = None
    source_event = None

    for raw in latest.read_text(errors="replace").splitlines():
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if obj.get("event") == "run_start" and obj.get("cmd"):
            cmd_str      = obj["cmd"]
            source_event = "run_start"
            break
        if obj.get("event") == "attempt_start" and obj.get("cmd") and cmd_str is None:
            cmd_str      = obj["cmd"]
            source_event = "attempt_start"

    if not cmd_str:
        print(f"No command found in {latest.name}", file=sys.stderr)
        sys.exit(1)

    tokens = cmd_str.split()
    binary = tokens[0]
    flags  = tokens[1:]

    print(f"Log file : {latest.name}")
    print(f"Source   : event={source_event}")
    print(f"Tag      : CUDA-ONLY")
    print(f"Command  :")
    print(f"  {binary} \\")
    i = 0
    while i < len(flags):
        if (flags[i].startswith("-")
                and i + 1 < len(flags)
                and not flags[i + 1].startswith("-")):
            flag_line = f"  {flags[i]} {flags[i+1]}"
            i += 2
        else:
            flag_line = f"  {flags[i]}"
            i += 1
        suffix = " \\" if i < len(flags) else ""
        print(f"{flag_line}{suffix}")

    print()
    nkvo_val = None
    for idx, tok in enumerate(flags):
        if tok == "-nkvo" and idx + 1 < len(flags):
            nkvo_val = flags[idx + 1]
            break

    if nkvo_val == "0":
        print("  [VERIFIED] -nkvo 0  →  no_kv_offload=false  →  KV cache IS on VRAM (dGPU)")
    elif nkvo_val == "1":
        print("  [WARNING]  -nkvo 1  →  no_kv_offload=true   →  KV cache is on CPU RAM")
    else:
        print("  [WARNING]  -nkvo flag not found — KV offload state is implicit")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    script_dir     = Path(__file__).parent
    default_config = script_dir / "vai-cuda-bench-config.json"

    parser = argparse.ArgumentParser(
        description="VAI llama-bench CUDA-ONLY runner — RTX 4060 / code-completion / MTP models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default=str(default_config),
                        help=f"Path to config JSON (default: {default_config})")
    parser.add_argument("--model", default="all",
                        help="Model name(s) to bench, comma-separated, or 'all'")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--list-models", action="store_true",
                        help="List configured models and exit")
    parser.add_argument("--show-last-cmd", action="store_true",
                        help="Print llama-bench command from most recent CUDA-ONLY log "
                             "and verify -nkvo flag")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as fh:
        try:
            config = json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"ERROR: invalid JSON in config: {exc}", file=sys.stderr)
            sys.exit(1)

    errors = validate_config(config)
    if errors:
        for err in errors:
            print(f"CONFIG ERROR: {err}", file=sys.stderr)
        sys.exit(1)

    if args.list_models:
        print("Configured models:")
        for m in config["models"]:
            layers = m.get("total_layers", "?")
            mtp    = " [MTP]" if "mtp" in m["name"].lower() or "mtp" in m.get("_model_note","").lower() else ""
            print(f"  {m['name']}{mtp}  layers={layers}  ({m['path']})")
        sys.exit(0)

    if args.show_last_cmd:
        show_last_cmd(Path(config["logging"]["log_dir"]))
        sys.exit(0)

    selected_models = (
        ["all"] if args.model == "all"
        else [m.strip() for m in args.model.split(",")]
    )

    binary = config["backends"]["cuda"]["binary"]
    if not os.path.isfile(binary):
        print(f"WARNING: CUDA binary not found: {binary}", flush=True)

    if args.dry_run:
        print("DRY RUN — commands will be printed, not executed.", flush=True)

    print(f"Config:   {config_path}", flush=True)
    print(f"Backend:  CUDA-ONLY (nvidia-smi)", flush=True)
    print(f"Models:   {selected_models}", flush=True)
    print(f"Log dir:  {config['logging']['log_dir']}", flush=True)
    print(f"Log tag:  {LOG_TAG}", flush=True)

    log_dir = Path(config["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        run_benchmarks(config, selected_models, dry_run=args.dry_run)
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] run interrupted by user (KeyboardInterrupt)", flush=True)
        sys.exit(130)
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"\n[FATAL] Unhandled exception:\n{tb}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
