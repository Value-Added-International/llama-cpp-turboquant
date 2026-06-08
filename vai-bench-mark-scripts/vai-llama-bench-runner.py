#!/usr/bin/env python3
"""
VAI llama-bench runner — 2026-06-07
Crash-tolerant benchmark suite for Gemma 4 QAT models across CUDA/ROCm/Vulkan backends.
Measures token-generation throughput and VRAM at each context size and KV quantization.

Usage:
    ./vai-llama-bench-runner.py [--config path] [--backend cuda|rocm|vulkan|all]
                                [--model name|all] [--dry-run] [--show-last-cmd]

  --show-last-cmd   Read the most recent log file in log_dir and print the exact
                    llama-bench command that was stored in it, with each flag on its
                    own line. Use this to audit the precise flags passed to llama-bench
                    — particularly to verify -nkvo 0 (KV cache on VRAM) is present.

Log file naming
---------------
  {UUIDv7}-{BACKEND}-{model}-{ctk}-{ctv}-{ts}-{status}.jsonl

  Status suffix is written at the end of the run:
    -completed   run finished successfully
    -skipped     pre-flight safety check blocked the run
    -failed      all retries exhausted or timeout

Pre-flight safety checks
------------------------
  Before each run, parameters are validated against known kernel-safety rules
  (imported from rocm-safe-params.py). Currently active rules:

  ROCm/gfx1103: VEC and WMMA kernels only compute the KV_max OOB guard when
    Q->ne[1] (batch/prompt size) >= 1024. Below that threshold the kernel
    iterates over the full pre-allocated n_ctx, reading uninitialised UMA rows
    on the 780M → GPU hang. Any prompt < 1024 is skipped on ROCm.

  CUDA / Vulkan: no known kernel-safety issues at this time. Pre-flight always
    passes but the check is wired in so rules can be added per-backend later.

KV-cache quantization combinations — sourced from fattn.cu FATTN_VEC_CASES_ALL_D registrations
================================================================================================
ALWAYS AVAILABLE (no special build flag required):

  ctk (K type)  | ctv (V type)  | Notes
  --------------|---------------|--------------------------------------------------
  f16           | f16           | Baseline fp16
  q4_0          | q4_0          | Standard 4-bit
  q8_0          | q8_0          | Standard 8-bit
  bf16          | bf16          | Bfloat16
  turbo2        | turbo2        | TurboQuant2 symmetric
  turbo3        | turbo3        | TurboQuant3 symmetric
  turbo4        | turbo4        | TurboQuant4 symmetric
  turbo4        | q8_0          | Turbo4 K + q8_0 V
  q8_0          | turbo4        | q8_0 K + Turbo4 V
  q4_0          | turbo4        | q4_0 K + Turbo4 V  (V offloads more aggressively)
  q4_1          | turbo4        | q4_1 K + Turbo4 V
  f16           | turbo4        | f16 K + Turbo4 V
  turbo4        | f16           | Turbo4 K + f16 V
  turbo4        | turbo3        | Turbo4 K + Turbo3 V
  turbo3        | turbo4        | Turbo3 K + Turbo4 V
  turbo4        | turbo2        | Turbo4 K + Turbo2 V
  turbo2        | turbo4        | Turbo2 K + Turbo4 V
  turbo3        | q8_0          | Turbo3 K + q8_0 V
  q8_0          | turbo3        | q8_0 K + Turbo3 V
  q4_0          | turbo3        | q4_0 K + Turbo3 V
  q4_1          | turbo3        | q4_1 K + Turbo3 V
  f16           | turbo3        | f16 K + Turbo3 V
  turbo3        | f16           | Turbo3 K + f16 V
  turbo3        | turbo2        | Turbo3 K + Turbo2 V
  turbo2        | turbo3        | Turbo2 K + Turbo3 V
  turbo2        | q8_0          | Turbo2 K + q8_0 V
  q8_0          | turbo2        | q8_0 K + Turbo2 V
  q4_0          | turbo2        | q4_0 K + Turbo2 V
  q4_1          | turbo2        | q4_1 K + Turbo2 V
  f16           | turbo2        | f16 K + Turbo2 V
  turbo2        | f16           | Turbo2 K + f16 V

REQUIRES build flag -DGGML_CUDA_FA_ALL_QUANTS=ON (standard quant cross-combinations):
  q4_0/q4_1/q5_0/q5_1/q8_0/bf16 K x q4_0/q4_1/q5_0/q5_1/q8_0/bf16 V (all mixed pairs)
  Note: q5_0 and q5_1 types are ONLY available under FA_ALL_QUANTS.

NOT REGISTERED (will trigger GGML_ABORT fatal error — do not use):
  turbo4 K + q4_0 V,  turbo4 K + q4_1 V,  turbo4 K + q5_0 V,  turbo4 K + q5_1 V
  turbo3 K + q4_0 V,  turbo3 K + q4_1 V,  turbo3 K + q5_0 V,  turbo3 K + q5_1 V
  turbo2 K + q4_0 V,  turbo2 K + q4_1 V,  turbo2 K + q5_0 V,  turbo2 K + q5_1 V
  turbo2 K + bf16 V,  turbo3 K + bf16 V,  turbo4 K + bf16 V  (and their reverses)
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
# uuid_v7 — installed via: pipx install uuid-v7
# The package lives in a pipx venv; inject its site-packages so the system
# python can import it.
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
        # Fallback: UUIDv4 — monotonicity not guaranteed but still unique
        return str(_uuid.uuid4())
    print("WARNING: uuid_v7 not importable; falling back to UUIDv4. "
          "Run: pipx install uuid-v7", file=sys.stderr)

# ---------------------------------------------------------------------------
# Pre-flight safety checks — imported from rocm-safe-params.py
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent

def _load_safety_module():
    """Import classify() and KV_MAX_BATCH_THRESHOLD from rocm-safe-params.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rocm_safe_params", _SCRIPT_DIR / "rocm-safe-params.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

try:
    _safety = _load_safety_module()
    _classify       = _safety.classify
    _infer_head_dim = _safety.infer_head_dim
    _KV_MAX_THRESH  = _safety.KV_MAX_BATCH_THRESHOLD
except Exception as exc:
    print(f"WARNING: could not load rocm-safe-params.py: {exc}\n"
          "Pre-flight checks will be disabled.", file=sys.stderr)
    _classify       = None
    _infer_head_dim = None
    _KV_MAX_THRESH  = 1024


# Backend → head_dim lookup (used when model_type isn't known at pre-flight time).
# Maps model config "architecture" field → head_dim.
_ARCH_HEAD_DIM = {
    "gemma4-dense": 256,
    "gemma4-moe":   256,
    "gemma3":       256,
    "llama":        128,
    "mistral":      128,
    "qwen2":        128,
    "qwen3":        128,
    "deepseek":     128,
    "phi3":         96,
    "phi4":         96,
    "falcon":       64,
}


def preflight_check(
    backend: str,
    model: dict,
    n_prompt: int,
    n_ctx: int,
    type_k: str,
    type_v: str,
    flash_attn: int,
) -> dict:
    """
    Run the pre-flight safety check for a single (backend, prompt, kv-type) triple.

    Returns a dict:
      safe        : bool
      kernel      : str  — predicted kernel path
      rule        : str  — machine-readable rule identifier (empty string if safe)
      comment     : str  — human-readable explanation
      structured  : dict — machine-parseable fields for batch processing
    """
    # Derive head_dim from model architecture field (set in config JSON).
    arch     = model.get("architecture", "")
    head_dim = _ARCH_HEAD_DIM.get(arch)
    if head_dim is None and _infer_head_dim:
        head_dim = _infer_head_dim(arch) or _infer_head_dim(model.get("name", ""))
    if head_dim is None:
        head_dim = 128  # conservative default

    # CUDA / Vulkan: no known kernel-safety rules active yet.
    if backend in ("cuda", "vulkan"):
        return dict(
            safe=True,
            kernel="n/a",
            rule="",
            comment=f"No active pre-flight rules for backend={backend}",
            structured={
                "backend": backend,
                "n_prompt": n_prompt,
                "n_ctx": n_ctx,
                "type_k": type_k,
                "type_v": type_v,
                "flash_attn": flash_attn,
                "head_dim": head_dim,
                "rule": "",
            },
        )

    # ROCm: apply kernel-safety classification.
    if backend == "rocm" and _classify is not None:
        result = _classify(
            head_dim   = head_dim,
            n_ctx      = n_ctx,
            n_prompt   = n_prompt,
            type_k     = type_k,
            type_v     = type_v,
            flash_attn = bool(flash_attn),
        )
        safe   = result["safe"]
        kernel = result["kernel"]
        reason = result["reason"]
        rule   = "" if safe else "rocm_kv_max_oob_threshold"

        structured = {
            "backend":    backend,
            "n_prompt":   n_prompt,
            "n_ctx":      n_ctx,
            "type_k":     type_k,
            "type_v":     type_v,
            "flash_attn": flash_attn,
            "head_dim":   head_dim,
            "kernel":     kernel,
            "rule":       rule,
            "kv_max_threshold": _KV_MAX_THRESH,
        }

        return dict(safe=safe, kernel=kernel, rule=rule,
                    comment=reason, structured=structured)

    # Fallback if classify not available.
    return dict(
        safe=True, kernel="unknown", rule="",
        comment="rocm-safe-params.py not loaded; pre-flight skipped",
        structured={"backend": backend, "rule": "preflight_unavailable"},
    )


# ---------------------------------------------------------------------------
# VRAM monitoring
# ---------------------------------------------------------------------------

class VRAMMonitor:
    """Abstracts nvidia-smi (CUDA) and rocm-smi (ROCm/Vulkan) queries."""

    def __init__(self, monitor_tool: str):
        self.tool = monitor_tool  # "nvidia-smi" or "rocm-smi"

    def _query_nvidia(self) -> dict:
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

    def _query_rocm(self) -> dict:
        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).decode().strip()
            data = json.loads(out)
            card = next(iter(data.values()))
            total_b = int(card.get("VRAM Total Memory (B)", 0))
            used_b  = int(card.get("VRAM Total Used Memory (B)", 0))
            total_mib = total_b // (1024 * 1024)
            used_mib  = used_b  // (1024 * 1024)
            free_mib  = total_mib - used_mib
            return {"free_mib": free_mib, "used_mib": used_mib, "total_mib": total_mib}
        except Exception as exc:
            return {"free_mib": 0, "used_mib": 0, "total_mib": 0, "error": str(exc)}

    def snapshot(self) -> dict:
        if self.tool == "nvidia-smi":
            return self._query_nvidia()
        return self._query_rocm()

    def free_mib(self) -> int:
        return self.snapshot().get("free_mib", 0)


# ---------------------------------------------------------------------------
# NGL resolver — pick highest layer count that fits in available VRAM
# ---------------------------------------------------------------------------

def resolve_ngl(model: dict, backend: str, free_mib: int,
                safety_margin_mib: int = 512) -> int:
    candidates = model.get(f"ngl_{backend}_candidates", [])
    if not candidates:
        return 0

    model_path   = model["path"]
    total_layers = model.get("total_layers", 1)
    usable_mib   = free_mib - safety_margin_mib

    try:
        file_size_mib = os.path.getsize(model_path) // (1024 * 1024)
        vram_per_layer = (file_size_mib / total_layers) * 1.15
    except OSError:
        return candidates[0]

    for ngl in candidates:
        if ngl * vram_per_layer <= usable_mib:
            return ngl
    return 0


# ---------------------------------------------------------------------------
# Log filename helpers
# ---------------------------------------------------------------------------

def make_log_stem(run_uuid: str, backend: str, model_name: str,
                  ctk: str, ctv: str, ts: str) -> str:
    """
    Build the log filename stem (no suffix, no extension):
      {UUIDv7}-{BACKEND}-{model}-{ctk}-{ctv}-{ts}
    """
    return f"{run_uuid}-{backend.upper()}-{model_name}-{ctk}-{ctv}-{ts}"


def log_path_for(log_dir: Path, stem: str, status: str) -> Path:
    """Return full path: log_dir / {stem}-{status}.jsonl"""
    return log_dir / f"{stem}-{status}.jsonl"


def finalise_log(log_dir: Path, stem: str, status: str,
                 current_path: Path | None) -> Path:
    """
    Rename the in-progress log to its final status suffix.
    current_path is the path written during the run (status="inprogress").
    Returns the final path.
    """
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
        self.log_path     = log_path
        self.monitor      = monitor
        self.interval_s   = interval_s
        self._stop_event  = threading.Event()
        self.phase        = "running"
        self._phase_lock  = threading.Lock()
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
                "event": "vram_sample",
                "ts": time.time(),
                "phase": current_phase,
                "vram_used_mib": used,
                "vram_free_mib": snap.get("free_mib", 0),
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
    entry = {"event": event, "ts": time.time(), **kwargs}
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
        self.pipe         = pipe
        self.log_path     = log_path
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
    print(f"  PERFORMANCE SUMMARY  —  peak VRAM used: {peak_vram_mib} MiB",
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

    summary_event = {"event": "run_perf_summary", "ts": time.time(),
                     "peak_vram_mib": peak_vram_mib, "results": summary_entries}
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

        start      = time.monotonic()
        timed_out  = False
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
            log_event(log_path, "retry_wait", wait_s=10,
                      next_attempt=attempt + 1)
            time.sleep(10)

    return False, all_bench_results


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def build_cmd(binary: str, model: dict, ngl: int, split_mode: str,
              kv: dict, bench_params: dict) -> list:
    prompts_str = ",".join(str(p) for p in bench_params["prompts"])
    nkvo = str(bench_params.get("kv_offload", 0))
    return [
        binary,
        "-m",    model["path"],
        "-ngl",  str(ngl),
        "-sm",   split_mode,
        "-fa",   str(bench_params["flash_attn"]),
        "-nkvo", nkvo,
        "-ctk",  kv["ctk"],
        "-ctv",  kv["ctv"],
        "-p",    prompts_str,
        "-n",    str(bench_params["n_gen"]),
        "-r",    str(bench_params["repetitions"]),
        "-o",    "jsonl",
        "--progress",
    ]


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_benchmarks(config: dict, selected_backends: list, selected_models: list,
                   dry_run: bool):
    bench_params  = config["bench_params"]
    backends_cfg  = config["backends"]
    models_cfg    = config["models"]
    log_cfg       = config["logging"]
    watchdog_cfg  = config["watchdog"]

    log_dir       = Path(log_cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    vram_interval = log_cfg.get("vram_sample_interval_s", 5)
    timeout_s     = watchdog_cfg.get("timeout_s", 1800)
    max_retries   = watchdog_cfg.get("max_retries", 2)

    active_backends = {
        k: v for k, v in backends_cfg.items()
        if v.get("enabled", True)
        and (selected_backends == ["all"] or k in selected_backends)
    }
    active_models = [
        m for m in models_cfg
        if selected_models == ["all"] or m["name"] in selected_models
    ]

    total_runs = (len(active_backends) * len(active_models)
                  * len(bench_params["kv_types"]))
    run_num = 0

    for backend_name, backend_cfg in active_backends.items():
        binary     = backend_cfg["binary"]
        split_mode = backend_cfg.get("split_mode", "layer")
        monitor    = VRAMMonitor(backend_cfg["vram_monitor"])

        print(f"\n{'='*66}", flush=True)
        print(f"BACKEND: {backend_name.upper()}  binary: {binary}", flush=True)
        print(f"{'='*66}", flush=True)

        if not os.path.isfile(binary):
            print(f"  [SKIP] binary not found: {binary}", flush=True)
            continue

        for model in active_models:
            model_name = model["name"]

            print(f"\n  MODEL: {model_name}", flush=True)

            if not os.path.isfile(model["path"]) and not dry_run:
                print(f"  [SKIP] model file not found: {model['path']}",
                      flush=True)
                continue

            pre_snap = monitor.snapshot()
            print(f"  VRAM pre-load: free={pre_snap['free_mib']} MiB  "
                  f"used={pre_snap['used_mib']} MiB  "
                  f"total={pre_snap['total_mib']} MiB", flush=True)

            ngl = resolve_ngl(model, backend_name, pre_snap["free_mib"])
            if ngl == 0:
                print(f"  [SKIP] no NGL candidate fits in available VRAM "
                      f"({pre_snap['free_mib']} MiB free)", flush=True)
                continue

            print(f"  Resolved NGL: {ngl} / {model['total_layers']} layers",
                  flush=True)

            for kv in bench_params["kv_types"]:
                run_num += 1
                ctk = kv["ctk"]
                ctv = kv["ctv"]
                ts  = datetime.now().strftime("%Y%m%d-%H%M%S")

                run_uuid = make_run_uuid()
                stem     = make_log_stem(run_uuid, backend_name, model_name,
                                         ctk, ctv, ts)

                # ── Pre-flight check ─────────────────────────────────────────
                # Evaluate every prompt value in the list. If any is unsafe,
                # skip the entire run (the prompt list is passed as a single
                # llama-bench invocation; we cannot selectively drop values).
                n_ctx       = bench_params.get("n_batch", max(bench_params["prompts"]))
                flash_attn  = bench_params.get("flash_attn", 1)
                preflight_results = []
                any_unsafe  = False

                for n_prompt in bench_params["prompts"]:
                    pf = preflight_check(
                        backend    = backend_name,
                        model      = model,
                        n_prompt   = n_prompt,
                        n_ctx      = n_ctx,
                        type_k     = ctk,
                        type_v     = ctv,
                        flash_attn = flash_attn,
                    )
                    preflight_results.append({"n_prompt": n_prompt, **pf})
                    if not pf["safe"]:
                        any_unsafe = True

                print(f"\n  [{run_num}/{total_runs}] ctk={ctk} ctv={ctv}",
                      flush=True)

                # Print pre-flight summary
                print(f"  [PREFLIGHT] backend={backend_name.upper()}  "
                      f"ctk={ctk}  ctv={ctv}  n_ctx={n_ctx}",
                      flush=True)
                for pf in preflight_results:
                    status_str = "SAFE  " if pf["safe"] else "UNSAFE"
                    print(f"    [{status_str}] n_prompt={pf['n_prompt']:5d}  "
                          f"kernel={pf['kernel']:5s}  {pf['comment']}",
                          flush=True)

                if any_unsafe:
                    # Create log immediately with -skipped suffix
                    skip_path = log_path_for(log_dir, stem, "skipped")
                    print(f"  [SKIP] pre-flight blocked: {skip_path.name}",
                          flush=True)

                    # Build the skip event with both human comment and
                    # machine-readable structured fields
                    unsafe_entries = [pf for pf in preflight_results
                                      if not pf["safe"]]
                    skip_comment = (
                        f"Pre-flight safety check blocked this run. "
                        f"{len(unsafe_entries)} of {len(preflight_results)} "
                        f"prompt sizes failed: "
                        + "; ".join(
                            f"n_prompt={e['n_prompt']} ({e['comment']})"
                            for e in unsafe_entries
                        )
                    )

                    log_event(skip_path, "run_skipped",
                              model      = model_name,
                              backend    = backend_name,
                              ngl        = ngl,
                              ctk        = ctk,
                              ctv        = ctv,
                              run_uuid   = run_uuid,
                              # Human-readable plain text
                              comment    = skip_comment,
                              # Machine-readable structured fields
                              structured = {
                                  "backend":   backend_name,
                                  "model":     model_name,
                                  "ctk":       ctk,
                                  "ctv":       ctv,
                                  "n_ctx":     n_ctx,
                                  "prompts":   bench_params["prompts"],
                                  "flash_attn":flash_attn,
                                  "unsafe_prompts": [
                                      {"n_prompt":   e["n_prompt"],
                                       "rule":       e["rule"],
                                       "kernel":     e["kernel"],
                                       "comment":    e["comment"],
                                       "structured": e["structured"]}
                                      for e in unsafe_entries
                                  ],
                              })
                    continue

                # ── Run ──────────────────────────────────────────────────────
                inprogress_path = log_path_for(log_dir, stem, "inprogress")
                cmd = build_cmd(binary, model, ngl, split_mode, kv, bench_params)

                log_event(inprogress_path, "run_start",
                          model    = model_name,
                          backend  = backend_name,
                          ngl      = ngl,
                          ctk      = ctk,
                          ctv      = ctv,
                          run_uuid = run_uuid,
                          cmd      = " ".join(cmd))

                # Log pre-flight pass results into the run log
                log_event(inprogress_path, "preflight_passed",
                          prompts    = bench_params["prompts"],
                          backend    = backend_name,
                          structured = preflight_results)

                pre_run_snap = monitor.snapshot()
                log_event(inprogress_path, "vram_pre_run",
                          vram_free_mib  = pre_run_snap["free_mib"],
                          vram_used_mib  = pre_run_snap["used_mib"],
                          vram_total_mib = pre_run_snap["total_mib"])

                telemetry = VRAMTelemetryThread(inprogress_path, monitor,
                                                vram_interval)
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
                          success      = success,
                          duration_s   = int(duration),
                          peak_vram_mib= peak_vram,
                          status       = final_status)

                print_result_summary(bench_results, peak_vram, inprogress_path)

                # Rename to final status suffix
                final_path = finalise_log(log_dir, stem, final_status,
                                          inprogress_path)

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
    for model in config.get("models", []):
        for field in ("name", "path", "total_layers"):
            if field not in model:
                errors.append(
                    f"Model missing field '{field}': {model.get('name', '?')}")
    return errors


def show_last_cmd(log_dir: Path):
    logs = sorted(log_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime,
                  reverse=True)
    if not logs:
        print(f"No log files found in {log_dir}", file=sys.stderr)
        sys.exit(1)

    latest   = logs[0]
    cmd_str  = None
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
        print("  [VERIFIED] -nkvo 0  →  no_kv_offload=false  →  KV cache IS on VRAM")
    elif nkvo_val == "1":
        print("  [WARNING]  -nkvo 1  →  no_kv_offload=true   →  KV cache is on CPU RAM")
    else:
        print("  [WARNING]  -nkvo flag not found in command — KV offload state is implicit")


def check_binaries(config: dict, selected_backends: list) -> list:
    warnings = []
    for name, cfg in config["backends"].items():
        if selected_backends != ["all"] and name not in selected_backends:
            continue
        if not cfg.get("enabled", True):
            continue
        binary = cfg.get("binary", "")
        if not os.path.isfile(binary):
            warnings.append(f"Backend '{name}' binary not found: {binary}")
    return warnings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    script_dir      = Path(__file__).parent
    default_config  = script_dir / "vai-bench-config.json"

    parser = argparse.ArgumentParser(
        description="VAI llama-bench runner — crash-tolerant benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default=str(default_config),
                        help=f"Path to config JSON (default: {default_config})")
    parser.add_argument("--backend", default="all",
                        help="Backend(s) to run: cuda, rocm, vulkan, or all")
    parser.add_argument("--model", default="all",
                        help="Model name(s) to bench, or 'all'")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--list-models", action="store_true",
                        help="List configured models and exit")
    parser.add_argument("--show-last-cmd", action="store_true",
                        help="Print the llama-bench command from the most "
                             "recent log file and verify -nkvo flag")
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
            print(f"  {m['name']}  ({m['path']})")
        sys.exit(0)

    if args.show_last_cmd:
        show_last_cmd(Path(config["logging"]["log_dir"]))
        sys.exit(0)

    selected_backends = (
        ["all"] if args.backend == "all"
        else [b.strip() for b in args.backend.split(",")]
    )
    selected_models = (
        ["all"] if args.model == "all"
        else [m.strip() for m in args.model.split(",")]
    )

    warnings = check_binaries(config, selected_backends)
    for w in warnings:
        print(f"WARNING: {w}", flush=True)

    if args.dry_run:
        print("DRY RUN — commands will be printed, not executed.", flush=True)

    print(f"Config:   {config_path}", flush=True)
    print(f"Backends: {selected_backends}", flush=True)
    print(f"Models:   {selected_models}", flush=True)
    print(f"Log dir:  {config['logging']['log_dir']}", flush=True)

    log_dir = Path(config["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        run_benchmarks(config, selected_backends, selected_models,
                       dry_run=args.dry_run)
    except KeyboardInterrupt:
        msg = "run interrupted by user (KeyboardInterrupt)"
        print(f"\n[INTERRUPTED] {msg}", flush=True)
        sys.exit(130)
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"\n[FATAL] Unhandled exception:\n{tb}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
