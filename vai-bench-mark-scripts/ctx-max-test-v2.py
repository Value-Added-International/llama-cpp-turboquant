#!/usr/bin/env python3
"""
ctx-max-test.py
Find the maximum usable context window for Tesslate_OmniCoder-9B-Q4_K_M
on the RTX 4060 8 GiB (CUDA).

Four KV cache configs tested, descending quality / ascending compression:
  q8_0   / q8_0    — 8-bit K+V, quality baseline
  q4_0   / q4_0    — 4-bit K+V, standard baseline
  turbo4 / turbo4  — TurboQuant Walsh-Hadamard 4-bit K+V
  turbo4 / turbo3  — TurboQuant asymmetric: quality K, compressed V

Strategy per config:
  1. Query free VRAM (nvidia-smi) before model load.
  2. Estimate max_ctx = (free − MODEL_VRAM_MIB − COMPUTE_BUF_MIB − VRAM_SAFETY_MIB)
                        / bytes_per_token_for_this_KV_type
     Aligned to CTX_ALIGN, clamped to [512, MAX_CTX_ARCH].
  3. Fill (n_ctx − GEN_TOKENS − PROMPT_SLACK) tokens with synthetic code.
  4. Run llama-completion with -n 64; capture --perf timings.
  5. On OOM: step down by OOM_STEP tokens, retry up to MAX_RETRIES times.

Tune MODEL_VRAM_MIB if you keep hitting OOM on first attempt or if you
have consistent headroom — it is the only knob that moves estimated n_ctx.

Usage:
  python3 ctx-max-test.py [--dry-run] [--log-dir PATH] [--timeout SECS]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# UUIDv7 — fall back to UUIDv4 if not installed
# ---------------------------------------------------------------------------

_UUID_V7_SITE = Path.home() / ".local/share/pipx/venvs/uuid-v7/lib"


def _find_uuid_v7_site():
    if _UUID_V7_SITE.exists():
        for py_dir in sorted(_UUID_V7_SITE.iterdir(), reverse=True):
            sp = py_dir / "site-packages"
            if sp.exists():
                return sp
    return None


_sp = _find_uuid_v7_site()
if _sp and str(_sp) not in sys.path:
    sys.path.insert(0, str(_sp))

try:
    from uuid_v7.base import uuid7 as _uuid7
    def make_uuid(): return str(_uuid7())
except ImportError:
    import uuid as _uuid
    def make_uuid(): return str(_uuid.uuid4())
    print("WARNING: uuid_v7 not available, using UUIDv4. pip install uuid-v7",
          file=sys.stderr)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_TAG    = "CTX-MAX"
BINARY     = (
    "/home/none/git/llama-cpp-turboquant"
    "/build-x64-linux-cuda-nvidia-RTX4060-release/bin/llama-completion"
)
MODEL_PATH = (
    "/mnt/512_ssd_internal/vai-llm-tei/llm-models/bartowski"
    "/Tesslate_OmniCoder-9B-Q4_K_M.gguf"
)
MODEL_NAME = "Tesslate_OmniCoder-9B-Q4_K_M"

# Qwen GQA architecture — used only for KV cache byte-size estimation.
N_LAYERS  = 32
KV_HEADS  = 4       # GQA groups
HEAD_DIM  = 128

# VRAM budget.
# MODEL_VRAM_MIB: observed peak weight VRAM with -ngl 32 (all layers on GPU).
# Tune this down if calc_max_ctx is too conservative; up if you keep OOM-ing
# on the first attempt before any retry.
MODEL_VRAM_MIB  = 5500   # Q4_K_M 9B weights ~5.4 GiB; adjust empirically
COMPUTE_BUF_MIB = 350    # scratch / compute buffers (batch-size dependent)
VRAM_SAFETY_MIB = 512    # hard free-VRAM floor to leave after load

MAX_CTX_ARCH = 131072    # hard architecture cap
CTX_ALIGN    = 512       # align n_ctx to multiples of this

# Run parameters.
GEN_TOKENS   = 64   # tokens to generate after filling the context
PROMPT_SLACK = 128  # reserved for BOS + any llama-completion header tokens

# OOM retry.
OOM_STEP    = 8192  # step down on each OOM retry
MAX_RETRIES = 4     # max attempts per config before giving up

# KV scalar size in bytes (per element; applies identically to K and V).
KV_BYTES = {
    "f16":    2.0,
    "q8_0":   1.0,
    "q4_0":   0.5,
    "q4_1":   0.5,
    "turbo4": 0.5,
    "turbo3": 0.375,
    "turbo2": 0.25,
}

# Flat config list — the only 4 we care about, quality → compression order.
CONFIGS = [
    {"ctk": "q8_0",   "ctv": "q8_0",   "label": "8-bit K+V baseline"},
    {"ctk": "q4_0",   "ctv": "q4_0",   "label": "4-bit K+V baseline"},
    {"ctk": "turbo4", "ctv": "turbo4", "label": "TurboQuant 4-bit symmetric"},
    {"ctk": "turbo4", "ctv": "turbo3", "label": "TurboQuant asymmetric K4/V3"},
]

# ---------------------------------------------------------------------------
# Synthetic prompt builder
# ---------------------------------------------------------------------------

# Repeating Python code snippet — real-looking source avoids content refusals;
# ends mid-file so llama-completion naturally extends it.
_CODE_BLOCK = """\
def process(data, offset=0):
    result = []
    for i, item in enumerate(data):
        value = item * 2 + offset
        if value > 100:
            result.append(value)
    return result

x = process([1, 2, 3, 4, 5, 6, 7, 8], offset=10)
print(x)

"""
# Conservative undershoot so we never overshoot n_ctx; PROMPT_SLACK absorbs
# the remainder.  Measured ~2.75 chars/token for Qwen BPE on Python; 2.6 gives
# ~5 % margin.
_CHARS_PER_TOKEN = 2.6


def build_prompt_file(n_tokens: int) -> str:
    """Build a temp file of ~n_tokens tokens (char-count approximation)."""
    target_chars = int(n_tokens * _CHARS_PER_TOKEN)
    n_blocks = target_chars // len(_CODE_BLOCK) + 2
    content  = (_CODE_BLOCK * n_blocks)[:target_chars]
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix="ctx_max_prompt_", delete=False,
    )
    tmp.write(content)
    tmp.flush()
    tmp.close()
    return tmp.name

# ---------------------------------------------------------------------------
# VRAM helpers
# ---------------------------------------------------------------------------

def vram_snapshot() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=memory.free,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode().strip()
        free, used, total = [int(x.strip()) for x in out.split(",")]
        return {"free_mib": free, "used_mib": used, "total_mib": total}
    except Exception as exc:
        return {"free_mib": 0, "used_mib": 0, "total_mib": 0, "error": str(exc)}


def kv_mib_per_token(ctk: str, ctv: str) -> float:
    """Total MiB per context token across all K+V tensors for all layers."""
    bk = KV_BYTES.get(ctk, 2.0)
    bv = KV_BYTES.get(ctv, 2.0)
    return (bk + bv) * N_LAYERS * KV_HEADS * HEAD_DIM / (1024 * 1024)


def calc_max_ctx(ctk: str, ctv: str, free_mib: int) -> int:
    """
    Estimate the largest safe context given pre-load free VRAM.

    Reserves MODEL_VRAM_MIB (weights) + COMPUTE_BUF_MIB (scratch) +
    VRAM_SAFETY_MIB (headroom) from the reported free VRAM, then divides
    the remainder by the KV-cache bytes-per-token for this config.

    Result is aligned down to CTX_ALIGN and clamped to [512, MAX_CTX_ARCH].
    If the estimate still OOMs (e.g. MODEL_VRAM_MIB is set too low), the
    OOM retry loop in run_config() steps down in OOM_STEP increments.
    """
    usable = free_mib - MODEL_VRAM_MIB - COMPUTE_BUF_MIB - VRAM_SAFETY_MIB
    if usable <= 0:
        return 512
    mib_per_tok = kv_mib_per_token(ctk, ctv)
    raw_ctx = int(usable / mib_per_tok)
    ctx = (raw_ctx // CTX_ALIGN) * CTX_ALIGN
    return max(512, min(ctx, MAX_CTX_ARCH))

# ---------------------------------------------------------------------------
# Timing parser
# ---------------------------------------------------------------------------
# Handles both output formats (prefix varies by llama.cpp version):
#   llama_perf_context_print: prompt eval time = ...  (newer)
#   llama_print_timings:      prompt eval time = ...  (older)
# We match on the content substring only — no prefix anchor.
#
# Sample lines:
#   llama_perf_context_print: prompt eval time = 12345.67 ms / 43008 tokens (..., 3484.12 tokens per second)
#   llama_perf_context_print:        eval time =   234.56 ms /    64 runs   (...,   27.31 tokens per second)
#   llama_perf_context_print:        load time =  8123.45 ms
#   llama_perf_context_print:       total time = 12580.23 ms / 43072 tokens

_RE_LOAD  = re.compile(r"load time\s*=\s*([\d.]+)\s*ms")
_RE_TOTAL = re.compile(r"total time\s*=\s*([\d.]+)\s*ms")
_RE_PROMPT = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"
    r".*?([\d.]+)\s*tokens per second"
)
# Standalone "eval time" line — distinguish from "prompt eval time" by testing
# which key is present in the line before matching.
_RE_EVAL = re.compile(
    r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs"
    r".*?([\d.]+)\s*tokens per second"
)


def parse_timings(output: str) -> dict:
    result = {}
    for line in output.splitlines():
        if "prompt eval time" in line:
            m = _RE_PROMPT.search(line)
            if m and "prompt_ts" not in result:
                result["prompt_eval_ms"] = float(m.group(1))
                result["prompt_tokens"]  = int(m.group(2))
                result["prompt_ts"]      = float(m.group(3))
        elif "eval time" in line:
            # "eval time" without "prompt" prefix → the decode line
            m = _RE_EVAL.search(line)
            if m and "eval_ts" not in result:
                result["eval_ms"]   = float(m.group(1))
                result["eval_runs"] = int(m.group(2))
                result["eval_ts"]   = float(m.group(3))
        elif "load time" in line:
            m = _RE_LOAD.search(line)
            if m and "load_ms" not in result:
                result["load_ms"] = float(m.group(1))
        elif "total time" in line:
            m = _RE_TOTAL.search(line)
            if m and "total_ms" not in result:
                result["total_ms"] = float(m.group(1))
    return result

# ---------------------------------------------------------------------------
# VRAM telemetry thread
# ---------------------------------------------------------------------------

class VRAMTelemetry(threading.Thread):
    def __init__(self, log_path: Path, interval_s: int = 5):
        super().__init__(daemon=True)
        self.log_path      = log_path
        self.interval_s    = interval_s
        self._stop         = threading.Event()
        self.peak_used_mib = 0

    def stop(self):
        self._stop.set()
        self.join(timeout=self.interval_s + 2)

    def run(self):
        while not self._stop.is_set():
            snap = vram_snapshot()
            used = snap.get("used_mib", 0)
            if used > self.peak_used_mib:
                self.peak_used_mib = used
            _append_log(self.log_path, {
                "event":          "vram_sample",
                "backend":        "cuda",
                "log_tag":        LOG_TAG,
                "ts":             time.time(),
                "vram_used_mib":  used,
                "vram_free_mib":  snap.get("free_mib", 0),
                "vram_total_mib": snap.get("total_mib", 0),
            })
            self._stop.wait(timeout=self.interval_s)

# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------

def _append_log(path: Path, obj: dict):
    try:
        with open(path, "a") as fh:
            fh.write(json.dumps(obj) + "\n")
    except OSError:
        pass


def _log_event(path: Path, event: str, **kwargs):
    entry = {"event": event, "backend": "cuda", "log_tag": LOG_TAG,
             "ts": time.time(), **kwargs}
    _append_log(path, entry)
    # Print everything except vram_sample noise and raw cmd strings.
    if event != "vram_sample":
        details = "  ".join(
            f"{k}={v}" for k, v in kwargs.items() if k != "cmd"
        )
        print(f"  [{event.upper()}] {details}", flush=True)


def _log_path(log_dir: Path, run_uuid: str, ctk: str, ctv: str,
              ctx: int, ts: str, status: str) -> Path:
    stem = f"{LOG_TAG}-{run_uuid}-{MODEL_NAME}-{ctk}-{ctv}-ctx{ctx}-{ts}"
    return log_dir / f"{stem}-{status}.jsonl"


def _finalise_log(log_dir, run_uuid, ctk, ctv, ctx, ts, current, status):
    final = _log_path(log_dir, run_uuid, ctk, ctv, ctx, ts, status)
    if current and current.exists() and current != final:
        current.rename(final)
    return final

# ---------------------------------------------------------------------------
# Single attempt executor
# ---------------------------------------------------------------------------

def _run_one(ctk: str, ctv: str, n_ctx: int,
             log_dir: Path, run_uuid: str, ts: str,
             timeout_s: int) -> dict:
    """
    Execute one llama-completion call at the given n_ctx.
    Returns result dict; status in {"completed", "oom", "failed", "timeout"}.
    """
    n_prompt = n_ctx - GEN_TOKENS - PROMPT_SLACK
    inprog   = _log_path(log_dir, run_uuid, ctk, ctv, n_ctx, ts, "inprogress")

    snap_pre = vram_snapshot()
    _log_event(inprog, "run_start",
               model=MODEL_NAME, ctk=ctk, ctv=ctv,
               n_ctx=n_ctx, n_prompt=n_prompt, n_gen=GEN_TOKENS,
               run_uuid=run_uuid, vram_free_mib=snap_pre["free_mib"])

    prompt_file = build_prompt_file(n_prompt)

    cmd = [
        BINARY,
        "-m",    MODEL_PATH,
        "-ngl",  "32",          # all layers on GPU
        "-sm",   "layer",       # split mode: each layer assigned to one device
        "-fa",   "on",          # flash attention
        "-nkvo",                # KV cache stays on GPU (no offload to CPU)
        "-ctk",  ctk,
        "-ctv",  ctv,
        "-c",    str(n_ctx),
        "-f",    prompt_file,
        "-n",    str(GEN_TOKENS),
        "-b",    "2048",
        "-ub",   "512",
        "-no-cnv",              # raw completion — no chat-template header tokens
        "--no-warmup",          # skip empty warmup pass
        "--perf",               # emit llama_perf_context_print timing lines
    ]

    _log_event(inprog, "cmd", cmd=" ".join(cmd))
    print(f"\n  $ {' '.join(cmd)}\n", flush=True)

    telemetry = VRAMTelemetry(inprog, interval_s=5)
    telemetry.start()

    t_start = time.monotonic()
    output  = ""
    proc_ok = False
    status  = "failed"

    try:
        proc    = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr so --perf lines are captured
            timeout=timeout_s,
        )
        output  = proc.stdout.decode("utf-8", errors="replace")
        proc_ok = proc.returncode == 0
    except subprocess.TimeoutExpired:
        status = "timeout"
        _log_event(inprog, "timeout", timeout_s=timeout_s)
    except Exception as exc:
        _log_event(inprog, "process_error", error=str(exc))
    finally:
        try:
            os.unlink(prompt_file)
        except OSError:
            pass

    duration_s = int(time.monotonic() - t_start)
    telemetry.stop()
    peak_vram = telemetry.peak_used_mib

    # Stream output to log + console.
    for line in output.splitlines():
        _append_log(inprog, {
            "event": "output_line", "backend": "cuda", "log_tag": LOG_TAG,
            "ts": time.time(), "line": line,
        })
        print(f"  | {line}", flush=True)

    timings = parse_timings(output)

    oom = (
        "CUDA out of memory"   in output or
        "cudaMalloc failed"    in output or
        "failed to allocate"   in output.lower() or
        "out of memory"        in output.lower()
    )

    if oom:
        status = "oom"
    elif proc_ok and timings:
        status = "completed"
    elif status not in ("timeout",):
        status = "failed"

    _log_event(inprog, "run_end",
               status=status, proc_ok=proc_ok, oom=oom,
               duration_s=duration_s, peak_vram_mib=peak_vram)

    final = _finalise_log(log_dir, run_uuid, ctk, ctv, n_ctx, ts, inprog, status)

    return {
        "status":        status,
        "ctk":           ctk,
        "ctv":           ctv,
        "n_ctx":         n_ctx,
        "n_prompt":      n_prompt,
        "peak_vram_mib": peak_vram,
        "duration_s":    duration_s,
        "log":           str(final),
        **timings,
    }

# ---------------------------------------------------------------------------
# Config runner (OOM retry loop)
# ---------------------------------------------------------------------------

def run_config(cfg: dict, log_dir: Path, dry_run: bool, timeout_s: int) -> dict:
    """
    Run a single KV config.  On OOM, step down by OOM_STEP and retry
    up to MAX_RETRIES times.  Returns the final result dict.
    """
    ctk, ctv, label = cfg["ctk"], cfg["ctv"], cfg["label"]

    snap = vram_snapshot()
    free = snap["free_mib"]
    mib_per_tok = kv_mib_per_token(ctk, ctv)
    n_ctx = calc_max_ctx(ctk, ctv, free)

    print(f"\n  VRAM:  free={free} MiB  used={snap['used_mib']} MiB"
          f"  total={snap['total_mib']} MiB", flush=True)
    print(f"  KV:    {mib_per_tok*1024:.2f} KiB/token  "
          f"estimated max_ctx={n_ctx}  "
          f"KV budget ~{n_ctx * mib_per_tok:.0f} MiB", flush=True)

    run_uuid = make_uuid()
    ts       = datetime.now().strftime("%Y%m%d-%H%M%S")

    # ---- dry-run path -------------------------------------------------------
    if dry_run:
        n_prompt = n_ctx - GEN_TOKENS - PROMPT_SLACK
        cmd = [
            BINARY, "-m", MODEL_PATH, "-ngl", "32", "-sm", "layer",
            "-fa", "on", "-nkvo", "-ctk", ctk, "-ctv", ctv,
            "-c", str(n_ctx), "-f", "<prompt_file>",
            "-n", str(GEN_TOKENS), "-b", "2048", "-ub", "512",
            "-no-cnv", "--no-warmup", "--perf",
        ]
        print(f"\n  [DRY-RUN] $ {' '.join(cmd)}", flush=True)
        inprog = _log_path(log_dir, run_uuid, ctk, ctv, n_ctx, ts, "inprogress")
        _log_event(inprog, "run_start",
                   model=MODEL_NAME, ctk=ctk, ctv=ctv, n_ctx=n_ctx,
                   n_prompt=n_prompt, n_gen=GEN_TOKENS, run_uuid=run_uuid,
                   vram_free_mib=free)
        _log_event(inprog, "cmd", cmd=" ".join(cmd))
        _log_event(inprog, "dry_run")
        final = _finalise_log(log_dir, run_uuid, ctk, ctv, n_ctx, ts, inprog, "dry-run")
        return {"status": "dry-run", "ctk": ctk, "ctv": ctv,
                "n_ctx": n_ctx, "log": str(final)}

    # ---- real run with OOM retry --------------------------------------------
    result = {"status": "oom", "ctk": ctk, "ctv": ctv, "n_ctx": 0}

    for attempt in range(MAX_RETRIES + 1):
        if attempt > 0:
            n_ctx -= OOM_STEP
            n_ctx  = (n_ctx // CTX_ALIGN) * CTX_ALIGN
            if n_ctx < 512:
                print(f"\n  [GIVE-UP] ctx stepped below 512 after"
                      f" {attempt} OOM retries — skipping", flush=True)
                return result
            print(f"\n  [RETRY #{attempt}/{MAX_RETRIES}] "
                  f"stepping down to n_ctx={n_ctx}", flush=True)

        result = _run_one(ctk, ctv, n_ctx, log_dir, run_uuid, ts, timeout_s)

        if result["status"] != "oom":
            return result  # success, failure, or timeout — stop retrying

    return result  # final OOM result after exhausting retries

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: list):
    sep = "=" * 92
    print(f"\n{sep}", flush=True)
    print(f"  CTX-MAX RESULTS  —  {MODEL_NAME}  —  RTX 4060 8 GiB CUDA",
          flush=True)
    print(sep, flush=True)
    hdr = (f"  {'ctk':8s} {'ctv':8s} {'n_ctx':>8s} "
           f"{'prefill t/s':>12s} {'decode t/s':>11s} "
           f"{'peak VRAM':>10s}  {'status'}")
    print(hdr, flush=True)
    print("  " + "-" * 88, flush=True)
    for r in results:
        ctk    = r.get("ctk", "?")
        ctv    = r.get("ctv", "?")
        ctx    = r.get("n_ctx", 0)
        pts    = r.get("prompt_ts", 0.0)
        ets    = r.get("eval_ts",   0.0)
        peak   = r.get("peak_vram_mib", 0)
        status = r.get("status", "?")
        p_str  = f"{pts:>12.1f}" if pts  else f"{'—':>12s}"
        e_str  = f"{ets:>11.2f}" if ets  else f"{'—':>11s}"
        pk_str = f"{peak:>9d} M" if peak else f"{'—':>10s}"
        print(f"  {ctk:8s} {ctv:8s} {ctx:>8d} {p_str} {e_str} {pk_str}  {status}",
              flush=True)
    print(sep, flush=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="ctx-max-test — maximum context window finder (CUDA/llama-completion)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Log and print commands without executing them")
    ap.add_argument("--log-dir", default=None,
                    help="Output log directory (default: ./ctx-max-logs/)")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="Per-run timeout in seconds (default: 3600)")
    args = ap.parse_args()

    log_dir = (
        Path(args.log_dir) if args.log_dir
        else Path(__file__).parent / "ctx-max-logs"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.isfile(BINARY):
        print(f"FATAL: binary not found: {BINARY}", file=sys.stderr)
        sys.exit(1)
    if not args.dry_run and not os.path.isfile(MODEL_PATH):
        print(f"FATAL: model not found: {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  CTX-MAX TEST  —  {MODEL_NAME}")
    print(f"  binary:          {BINARY}")
    print(f"  model:           {MODEL_PATH}")
    print(f"  log dir:         {log_dir}")
    print(f"  MODEL_VRAM_MIB:  {MODEL_VRAM_MIB}  ← adjust if estimates are wrong")
    print(f"  COMPUTE_BUF_MIB: {COMPUTE_BUF_MIB}")
    print(f"  VRAM_SAFETY_MIB: {VRAM_SAFETY_MIB}")
    print(f"  gen tokens:      {GEN_TOKENS}")
    print(f"  OOM retry:       up to {MAX_RETRIES}× (−{OOM_STEP} tokens each)")
    print(f"  dry run:         {args.dry_run}")
    print(f"{'='*70}\n")

    results = []

    for i, cfg in enumerate(CONFIGS):
        ctk, ctv, label = cfg["ctk"], cfg["ctv"], cfg["label"]
        print(f"  ── [{i+1}/{len(CONFIGS)}]  {ctk}/{ctv}  —  {label}", flush=True)

        r = run_config(cfg, log_dir, args.dry_run, args.timeout)
        results.append(r)

        if not args.dry_run:
            time.sleep(5)   # let VRAM drain before the next run

    print_summary(results)
    print(f"\n  Logs: {log_dir}\n", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]", flush=True)
        sys.exit(130)
