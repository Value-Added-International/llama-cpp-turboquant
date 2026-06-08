#!/usr/bin/env python3
"""
ctx-max-test.py — Maximum context window finder for RTX 4060 8GiB (CUDA)
Uses llama-completion (not llama-bench) with a synthetic prompt sized to fill the target context.

Phases (ordered by KV size descending → ascending compression):
  Phase 1 — Baselines:
    q8_0 / q8_0   — quality reference, expected ~16-29K ctx with 1GiB safety
    q4_0 / q4_0   — standard 4-bit, ~59K theoretical max
  Phase 2 — TurboQuant compressed:
    turbo4 / turbo4  — ~59K theoretical max
    turbo3 / turbo3  — ~79K theoretical max
  Phase 3 — Maximum stretch (aggressive V compression):
    turbo4 / turbo3  — asymmetric: larger K quality, aggressive V
    turbo3 / turbo2  — near-maximum compression with usable K quality
    turbo2 / turbo2  — absolute maximum ctx; lowest KV quality floor

For each phase the script:
  1. Queries nvidia-smi for current free VRAM
  2. Calculates the theoretical max ctx at that VRAM budget (1 GiB safety)
  3. Clamps to model arch max (131072) and aligns to 512-token boundary
  4. Builds a synthetic code-snippet prompt (ctx - 64 - 128 slack) tokens long
  5. Runs llama-completion with -n 64 to generate 64 tokens into that filled context
  6. Captures --perf output: prefill t/s, generation t/s, VRAM peak
  7. Logs everything to JSONL (same format as vai-cuda-bench-runner.py)

The 64-token generation tail is intentional: it forces a real decode step after
filling the context, giving a combined prefill+decode measurement that reflects
real long-context inference cost.

Usage:
  python3 ctx-max-test.py [--dry-run] [--phases 1,2,3] [--log-dir PATH]

Log files:
  CTX-MAX-{UUIDv7}-{model}-{ctk}-{ctv}-{ctx}-{ts}-{status}.jsonl
  Written to lamma-bench-logs/ctx-max/ (sibling dir to cuda-only/)
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
# UUIDv7 — same bootstrap as vai-cuda-bench-runner.py
# ---------------------------------------------------------------------------

_UUID_V7_SITE = Path.home() / ".local/share/pipx/venvs/uuid-v7/lib"


def _find_uuid_v7_site() -> Path | None:
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
    def make_uuid() -> str:
        return str(_uuid7())
except ImportError:
    import uuid as _uuid
    def make_uuid() -> str:  # type: ignore[misc]
        return str(_uuid.uuid4())
    print("WARNING: uuid_v7 not available; using UUIDv4. pip install uuid-v7",
          file=sys.stderr)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_TAG       = "CTX-MAX"
BINARY        = ("/home/none/git/llama-cpp-turboquant"
                 "/build-x64-linux-cuda-nvidia-RTX4060-release/bin/llama-completion")
MODEL_PATH    = ("/mnt/512_ssd_internal/vai-llm-tei/llm-models/bartowski"
                 "/Tesslate_OmniCoder-9B-Q4_K_M.gguf")
MODEL_NAME    = "Tesslate_OmniCoder-9B-Q4_K_M"

# Qwen3.5 9B architecture — for KV cache size estimation
N_LAYERS      = 32
KV_HEADS      = 4
HEAD_DIM      = 128

# VRAM budget constants
VRAM_TOTAL_MIB    = 8053    # RTX 4060 Laptop usable VRAM
SAFETY_MIB        = 1024    # 1 GiB hard floor — never go below this free headroom
MODEL_VRAM_MIB    = 6100    # observed peak model weight VRAM (all 32 layers)
MAX_CTX_ARCH      = 131072  # conservative arch cap; model trains to 262144 but VRAM limits earlier
CTX_ALIGN         = 512     # align context to 512-token boundaries
GEN_TOKENS        = 64      # tokens to generate after filling the context
# Reserve headroom beyond GEN_TOKENS for BOS token + any chat-template header that
# llama-completion injects before the raw prompt text (~10 tokens observed).
# Without this the context fills exactly to n_ctx and generation stops early.
PROMPT_SLACK      = 128

# KV element size in bytes — must match ggml quant definitions
KV_BYTES = {
    "f16":    2.0,
    "q8_0":   1.0,
    "q4_0":   0.5,
    "q4_1":   0.5,
    "turbo4": 0.5,
    "turbo3": 0.375,
    "turbo2": 0.25,
}

# ---------------------------------------------------------------------------
# Test phases — ordered from largest KV type to smallest (most → least VRAM)
# ---------------------------------------------------------------------------

PHASES = {
    1: [
        {"ctk": "q8_0",   "ctv": "q8_0",
         "label": "Baseline q8_0/q8_0 — quality reference"},
        {"ctk": "q4_0",   "ctv": "q4_0",
         "label": "Baseline q4_0/q4_0 — standard 4-bit"},
    ],
    2: [
        {"ctk": "turbo4", "ctv": "turbo4",
         "label": "TurboQuant turbo4/turbo4 — primary KV compression target"},
        {"ctk": "turbo3", "ctv": "turbo3",
         "label": "TurboQuant turbo3/turbo3 — 3-bit symmetric"},
    ],
    3: [
        {"ctk": "turbo4", "ctv": "turbo3",
         "label": "Asymmetric turbo4/turbo3 — larger K quality, aggressive V"},
        {"ctk": "turbo3", "ctv": "turbo2",
         "label": "Asymmetric turbo3/turbo2 — near-max compression"},
        {"ctk": "turbo2", "ctv": "turbo2",
         "label": "Maximum turbo2/turbo2 — absolute ctx ceiling"},
    ],
}

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
    """Bytes per token across the full KV cache (both K and V, all layers)."""
    bk = KV_BYTES.get(ctk, 2.0)
    bv = KV_BYTES.get(ctv, 2.0)
    return (bk + bv) * N_LAYERS * KV_HEADS * HEAD_DIM / (1024 * 1024)


def calc_max_ctx(ctk: str, ctv: str, free_mib: int) -> int:
    """
    Return the largest ctx that leaves SAFETY_MIB free after model + KV cache.
    Uses live free_mib (from nvidia-smi) rather than a fixed MODEL_VRAM_MIB budget,
    because the model may or may not already be resident in VRAM between runs.
    Aligns down to CTX_ALIGN and clamps to [512, MAX_CTX_ARCH].
    """
    usable = free_mib - SAFETY_MIB
    if usable <= 0:
        return 512
    mib_per_tok = kv_mib_per_token(ctk, ctv)
    raw_ctx = int(usable / mib_per_tok)
    # Align down
    ctx = (raw_ctx // CTX_ALIGN) * CTX_ALIGN
    ctx = max(512, min(ctx, MAX_CTX_ARCH))
    return ctx


# ---------------------------------------------------------------------------
# Synthetic prompt builder
# ---------------------------------------------------------------------------

# Repeating source-code snippet used as filler.
# - Looks like real code → no content refusal from the model
# - Ends mid-function so llama-completion just extends it naturally
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
# Measured chars/token for _CODE_BLOCK on Qwen3.5 BPE: 250 chars / ~91 tokens = 2.75.
# Python keywords are well-known BPE subwords so they compress more than plain prose.
# Use 2.6 (slightly under) so we never overshoot — the PROMPT_SLACK absorbs the rest.
_CHARS_PER_TOKEN = 2.6


def build_prompt_file(n_tokens: int) -> str:
    """
    Write a temporary prompt file of approximately n_tokens tokens.
    Returns the path; caller must delete.

    Repeats a Python code snippet so the content looks like source code,
    preventing the model from refusing to continue. Sizes by character count
    using the measured BPE ratio (2.75 chars/token) with a small undershoot
    margin so we never exceed n_tokens.
    """
    target_chars = int(n_tokens * _CHARS_PER_TOKEN)
    # Build by repeating full blocks then trim to target_chars
    n_blocks = target_chars // len(_CODE_BLOCK) + 2
    content  = (_CODE_BLOCK * n_blocks)[:target_chars]

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix="ctx_max_prompt_", delete=False
    )
    tmp.write(content)
    tmp.flush()
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Timing parser — reads llama-cli stderr/stdout for llama_perf_context_print lines
# ---------------------------------------------------------------------------
# Example lines from --perf (llama-completion / common_perf_print):
#   0.10.136.260 I common_perf_print: prompt eval time =   167.55 ms /    18 tokens ( 9.31 ms per token,  107.43 tokens per second)
#   0.10.136.262 I common_perf_print:        eval time =   806.09 ms /    15 runs   (53.74 ms per token,   18.61 tokens per second)
#   0.10.136.258 I common_perf_print:        load time =  8481.94 ms
#   0.10.136.262 I common_perf_print:       total time =  1031.68 ms /    33 tokens

_RE_PROMPT_EVAL = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second"
)
_RE_EVAL = re.compile(
    r"^\S+\s+I\s+common_perf_print:\s+eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs.*?([\d.]+)\s*tokens per second",
    re.MULTILINE,
)
_RE_LOAD = re.compile(r"load time\s*=\s*([\d.]+)\s*ms")
_RE_TOTAL = re.compile(r"total time\s*=\s*([\d.]+)\s*ms")


def parse_timings(output: str) -> dict:
    result = {}
    for line in output.splitlines():
        m = _RE_PROMPT_EVAL.search(line)
        if m:
            result["prompt_eval_ms"] = float(m.group(1))
            result["prompt_tokens"]  = int(m.group(2))
            result["prompt_ts"]      = float(m.group(3))
            continue
        # Only match standalone eval time line (not the prompt eval line above)
        m = _RE_EVAL.search(line)
        if m:
            result["eval_ms"]   = float(m.group(1))
            result["eval_runs"] = int(m.group(2))
            result["eval_ts"]   = float(m.group(3))
            continue
        m = _RE_LOAD.search(line)
        if m:
            result["load_ms"] = float(m.group(1))
            continue
        m = _RE_TOTAL.search(line)
        if m:
            result["total_ms"] = float(m.group(1))
    return result


# ---------------------------------------------------------------------------
# VRAM telemetry thread — mirrors vai-cuda-bench-runner.py
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
            entry = {
                "event":          "vram_sample",
                "backend":        "cuda",
                "log_tag":        LOG_TAG,
                "ts":             time.time(),
                "vram_used_mib":  used,
                "vram_free_mib":  snap.get("free_mib", 0),
                "vram_total_mib": snap.get("total_mib", 0),
            }
            _append_log(self.log_path, entry)
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


def log_event(path: Path, event: str, **kwargs):
    entry = {"event": event, "backend": "cuda", "log_tag": LOG_TAG,
             "ts": time.time(), **kwargs}
    _append_log(path, entry)
    details = "  ".join(f"{k}={v}" for k, v in kwargs.items()
                        if k not in ("cmd", "output"))
    print(f"  [{event.upper()}] {details}", flush=True)


def make_log_path(log_dir: Path, run_uuid: str, ctk: str, ctv: str,
                  ctx: int, ts: str, status: str) -> Path:
    stem = f"{LOG_TAG}-{run_uuid}-{MODEL_NAME}-{ctk}-{ctv}-ctx{ctx}-{ts}"
    return log_dir / f"{stem}-{status}.jsonl"


def finalise_log(log_dir: Path, run_uuid: str, ctk: str, ctv: str,
                 ctx: int, ts: str, current: Path, status: str) -> Path:
    final = make_log_path(log_dir, run_uuid, ctk, ctv, ctx, ts, status)
    if current and current.exists() and current != final:
        current.rename(final)
    return final


# ---------------------------------------------------------------------------
# Single test runner
# ---------------------------------------------------------------------------

def run_ctx_test(
    ctk: str,
    ctv: str,
    n_ctx: int,
    log_dir: Path,
    dry_run: bool = False,
    timeout_s: int = 3600,
) -> dict:
    """
    Run one llama-completion test: fill (n_ctx - GEN_TOKENS - PROMPT_SLACK) tokens
    then generate GEN_TOKENS. PROMPT_SLACK reserves space for the BOS token and any
    chat-template header injected by llama-completion before the raw prompt text.
    Returns a result dict with timing fields and status.
    """
    run_uuid  = make_uuid()
    ts        = datetime.now().strftime("%Y%m%d-%H%M%S")
    n_prompt  = n_ctx - GEN_TOKENS - PROMPT_SLACK

    inprog = make_log_path(log_dir, run_uuid, ctk, ctv, n_ctx, ts, "inprogress")

    snap_pre = vram_snapshot()
    log_event(inprog, "run_start",
              model=MODEL_NAME, ctk=ctk, ctv=ctv, n_ctx=n_ctx,
              n_prompt=n_prompt, n_gen=GEN_TOKENS, run_uuid=run_uuid,
              vram_free_mib=snap_pre["free_mib"])

    # Build synthetic prompt file
    prompt_file = build_prompt_file(n_prompt)

    cmd = [
        BINARY,
        "-m",    MODEL_PATH,
        "-ngl",  "32",
        "-sm",   "layer",
        "-fa",   "on",
        "-nkvo",              # KV cache stays on VRAM
        "-ctk",  ctk,
        "-ctv",  ctv,
        "-c",    str(n_ctx),
        "-f",    prompt_file,
        "-n",    str(GEN_TOKENS),
        "-b",    "2048",
        "-ub",   "512",
        "-no-cnv",            # disable chat template — raw completion, no system/user header tokens
        "--no-warmup",        # skip empty warmup run
        "--perf",             # emit common_perf_print timing lines on exit
    ]

    log_event(inprog, "cmd", cmd=" ".join(cmd))
    print(f"\n  cmd: {' '.join(cmd)}", flush=True)

    if dry_run:
        os.unlink(prompt_file)
        log_event(inprog, "dry_run")
        final = finalise_log(log_dir, run_uuid, ctk, ctv, n_ctx, ts, inprog, "dry-run")
        return {"status": "dry-run", "ctk": ctk, "ctv": ctv, "n_ctx": n_ctx,
                "log": final.name}

    telemetry = VRAMTelemetry(inprog, interval_s=5)
    telemetry.start()

    t_start = time.monotonic()
    output  = ""
    success = False

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
        output  = result.stdout.decode("utf-8", errors="replace")
        success = result.returncode == 0
    except subprocess.TimeoutExpired:
        log_event(inprog, "timeout", timeout_s=timeout_s)
    except Exception as exc:
        log_event(inprog, "error", error=str(exc))
    finally:
        try:
            os.unlink(prompt_file)
        except OSError:
            pass

    duration_s = int(time.monotonic() - t_start)
    telemetry.stop()
    peak_vram = telemetry.peak_used_mib

    # Echo raw output to log and console
    for line in output.splitlines():
        _append_log(inprog, {"event": "output_line", "backend": "cuda",
                              "log_tag": LOG_TAG, "ts": time.time(), "line": line})
        print(f"    | {line}", flush=True)

    timings = parse_timings(output)

    if not success and not timings:
        # Check for OOM signature
        oom = "CUDA out of memory" in output or "cudaMalloc failed" in output
        log_event(inprog, "run_end",
                  success=False, duration_s=duration_s,
                  peak_vram_mib=peak_vram, oom=oom,
                  status="failed")
        final = finalise_log(log_dir, run_uuid, ctk, ctv, n_ctx, ts, inprog, "failed")
        return {"status": "oom" if oom else "failed",
                "ctk": ctk, "ctv": ctv, "n_ctx": n_ctx,
                "peak_vram_mib": peak_vram, "log": final.name}

    result_entry = {
        "event":           "ctx_max_result",
        "backend":         "cuda",
        "log_tag":         LOG_TAG,
        "ts":              time.time(),
        "model":           MODEL_NAME,
        "ctk":             ctk,
        "ctv":             ctv,
        "n_ctx":           n_ctx,
        "n_prompt":        n_prompt,
        "n_gen":           GEN_TOKENS,
        "peak_vram_mib":   peak_vram,
        "duration_s":      duration_s,
        **timings,
    }
    _append_log(inprog, result_entry)

    log_event(inprog, "run_end",
              success=True, duration_s=duration_s,
              peak_vram_mib=peak_vram, status="completed")

    final = finalise_log(log_dir, run_uuid, ctk, ctv, n_ctx, ts, inprog, "completed")

    return {
        "status":         "completed",
        "ctk":            ctk,
        "ctv":            ctv,
        "n_ctx":          n_ctx,
        "n_prompt":       n_prompt,
        "peak_vram_mib":  peak_vram,
        "duration_s":     duration_s,
        "log":            final.name,
        **timings,
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]):
    sep = "=" * 80
    print(f"\n{sep}", flush=True)
    print(f"  CTX-MAX RESULTS  —  {MODEL_NAME}  —  RTX 4060 8GiB CUDA", flush=True)
    print(sep, flush=True)
    hdr = f"  {'ctk':8s} {'ctv':8s} {'ctx':>8s} {'prompt_ts':>12s} {'eval_ts':>10s} {'peak_vram':>10s} {'status'}"
    print(hdr, flush=True)
    print("  " + "-" * 76, flush=True)
    for r in results:
        ctk         = r.get("ctk", "?")
        ctv         = r.get("ctv", "?")
        ctx         = r.get("n_ctx", 0)
        prompt_ts   = r.get("prompt_ts", 0.0)
        eval_ts     = r.get("eval_ts",   0.0)
        peak        = r.get("peak_vram_mib", 0)
        status      = r.get("status", "?")
        p_str = f"{prompt_ts:>12.1f}" if prompt_ts else f"{'—':>12s}"
        e_str = f"{eval_ts:>10.2f}"   if eval_ts   else f"{'—':>10s}"
        print(f"  {ctk:8s} {ctv:8s} {ctx:>8d} {p_str} {e_str} {peak:>9d}M  {status}",
              flush=True)
    print(sep, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ctx-max-test — iterative maximum context finder (CUDA/llama-cli)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--phases", default="1,2,3",
                        help="Comma-separated phase numbers to run (default: 1,2,3)")
    parser.add_argument("--log-dir", default=None,
                        help="Log directory (default: lamma-bench-logs/ctx-max/ next to this script)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Per-run timeout in seconds (default: 3600 = 1 hour)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    log_dir    = Path(args.log_dir) if args.log_dir else (
        script_dir.parent / "lamma-bench-logs" / "ctx-max"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    phases_to_run = [int(p.strip()) for p in args.phases.split(",")]

    if not os.path.isfile(BINARY):
        print(f"FATAL: llama-cli not found: {BINARY}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(MODEL_PATH) and not args.dry_run:
        print(f"FATAL: model not found: {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*70}", flush=True)
    print(f"  CTX-MAX TEST  —  {MODEL_NAME}", flush=True)
    print(f"  binary:    {BINARY}", flush=True)
    print(f"  model:     {MODEL_PATH}", flush=True)
    print(f"  log dir:   {log_dir}", flush=True)
    print(f"  phases:    {phases_to_run}", flush=True)
    print(f"  safety:    {SAFETY_MIB} MiB (1 GiB)", flush=True)
    print(f"  gen tokens: {GEN_TOKENS} per run", flush=True)
    print(f"  dry run:   {args.dry_run}", flush=True)
    print(f"{'='*70}", flush=True)

    all_results = []

    for phase in phases_to_run:
        if phase not in PHASES:
            print(f"  [WARN] unknown phase {phase}, skipping", flush=True)
            continue

        configs = PHASES[phase]
        print(f"\n  ── Phase {phase} ({'baseline' if phase==1 else 'turbo' if phase==2 else 'max-stretch'}) ──",
              flush=True)

        for cfg in configs:
            ctk   = cfg["ctk"]
            ctv   = cfg["ctv"]
            label = cfg["label"]

            print(f"\n  [{ctk}/{ctv}]  {label}", flush=True)

            # Fresh VRAM query before every run — the previous run may have
            # evicted the model from VRAM, changing the free headroom.
            snap = vram_snapshot()
            free = snap["free_mib"]
            print(f"  VRAM: free={free} MiB  used={snap['used_mib']} MiB  total={snap['total_mib']} MiB",
                  flush=True)

            if free <= SAFETY_MIB + 512:
                print(f"  [SKIP] VRAM too low (free={free} MiB, safety={SAFETY_MIB} MiB). "
                      f"Skipping {ctk}/{ctv}.", flush=True)
                all_results.append({"status": "skipped-vram",
                                     "ctk": ctk, "ctv": ctv, "n_ctx": 0})
                continue

            n_ctx = calc_max_ctx(ctk, ctv, free)
            mib_per_tok = kv_mib_per_token(ctk, ctv)
            kv_budget = n_ctx * mib_per_tok

            print(f"  ctx: {n_ctx} tokens  ({mib_per_tok*1000:.3f} bytes/token  "
                  f"KV budget={kv_budget:.0f} MiB  safety={SAFETY_MIB} MiB)",
                  flush=True)

            r = run_ctx_test(
                ctk=ctk, ctv=ctv, n_ctx=n_ctx,
                log_dir=log_dir,
                dry_run=args.dry_run,
                timeout_s=args.timeout,
            )
            all_results.append(r)

            # Brief inter-run pause — let VRAM stabilise / model unload
            if not args.dry_run:
                time.sleep(5)

    print_summary(all_results)

    print(f"\n  Logs written to: {log_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]", flush=True)
        sys.exit(130)
