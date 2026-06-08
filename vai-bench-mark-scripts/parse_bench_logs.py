#!/usr/bin/env python3
"""
parse_bench_logs.py — Reusable parser for TurboQuant benchmark JSONL logs.

Handles two log formats produced by this project:

  CTX-MAX logs   (CTX-MAX-*.jsonl)
    Written by ctx-max-test.py. One run per file. Key events:
      run_start       — model, ctk, ctv, n_ctx, n_prompt, n_gen, vram_free_mib
      ctx_max_result  — all timing fields: prompt_ts, eval_ts, load_ms, etc.
      run_end         — success, duration_s, peak_vram_mib, status

  CUDA-ONLY bench logs   (CUDA-ONLY-*.jsonl)
    Written by vai-cuda-bench-runner.py. One run per file. Key events:
      run_start         — model, ctk, ctv, n_ctx, ngl
      preflight_passed  — VRAM estimate breakdown
      attempt_success   — duration_s for the successful attempt
      run_perf_summary  — peak_vram_mib + list of {type, n_prompt/n_gen, avg_ts, stddev_ts}
      run_end           — success, duration_s, peak_vram_mib, status
      crash             — returncode (present in failed logs)

Usage (library):
    from parse_bench_logs import load_ctx_max_results, load_bench_results, summarise

Usage (CLI):
    python3 parse_bench_logs.py [--log-dir PATH] [--format ctx-max|bench|all]
                                [--csv] [--sort FIELD]

    Defaults to scanning both known log directories under the repo.
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CtxMaxRun:
    """One completed ctx-max-test run."""
    log_file:       str
    run_uuid:       str
    model:          str
    ctk:            str
    ctv:            str
    n_ctx:          int
    n_prompt:       int
    n_gen:          int
    status:         str           # completed | failed | dry-run | inprogress
    success:        bool

    # Timing (None if run failed before producing output)
    load_ms:        Optional[float] = None
    prompt_eval_ms: Optional[float] = None
    prompt_tokens:  Optional[int]   = None
    prompt_ts:      Optional[float] = None   # tokens/sec prefill
    eval_ms:        Optional[float] = None
    eval_runs:      Optional[int]   = None
    eval_ts:        Optional[float] = None   # tokens/sec generation
    total_ms:       Optional[float] = None

    # VRAM
    peak_vram_mib:  Optional[int]   = None
    vram_free_pre:  Optional[int]   = None   # free VRAM at run_start

    # Run metadata
    duration_s:     Optional[int]   = None
    flags_no_cnv:   bool            = False
    flags_no_warmup: bool           = False
    ts_start:       Optional[float] = None

    # Derived
    kv_cache_mib:   Optional[float] = field(default=None, repr=False)


@dataclass
class BenchRun:
    """One completed llama-bench run (CUDA-ONLY format)."""
    log_file:       str
    run_uuid:       str
    model:          str
    ctk:            str
    ctv:            str
    n_ctx:          Optional[int]
    ngl:            int
    status:         str
    success:        bool

    # Aggregated results: list of {type, n_prompt/n_gen, avg_ts, stddev_ts}
    results:        list = field(default_factory=list)

    # Convenience accessors (populated from results)
    gen_avg_ts:     Optional[float] = None
    gen_stddev_ts:  Optional[float] = None

    # VRAM
    peak_vram_mib:  Optional[int]   = None
    vram_estimate:  Optional[dict]  = None   # from preflight_passed

    # Run metadata
    duration_s:     Optional[int]   = None
    ts_start:       Optional[float] = None


@dataclass
class RawBenchResult:
    """One row from a raw llama-bench JSONL output line (build_commit present)."""
    log_file:       str
    model:          str
    ctk:            str
    ctv:            str
    n_prompt:       int
    n_gen:          int
    avg_ts:         float
    stddev_ts:      float
    build_commit:   str
    gpu_info:       str
    flash_attn:     bool
    no_kv_offload:  bool


# ---------------------------------------------------------------------------
# KV cache size estimator (mirrors ctx-max-test.py and vai-cuda-bench-runner.py)
# ---------------------------------------------------------------------------

_KV_BYTES = {
    "f16":    2.0,
    "bf16":   2.0,
    "q8_0":   1.0,
    "q4_0":   0.5,
    "q4_1":   0.5,
    "turbo4": 0.5,
    "turbo3": 0.375,
    "turbo2": 0.25,
}

# Qwen3.5 9B architecture constants (used for KV size estimates when not explicit)
_DEFAULT_N_LAYERS = 32
_DEFAULT_KV_HEADS = 4
_DEFAULT_HEAD_DIM = 128


def estimate_kv_mib(
    n_ctx: int,
    ctk: str,
    ctv: str,
    n_layers: int = _DEFAULT_N_LAYERS,
    kv_heads: int = _DEFAULT_KV_HEADS,
    head_dim: int = _DEFAULT_HEAD_DIM,
) -> float:
    bk = _KV_BYTES.get(ctk, 2.0)
    bv = _KV_BYTES.get(ctv, 2.0)
    return n_ctx * n_layers * kv_heads * head_dim * (bk + bv) / (1024 * 1024)


# ---------------------------------------------------------------------------
# JSONL line reader — tolerates mixed raw bench lines and event dicts
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for raw in path.read_text(errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError:
            pass
    return rows


# ---------------------------------------------------------------------------
# CTX-MAX log parser
# ---------------------------------------------------------------------------

def _parse_ctx_max_file(path: Path) -> Optional[CtxMaxRun]:
    rows = _read_jsonl(path)

    run_start   = None
    result      = None
    run_end     = None
    cmd_line    = ""

    for d in rows:
        ev = d.get("event", "")
        if ev == "run_start":
            run_start = d
        elif ev == "ctx_max_result":
            result = d
        elif ev == "run_end":
            run_end = d
        elif ev == "cmd":
            cmd_line = d.get("cmd", "")

    if run_start is None:
        return None

    status = _status_from_filename(path.name)
    success = run_end.get("success", False) if run_end else False

    run = CtxMaxRun(
        log_file    = path.name,
        run_uuid    = run_start.get("run_uuid", ""),
        model       = run_start.get("model", ""),
        ctk         = run_start.get("ctk", ""),
        ctv         = run_start.get("ctv", ""),
        n_ctx       = run_start.get("n_ctx", 0),
        n_prompt    = run_start.get("n_prompt", 0),
        n_gen       = run_start.get("n_gen", 0),
        status      = status,
        success     = success,
        vram_free_pre = run_start.get("vram_free_mib"),
        ts_start    = run_start.get("ts"),
        flags_no_cnv    = "-no-cnv" in cmd_line,
        flags_no_warmup = "--no-warmup" in cmd_line,
    )

    if result:
        run.load_ms        = result.get("load_ms")
        run.prompt_eval_ms = result.get("prompt_eval_ms")
        run.prompt_tokens  = result.get("prompt_tokens")
        run.prompt_ts      = result.get("prompt_ts")
        run.eval_ms        = result.get("eval_ms")
        run.eval_runs      = result.get("eval_runs")
        run.eval_ts        = result.get("eval_ts")
        run.total_ms       = result.get("total_ms")
        run.peak_vram_mib  = result.get("peak_vram_mib")
        run.duration_s     = result.get("duration_s")

    if run_end and run.peak_vram_mib is None:
        run.peak_vram_mib = run_end.get("peak_vram_mib")
    if run_end and run.duration_s is None:
        run.duration_s = run_end.get("duration_s")

    if run.n_ctx and run.ctk and run.ctv:
        run.kv_cache_mib = estimate_kv_mib(run.n_ctx, run.ctk, run.ctv)

    return run


def load_ctx_max_results(log_dir: Path) -> list[CtxMaxRun]:
    """
    Load all CTX-MAX log files from log_dir, including subdirectories.
    Returns a list of CtxMaxRun sorted by (model, ctk, ctv, ts_start).
    Skips dry-run files.
    """
    runs = []
    for path in sorted(log_dir.rglob("CTX-MAX-*.jsonl")):
        if "dry-run" in path.name:
            continue
        r = _parse_ctx_max_file(path)
        if r is not None:
            runs.append(r)
    runs.sort(key=lambda r: (r.model, r.ctk, r.ctv, r.ts_start or 0))
    return runs


# ---------------------------------------------------------------------------
# CUDA-ONLY bench log parser
# ---------------------------------------------------------------------------

def _parse_bench_file(path: Path) -> Optional[BenchRun]:
    rows = _read_jsonl(path)

    run_start   = None
    preflight   = None
    run_end     = None
    perf_summary = None
    raw_results: list[RawBenchResult] = []

    for d in rows:
        ev = d.get("event", "")
        if ev == "run_start":
            run_start = d
        elif ev == "preflight_passed":
            preflight = d
        elif ev == "run_end":
            run_end = d
        elif ev == "run_perf_summary":
            perf_summary = d
        elif "build_commit" in d and "avg_ts" in d:
            raw_results.append(RawBenchResult(
                log_file      = path.name,
                model         = _model_from_path(d.get("model_filename", "")),
                ctk           = d.get("type_k", ""),
                ctv           = d.get("type_v", ""),
                n_prompt      = d.get("n_prompt", 0),
                n_gen         = d.get("n_gen", 0),
                avg_ts        = d.get("avg_ts", 0.0),
                stddev_ts     = d.get("stddev_ts", 0.0),
                build_commit  = d.get("build_commit", ""),
                gpu_info      = d.get("gpu_info", ""),
                flash_attn    = bool(d.get("flash_attn", False)),
                no_kv_offload = bool(d.get("no_kv_offload", False)),
            ))

    if run_start is None:
        return None

    status  = _status_from_filename(path.name)
    success = run_end.get("success", False) if run_end else False

    results = []
    if perf_summary:
        results = perf_summary.get("results", [])
    elif raw_results:
        # Reconstruct from raw bench lines (older logs without run_perf_summary)
        for r in raw_results:
            if r.n_prompt > 0 and r.n_gen == 0:
                results.append({"type": "prompt", "n_prompt": r.n_prompt,
                                 "avg_ts": r.avg_ts, "stddev_ts": r.stddev_ts,
                                 "ctk": r.ctk, "ctv": r.ctv})
            elif r.n_gen > 0 and r.n_prompt == 0:
                results.append({"type": "gen", "n_gen": r.n_gen,
                                 "avg_ts": r.avg_ts, "stddev_ts": r.stddev_ts,
                                 "ctk": r.ctk, "ctv": r.ctv})

    gen_rows = [r for r in results if r.get("type") == "gen"]
    gen_avg    = gen_rows[0]["avg_ts"]    if gen_rows else None
    gen_stddev = gen_rows[0]["stddev_ts"] if gen_rows else None

    peak_vram = None
    if perf_summary:
        peak_vram = perf_summary.get("peak_vram_mib")
    if peak_vram is None and run_end:
        peak_vram = run_end.get("peak_vram_mib")

    run = BenchRun(
        log_file      = path.name,
        run_uuid      = run_start.get("run_uuid", ""),
        model         = run_start.get("model", ""),
        ctk           = run_start.get("ctk", ""),
        ctv           = run_start.get("ctv", ""),
        n_ctx         = run_start.get("n_ctx"),
        ngl           = run_start.get("ngl", 0),
        status        = status,
        success       = success,
        results       = results,
        gen_avg_ts    = gen_avg,
        gen_stddev_ts = gen_stddev,
        peak_vram_mib = peak_vram,
        vram_estimate = preflight.get("estimate") if preflight else None,
        duration_s    = run_end.get("duration_s") if run_end else None,
        ts_start      = run_start.get("ts"),
    )
    return run


def load_bench_results(log_dir: Path) -> list[BenchRun]:
    """
    Load all CUDA-ONLY bench log files from log_dir (and subdirs).
    Returns completed runs sorted by (model, ctk, ctv, ts_start).
    Skips zero-duration dry-run completions (duration_s == 0, no results).
    """
    runs = []
    for path in sorted(log_dir.rglob("CUDA-ONLY-*.jsonl")):
        r = _parse_bench_file(path)
        if r is None:
            continue
        # Skip dry-run ghost completions (duration=0, no bench data)
        if r.duration_s == 0 and not r.results:
            continue
        runs.append(r)
    runs.sort(key=lambda r: (r.model, r.ctk, r.ctv, r.ts_start or 0))
    return runs


def latest_bench_per_combo(runs: list[BenchRun]) -> list[BenchRun]:
    """
    Deduplicate bench runs: keep only the latest completed run per
    (model, ctk, ctv) combination. Falls back to latest of any status
    if no completed run exists for a combo.
    """
    from collections import defaultdict
    by_combo: dict[tuple, list[BenchRun]] = defaultdict(list)
    for r in runs:
        by_combo[(r.model, r.ctk, r.ctv)].append(r)

    best = []
    for combo, combo_runs in by_combo.items():
        combo_runs.sort(key=lambda r: r.ts_start or 0)
        with_data = [r for r in combo_runs if r.results]
        best.append(with_data[-1] if with_data else combo_runs[-1])
    best.sort(key=lambda r: (r.model, r.ctk, r.ctv))
    return best


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status_from_filename(name: str) -> str:
    for s in ("completed", "failed", "inprogress", "skipped", "dry-run"):
        if name.endswith(f"-{s}.jsonl"):
            return s
    return "unknown"


def _model_from_path(model_path: str) -> str:
    """Extract model name (stem without .gguf) from a full file path."""
    return Path(model_path).stem if model_path else ""


# ---------------------------------------------------------------------------
# Summarise helpers (mirrors the analysis done during the session)
# ---------------------------------------------------------------------------

def summarise_ctx_max(runs: list[CtxMaxRun]) -> None:
    completed = [r for r in runs if r.status == "completed" and r.prompt_ts is not None]
    if not completed:
        print("No completed ctx-max runs with timing data.")
        return

    _KV_ORDER = {"q8_0":1, "q4_0":2, "turbo4":3, "turbo3":4, "turbo2":5}
    completed.sort(key=lambda r: (_KV_ORDER.get(r.ctk, 9), _KV_ORDER.get(r.ctv, 9)))

    # Find baseline gen_ts for relative speedup column
    baseline_gen = next((r.eval_ts for r in completed
                         if r.ctk == "q8_0" and r.ctv == "q8_0"), None)

    sep = "-" * 108
    print(f"\n{'CTX-MAX SUMMARY':^108}")
    print(sep)
    hdr = (f"  {'ctk':8s} {'ctv':8s} {'n_ctx':>8s} {'KV MiB':>8s} "
           f"{'prefill t/s':>12s} {'gen t/s':>9s} {'vs q8_0':>8s} "
           f"{'peak VRAM':>10s} {'headroom':>9s}")
    print(hdr)
    print(sep)
    for r in completed:
        kv   = f"{r.kv_cache_mib:.0f}" if r.kv_cache_mib else "?"
        pts  = f"{r.prompt_ts:.1f}"    if r.prompt_ts   else "—"
        ets  = f"{r.eval_ts:.2f}"      if r.eval_ts     else "—"
        rel  = (f"{r.eval_ts / baseline_gen:.2f}×"
                if r.eval_ts and baseline_gen else "—")
        vram = f"{r.peak_vram_mib}M"   if r.peak_vram_mib else "?"
        head = (f"{8053 - r.peak_vram_mib}M"
                if r.peak_vram_mib else "?")
        print(f"  {r.ctk:8s} {r.ctv:8s} {r.n_ctx:>8d} {kv:>8s} "
              f"{pts:>12s} {ets:>9s} {rel:>8s} {vram:>10s} {head:>9s}")
    print(sep)


def summarise_bench(runs: list[BenchRun]) -> None:
    runs = latest_bench_per_combo([r for r in runs if r.results])
    if not runs:
        print("No bench runs with result data.")
        return

    _KV_ORDER = {"q8_0":1, "q4_0":2, "turbo4":3, "turbo3":4, "turbo2":5}
    runs.sort(key=lambda r: (r.model, _KV_ORDER.get(r.ctk, 9),
                             _KV_ORDER.get(r.ctv, 9)))

    sep = "-" * 90
    print(f"\n{'CUDA-ONLY BENCH SUMMARY':^90}")
    print(sep)

    current_model = None
    for r in runs:
        if r.model != current_model:
            current_model = r.model
            print(f"\n  Model: {r.model}")
            print(f"  {'ctk':8s} {'ctv':8s} {'gen t/s':>9s} {'±stddev':>8s} "
                  f"{'peak VRAM':>10s} {'duration':>9s}")
            print(f"  {'-'*70}")

        gen_ts  = f"{r.gen_avg_ts:.2f}"    if r.gen_avg_ts    else "—"
        std     = f"{r.gen_stddev_ts:.3f}" if r.gen_stddev_ts else "—"
        vram    = f"{r.peak_vram_mib}M"    if r.peak_vram_mib else "?"
        dur     = f"{r.duration_s}s"       if r.duration_s    else "?"
        print(f"  {r.ctk:8s} {r.ctv:8s} {gen_ts:>9s} {std:>8s} "
              f"{vram:>10s} {dur:>9s}")

        prompt_rows = sorted(
            [x for x in r.results if x.get("type") == "prompt"],
            key=lambda x: x.get("n_prompt", 0),
        )
        if prompt_rows:
            print(f"    {'tokens':>8s}  {'pp t/s':>10s}  {'±stddev':>8s}")
            for p in prompt_rows:
                print(f"    {p['n_prompt']:>8d}  {p['avg_ts']:>10.1f}  "
                      f"{p['stddev_ts']:>8.3f}")

    print()


def summarise(log_dir: Path) -> None:
    """Load and print both ctx-max and bench summaries from log_dir."""
    ctx_runs   = load_ctx_max_results(log_dir)
    bench_runs = load_bench_results(log_dir)

    if ctx_runs:
        summarise_ctx_max(ctx_runs)
    else:
        print("No ctx-max logs found.")

    if bench_runs:
        summarise_bench(bench_runs)
    else:
        print("No bench logs found.")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def to_csv_ctx_max(runs: list[CtxMaxRun], out=sys.stdout) -> None:
    completed = [r for r in runs if r.status == "completed"]
    if not completed:
        return
    field_names = [f.name for f in fields(CtxMaxRun)]
    w = csv.DictWriter(out, fieldnames=field_names)
    w.writeheader()
    for r in completed:
        w.writerow(asdict(r))


def to_csv_bench(runs: list[BenchRun], out=sys.stdout) -> None:
    # Flatten: one row per (run × result_row)
    rows = []
    for r in runs:
        base = {
            "log_file":      r.log_file,
            "run_uuid":      r.run_uuid,
            "model":         r.model,
            "ctk":           r.ctk,
            "ctv":           r.ctv,
            "n_ctx":         r.n_ctx,
            "ngl":           r.ngl,
            "status":        r.status,
            "peak_vram_mib": r.peak_vram_mib,
            "duration_s":    r.duration_s,
        }
        if not r.results:
            rows.append({**base, "type":"", "n_prompt":"", "n_gen":"",
                         "avg_ts":"", "stddev_ts":""})
        for res in r.results:
            rows.append({
                **base,
                "type":      res.get("type",""),
                "n_prompt":  res.get("n_prompt",""),
                "n_gen":     res.get("n_gen",""),
                "avg_ts":    res.get("avg_ts",""),
                "stddev_ts": res.get("stddev_ts",""),
            })
    if not rows:
        return
    w = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_DIR = _REPO_ROOT / "lamma-bench-logs"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse TurboQuant benchmark JSONL logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--log-dir", default=str(_DEFAULT_LOG_DIR),
        help=f"Root directory to search for log files (default: {_DEFAULT_LOG_DIR})",
    )
    parser.add_argument(
        "--format", choices=["ctx-max", "bench", "all"], default="all",
        help="Which log format to parse (default: all)",
    )
    parser.add_argument(
        "--csv", action="store_true",
        help="Output CSV to stdout instead of formatted tables",
    )
    parser.add_argument(
        "--completed-only", action="store_true", default=True,
        help="Only show completed runs (default: true)",
    )
    parser.add_argument(
        "--include-incomplete", action="store_true",
        help="Include failed/inprogress runs in output",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"ERROR: log directory not found: {log_dir}", file=sys.stderr)
        sys.exit(1)

    show_ctx   = args.format in ("ctx-max", "all")
    show_bench = args.format in ("bench", "all")

    if args.csv:
        if show_ctx:
            runs = load_ctx_max_results(log_dir)
            to_csv_ctx_max(runs)
        if show_bench:
            runs = load_bench_results(log_dir)
            to_csv_bench(runs)
    else:
        if show_ctx:
            runs = load_ctx_max_results(log_dir)
            if args.include_incomplete:
                summarise_ctx_max(runs)
            else:
                summarise_ctx_max([r for r in runs if r.status == "completed"])
        if show_bench:
            runs = load_bench_results(log_dir)
            summarise_bench(runs)


if __name__ == "__main__":
    main()
