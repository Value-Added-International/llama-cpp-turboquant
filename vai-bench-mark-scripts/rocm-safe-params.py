#!/usr/bin/env python3
"""
rocm-safe-params.py — ROCm gfx1103 flash-attention kernel safety analyser.

Purpose
-------
Given a llama-bench JSONL log directory (or a single JSONL file), determine
which combinations of KV-cache type (ctk/ctv), context length (n_prompt),
and flash-attention state are SAFE (no GPU hang) on AMD gfx1103 (Radeon 780M)
given the known kernel-selection rules in fattn.cu / fattn-common.cuh.

The analyser also accepts raw parameters directly (--head-dim, --n-ctx, etc.)
to evaluate whether a *proposed* bench run would be safe before you run it.

Key rules encoded
-----------------
1. gfx1103 is RDNA3 → `ggml_cuda_should_use_wmma_fattn()` returns true.
2. Quantised KV types (q4_0, q8_0, turbo2, turbo3, turbo4) are forced onto
   the VEC path by the `#ifdef GGML_USE_HIP` early-return in
   `ggml_cuda_get_best_fattn_kernel()` (line 560 fattn.cu).
3. f16/bf16 KV with head_dim not in {512,576,640,192,40,72} and
   K->ne[1] % 256 == 0  → WMMA path.
4. KV_max (the mask-based OOB clamp) is only computed when
   Q->ne[1] >= 1024  (line 1412 fattn-common.cuh).
   Below that threshold the kernel iterates to K->ne[1] (== n_ctx), reading
   uninitialised KV rows → GPU hang on 780M UMA.
5. For the VEC path, `can_use_vector_kernel` requires K->ne[1] % 256 == 0.
   If the KV cache n_ctx is not a multiple of 256 the VEC path is skipped
   and TILE is used instead — TILE is safe (no OOB issue).

Usage
-----
  # Analyse a directory of JSONL logs:
  python3 rocm-safe-params.py --log-dir lamma-bench-logs/

  # Analyse a single file:
  python3 rocm-safe-params.py --log-file lamma-bench-logs/foo.jsonl

  # Evaluate proposed parameters directly:
  python3 rocm-safe-params.py --head-dim 256 --n-ctx 8192 --n-prompt 512 \
      --type-k q8_0 --type-v q8_0 --flash-attn 1

  # Show only unsafe results:
  python3 rocm-safe-params.py --log-dir lamma-bench-logs/ --unsafe-only
"""

import argparse
import json
import os
import sys
from pathlib import Path

FATTN_KQ_STRIDE = 256
KV_MAX_BATCH_THRESHOLD = 1024  # Q->ne[1] must be >= this for KV_max to be computed

QUANTISED_TYPES = {
    "q4_0", "q4_1", "q5_0", "q5_1", "q8_0",
    "turbo2", "turbo3", "turbo4",
    "turbo2_0", "turbo3_0", "turbo4_0",
    "iq2_xxs", "iq2_xs", "iq3_xxs", "iq4_nl", "iq3_s", "iq2_s", "iq4_xs",
}

WMMA_EXCLUDED_HEAD_DIMS = {40, 72, 192, 512, 576, 640}

# llama-bench doesn't log head_dim. Derive it from model_type string.
# Pattern: model family → head_dim. Add entries as needed.
MODEL_HEAD_DIM_MAP = {
    "gemma4":   256,   # Gemma 4 (all sizes) — 256-dim attention heads
    "gemma3":   256,
    "gemma2":   256,
    "llama":    128,
    "mistral":  128,
    "qwen2":    128,
    "qwen3":    128,
    "deepseek": 128,
    "phi3":     96,
    "phi4":     96,
    "falcon":   64,
}

def infer_head_dim(model_type: str) -> int | None:
    """Best-effort head_dim from model_type string recorded in bench JSONL."""
    if not model_type:
        return None
    mt = model_type.lower()
    for key, dim in MODEL_HEAD_DIM_MAP.items():
        if key in mt:
            return dim
    return None


def is_quantised(t: str) -> bool:
    return t.lower() in QUANTISED_TYPES


def classify(
    head_dim: int,
    n_ctx: int,
    n_prompt: int,      # Q->ne[1] at prefill time  (batch size)
    type_k: str,
    type_v: str,
    flash_attn: bool,
) -> dict:
    """
    Return a dict with:
      kernel   : which FA kernel would be selected
      safe     : True / False
      reason   : human-readable explanation
    """
    if not flash_attn:
        return dict(kernel="none", safe=True, reason="Flash attention disabled — CPU path, no GPU hang risk")

    quant_kv = is_quantised(type_k) or is_quantised(type_v)
    kv_stride_ok = (n_ctx % FATTN_KQ_STRIDE == 0)
    can_use_vec = (head_dim <= 256 and head_dim % 64 == 0
                   and head_dim not in WMMA_EXCLUDED_HEAD_DIMS
                   and kv_stride_ok)

    # ── Rule 2: quantised KV → VEC (forced on HIP) ──────────────────────────
    if quant_kv and can_use_vec:
        kernel = "VEC"
        kv_max_active = (n_prompt >= KV_MAX_BATCH_THRESHOLD)
        if kv_max_active:
            return dict(kernel=kernel, safe=True,
                        reason=f"VEC path; KV_max computed (n_prompt={n_prompt} >= {KV_MAX_BATCH_THRESHOLD}); "
                               f"kernel bounded to actual KV length")
        else:
            return dict(kernel=kernel, safe=False,
                        reason=f"VEC path; KV_max NOT computed (n_prompt={n_prompt} < {KV_MAX_BATCH_THRESHOLD}); "
                               f"kernel iterates to n_ctx={n_ctx} with only {n_prompt} valid rows → UMA OOB → GPU hang")

    if quant_kv and not can_use_vec:
        # TILE fallback — safe, no OOB (tile kernel uses mask-guided bounds)
        return dict(kernel="TILE", safe=True,
                    reason=f"Quantised KV but can_use_vector_kernel=False "
                           f"(head_dim={head_dim}, n_ctx%{FATTN_KQ_STRIDE}={n_ctx%FATTN_KQ_STRIDE}); "
                           f"falls through to TILE — safe")

    # ── f16 / bf16 KV ────────────────────────────────────────────────────────
    # gfx1103 is RDNA3 → ggml_cuda_should_use_wmma_fattn() == True
    # WMMA path guard:
    wmma_eligible = (
        kv_stride_ok
        and head_dim not in WMMA_EXCLUDED_HEAD_DIMS
    )
    if wmma_eligible and n_prompt > 2:
        kernel = "WMMA"
        # WMMA kernel uses KV_max if provided, but launch_fattn passes KV_max.ptr
        # which is only allocated when Q->ne[1] >= KV_MAX_BATCH_THRESHOLD.
        # WMMA kernel: k_VKQ_max = KV_max ? KV_max[...] : ne11
        # ne11 == K->ne[1] == n_ctx (full pre-allocated size)
        kv_max_active = (n_prompt >= KV_MAX_BATCH_THRESHOLD)
        if kv_max_active:
            return dict(kernel=kernel, safe=True,
                        reason=f"WMMA path; KV_max computed (n_prompt={n_prompt} >= {KV_MAX_BATCH_THRESHOLD})")
        else:
            return dict(kernel=kernel, safe=False,
                        reason=f"WMMA path; KV_max NOT computed (n_prompt={n_prompt} < {KV_MAX_BATCH_THRESHOLD}); "
                               f"kernel reads to ne11=n_ctx={n_ctx} → OOB on gfx1103 UMA → GPU hang")

    if wmma_eligible and n_prompt <= 2:
        # VEC is preferred when n_prompt <= 2 even on WMMA-capable hardware
        kernel = "VEC"
        kv_max_active = (n_prompt >= KV_MAX_BATCH_THRESHOLD)
        # n_prompt <= 2 always < 1024, so KV_max is never active here
        # But n_prompt <= 2 means only decode phase; K->ne[1] is the cache
        # which is filled up to the actual sequence — depends on context
        return dict(kernel=kernel, safe=False,
                    reason=f"WMMA eligible but n_prompt={n_prompt} ≤ 2 → VEC chosen; "
                           f"KV_max not computed → OOB risk (n_ctx={n_ctx})")

    # Fallback: VEC or TILE for f16 without WMMA path
    if can_use_vec:
        kv_max_active = (n_prompt >= KV_MAX_BATCH_THRESHOLD)
        if kv_max_active:
            return dict(kernel="VEC", safe=True,
                        reason=f"VEC path (non-WMMA); KV_max computed (n_prompt={n_prompt})")
        else:
            return dict(kernel="VEC", safe=False,
                        reason=f"VEC path (non-WMMA); KV_max NOT computed (n_prompt={n_prompt} < {KV_MAX_BATCH_THRESHOLD}) → OOB")

    return dict(kernel="TILE", safe=True,
                reason="TILE fallback — safe (mask-bounded iteration)")


def analyse_result(r: dict) -> dict:
    """Analyse a single bench result JSON line."""
    model_type = r.get("model_type", "")
    # head_dim: llama-bench doesn't log it directly; infer from model family.
    head_dim = r.get("ne00") or r.get("head_dim") or infer_head_dim(model_type)

    # n_ctx: llama-bench pre-allocates KV cache for n_batch tokens.
    # n_batch is set to the largest -p value in the run (or user-specified).
    # The KV cache row count K->ne[1] == n_batch (not n_prompt).
    n_ctx    = r.get("n_ctx") or r.get("n_kv") or r.get("n_batch") or 2048

    n_prompt = r.get("n_prompt", 0)
    type_k   = r.get("type_k", "f16")
    type_v   = r.get("type_v", "f16")
    fa       = bool(r.get("flash_attn", True))

    # Only ROCm backend is affected by these kernel selection bugs.
    backend  = r.get("backends", "")
    if "ROCm" not in backend and "HIP" not in backend:
        result = dict(kernel="n/a (not ROCm)", safe=True,
                      reason=f"Backend={backend!r} — not ROCm, kernel analysis not applicable")
        result.update(dict(model=model_type, n_prompt=n_prompt, type_k=type_k,
                           type_v=type_v, n_ctx=n_ctx, head_dim=head_dim or 0, flash_attn=fa))
        return result

    if head_dim is None:
        return dict(kernel="unknown", safe=None,
                    reason=f"Cannot infer head_dim from model_type={model_type!r}; add to MODEL_HEAD_DIM_MAP",
                    model=model_type, n_prompt=n_prompt,
                    type_k=type_k, type_v=type_v, n_ctx=n_ctx,
                    head_dim=0, flash_attn=fa)

    result = classify(head_dim, n_ctx, n_prompt, type_k, type_v, fa)
    result.update({
        "model":    r.get("model_type", "?"),
        "n_prompt": n_prompt,
        "type_k":   type_k,
        "type_v":   type_v,
        "n_ctx":    n_ctx,
        "head_dim": head_dim,
        "flash_attn": fa,
    })
    return result


def analyse_file(path: Path, unsafe_only: bool = False):
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Only process actual bench result lines (have n_prompt + type_k)
            if "n_prompt" not in d or "type_k" not in d:
                continue
            r = analyse_result(d)
            if unsafe_only and r["safe"]:
                continue
            results.append(r)
    return results


def print_results(results: list, source: str):
    if not results:
        print(f"  (no results)")
        return
    for r in results:
        safe_str = "SAFE  " if r["safe"] else "UNSAFE"
        print(f"  [{safe_str}] kernel={r['kernel']:5s}  "
              f"ctk={r['type_k']:8s} ctv={r['type_v']:8s}  "
              f"n_prompt={r['n_prompt']:5d}  n_ctx={r['n_ctx']:5d}  "
              f"head_dim={r['head_dim']:3d}  fa={r['flash_attn']}")
        print(f"           → {r['reason']}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--log-dir",  type=Path, help="Directory of .jsonl bench logs to analyse")
    ap.add_argument("--log-file", type=Path, help="Single .jsonl bench log to analyse")
    ap.add_argument("--unsafe-only", action="store_true", help="Only show unsafe entries")

    # Direct parameter mode
    ap.add_argument("--head-dim",  type=int,   help="Attention head dimension (e.g. 256)")
    ap.add_argument("--n-ctx",     type=int,   help="Pre-allocated KV cache size (n_ctx)")
    ap.add_argument("--n-prompt",  type=int,   help="Batch size / prefill token count (Q->ne[1])")
    ap.add_argument("--type-k",    type=str,   default="f16", help="KV cache K type (e.g. q8_0, turbo3, f16)")
    ap.add_argument("--type-v",    type=str,   default="f16", help="KV cache V type")
    ap.add_argument("--flash-attn",type=int,   default=1, help="1=flash attn enabled (default), 0=disabled")

    args = ap.parse_args()

    # ── Direct param mode ─────────────────────────────────────────────────────
    if args.head_dim is not None:
        if args.n_ctx is None or args.n_prompt is None:
            ap.error("--head-dim requires --n-ctx and --n-prompt")
        r = classify(
            head_dim   = args.head_dim,
            n_ctx      = args.n_ctx,
            n_prompt   = args.n_prompt,
            type_k     = args.type_k,
            type_v     = args.type_v,
            flash_attn = bool(args.flash_attn),
        )
        safe_str = "SAFE" if r["safe"] else "UNSAFE"
        print(f"\nResult: {safe_str}")
        print(f"  Kernel  : {r['kernel']}")
        print(f"  Reason  : {r['reason']}")
        sys.exit(0 if r["safe"] else 1)

    # ── Log file / dir mode ───────────────────────────────────────────────────
    if args.log_file:
        print(f"\n=== {args.log_file} ===")
        results = analyse_file(args.log_file, args.unsafe_only)
        print_results(results, str(args.log_file))

    elif args.log_dir:
        jsonl_files = sorted(args.log_dir.glob("*.jsonl"))
        if not jsonl_files:
            print(f"No .jsonl files found in {args.log_dir}")
            sys.exit(1)
        for f in jsonl_files:
            results = analyse_file(f, args.unsafe_only)
            if results:
                print(f"\n=== {f.name} ===")
                print_results(results, f.name)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
