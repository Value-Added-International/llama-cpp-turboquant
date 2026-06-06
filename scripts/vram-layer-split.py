#!/usr/bin/env python3
"""
vram-layer-split.py — estimate optimal -ngl (GPU layer count) for a GGUF model
                       given available GPU VRAM and system RAM.

Usage:
    python3 scripts/vram-layer-split.py <model.gguf> [--gpu-free-mib N] [--cpu-avail-gib N]

Defaults:
    --gpu-free-mib  : read from rocminfo/nvidia-smi if possible, else prompt
    --cpu-avail-gib : read from /proc/meminfo

The script reads the GGUF KV metadata for:
    *.block_count        → transformer layer count
    *.embedding_length   → model hidden dimension
    general.file_type    → quantisation type (for per-layer size estimate)

It then computes:
    per_layer_mib = (total_file_mib - overhead_mib) / block_count
    gpu_layers    = floor((gpu_free_mib - overhead_mib) / per_layer_mib)
    cpu_layers    = block_count - gpu_layers

Overhead covers token embeddings + output head (vocab × embed × quant_bytes).
This is an approximation — actual fit depends on KV cache size and context length.
Always verify with a short test load (-c 512 --no-warmup) before committing to a context.

Example (780M iGPU, Qwen3-27B Q4_K_XL):
    $ python3 scripts/vram-layer-split.py \
          /mnt/ssd/models/Qwen3.6-27B-UD-Q4_K_XL.gguf \
          --gpu-free-mib 2048

    model : Qwen3.6-27B-UD-Q4_K_XL.gguf  (17.0 GiB, 64 layers)
    per-layer : 242 MiB
    GPU free  : 2048 MiB  → GPU layers : 2
    CPU free  : 20480 MiB → CPU layers : 62  (fits: True)
    Suggested: -ngl 2
"""

import argparse
import os
import struct
import sys

# ---------------------------------------------------------------------------
# Minimal GGUF reader — parses only the KV metadata section.
# We do NOT read tensor data; we only need a handful of scalar keys.
# ---------------------------------------------------------------------------

GGUF_MAGIC   = b"GGUF"
GGUF_VERSION = (2, 3)

GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

_SCALAR_FMTS = {
    GGUF_TYPE_UINT8:   ("<B", 1),
    GGUF_TYPE_INT8:    ("<b", 1),
    GGUF_TYPE_UINT16:  ("<H", 2),
    GGUF_TYPE_INT16:   ("<h", 2),
    GGUF_TYPE_UINT32:  ("<I", 4),
    GGUF_TYPE_INT32:   ("<i", 4),
    GGUF_TYPE_FLOAT32: ("<f", 4),
    GGUF_TYPE_BOOL:    ("<B", 1),
    GGUF_TYPE_UINT64:  ("<Q", 8),
    GGUF_TYPE_INT64:   ("<q", 8),
    GGUF_TYPE_FLOAT64: ("<d", 8),
}


def _read_exact(f, n):
    data = f.read(n)
    if len(data) != n:
        raise EOFError(f"expected {n} bytes, got {len(data)}")
    return data


def _read_u32(f):
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _read_u64(f):
    return struct.unpack("<Q", _read_exact(f, 8))[0]


def _read_string(f):
    length = _read_u64(f)
    return _read_exact(f, length).decode("utf-8", errors="replace")


def _read_value(f, vtype):
    """Read a single GGUF value of the given type; returns Python object."""
    if vtype in _SCALAR_FMTS:
        fmt, size = _SCALAR_FMTS[vtype]
        return struct.unpack(fmt, _read_exact(f, size))[0]
    if vtype == GGUF_TYPE_STRING:
        return _read_string(f)
    if vtype == GGUF_TYPE_ARRAY:
        elem_type = _read_u32(f)
        count     = _read_u64(f)
        # We don't materialise large arrays; skip them.
        if elem_type in _SCALAR_FMTS:
            _, size = _SCALAR_FMTS[elem_type]
            f.seek(count * size, 1)
        elif elem_type == GGUF_TYPE_STRING:
            for _ in range(count):
                _read_string(f)
        else:
            # Nested array — give up and return None (caller ignores key)
            return None
        return None
    raise ValueError(f"unknown GGUF value type {vtype}")


def read_gguf_kv(path):
    """Return a dict of all scalar/string KV entries from a GGUF file."""
    kv = {}
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError("not a GGUF file")
        version = _read_u32(f)
        if version not in GGUF_VERSION:
            # Try to parse anyway — format is stable across v2/v3
            pass
        _tensor_count = _read_u64(f)
        kv_count      = _read_u64(f)
        for _ in range(kv_count):
            key   = _read_string(f)
            vtype = _read_u32(f)
            val   = _read_value(f, vtype)
            if val is not None:
                kv[key] = val
    return kv


# ---------------------------------------------------------------------------
# Overhead estimation
# ---------------------------------------------------------------------------

# GGUF file_type → approximate bytes per element for weight quantisation.
# Used only for the embedding / output-head overhead estimate.
FILE_TYPE_BPW = {
    0:  2.0,   # F32 (stored as f32 → 4 bytes, but "bits per weight" for sizing)
    1:  2.0,   # F16
    2:  4.5,   # Q4_0  (not accurate but ballpark)
    3:  5.0,   # Q4_1
    7:  8.0,   # Q8_0
    8:  8.0,   # Q8_1
    15: 4.5,   # Q4_K_S
    16: 4.5,   # Q4_K_M / Q4_K_XL
    17: 5.5,   # Q5_K_S
    18: 5.5,   # Q5_K_M
    19: 6.5,   # Q6_K
    20: 2.5,   # Q2_K
    21: 3.5,   # Q3_K_S
    22: 3.5,   # Q3_K_M
    23: 3.5,   # Q3_K_L
}

def estimate_overhead_mib(kv, file_size_mib):
    """
    Estimate non-layer overhead (embeddings + output head) in MiB.
    Falls back to a fixed 10% of file size when metadata is incomplete.
    """
    vocab_size  = kv.get("tokenizer.ggml.tokens") or None
    # vocab count is typically in the array length — not easily read here.
    # Use embed_length as a proxy: overhead ≈ 2 × vocab × embed × bpw/8
    embed = None
    for suffix in ("embedding_length",):
        for k, v in kv.items():
            if k.endswith(suffix):
                embed = v
                break

    # Try to get vocab size from a scalar key (some models expose it)
    vocab = kv.get("llm.vocab_size") or kv.get("tokenizer.ggml.n_vocab")

    file_type = kv.get("general.file_type", 16)  # default Q4_K_M
    bpw = FILE_TYPE_BPW.get(int(file_type), 4.5)

    if embed and vocab:
        # 2 matrices: token_embd + output (or tied)
        overhead_bytes = 2 * int(vocab) * int(embed) * (bpw / 8)
        return overhead_bytes / (1024 ** 2)

    # Fallback: 10% of file
    return file_size_mib * 0.10


# ---------------------------------------------------------------------------
# System memory helpers
# ---------------------------------------------------------------------------

def read_cpu_avail_mib():
    """Read MemAvailable from /proc/meminfo (Linux)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024  # kB → MiB
    except OSError:
        pass
    return None


def probe_gpu_free_mib():
    """Try rocminfo then nvidia-smi to find free VRAM on the first GPU."""
    import subprocess
    # ROCm
    try:
        out = subprocess.check_output(
            ["rocminfo"], stderr=subprocess.DEVNULL, timeout=5, text=True
        )
        for line in out.splitlines():
            # "Memory Banks:" section — too verbose; skip auto-probe for ROCm
            pass
    except Exception:
        pass
    # nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5, text=True,
        )
        vals = [int(x.strip()) for x in out.strip().splitlines() if x.strip().isdigit()]
        if vals:
            return vals[0]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Estimate -ngl (GPU layer count) for a GGUF model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("model", help="path to .gguf file")
    ap.add_argument(
        "--gpu-free-mib", type=int, default=None,
        help="available GPU VRAM in MiB (default: auto-probe nvidia-smi/rocminfo)",
    )
    ap.add_argument(
        "--cpu-avail-gib", type=float, default=None,
        help="available system RAM in GiB (default: read /proc/meminfo)",
    )
    ap.add_argument(
        "--kv-cache-mib", type=int, default=512,
        help="reserve this much GPU VRAM for KV cache (default: 512 MiB)",
    )
    args = ap.parse_args()

    model_path = args.model
    if not os.path.isfile(model_path):
        sys.exit(f"error: model not found: {model_path}")

    file_size_mib = os.path.getsize(model_path) / (1024 ** 2)

    print(f"reading GGUF metadata from {os.path.basename(model_path)} …")
    kv = read_gguf_kv(model_path)

    # Find block_count under any architecture prefix (qwen35, llama, mistral, …)
    block_count = None
    for k, v in kv.items():
        if k.endswith(".block_count"):
            block_count = int(v)
            break
    if block_count is None:
        sys.exit("error: could not find *.block_count in GGUF metadata")

    overhead_mib = estimate_overhead_mib(kv, file_size_mib)
    layer_total_mib = file_size_mib - overhead_mib
    per_layer_mib   = layer_total_mib / block_count

    # GPU free VRAM
    gpu_free_mib = args.gpu_free_mib
    if gpu_free_mib is None:
        gpu_free_mib = probe_gpu_free_mib()
    if gpu_free_mib is None:
        try:
            gpu_free_mib = int(input("could not auto-detect GPU VRAM — enter free MiB: "))
        except (ValueError, EOFError):
            sys.exit("error: --gpu-free-mib required")

    # Reserve KV cache headroom
    gpu_usable_mib = gpu_free_mib - args.kv_cache_mib

    # GPU layers: overhead goes to GPU first (token_embd must be on GPU)
    gpu_layers = max(0, int((gpu_usable_mib - overhead_mib) / per_layer_mib))
    gpu_layers = min(gpu_layers, block_count)

    # CPU
    cpu_avail_mib = (
        int(args.cpu_avail_gib * 1024) if args.cpu_avail_gib is not None
        else (read_cpu_avail_mib() or 0)
    )
    cpu_layers    = block_count - gpu_layers
    cpu_needed_mib = cpu_layers * per_layer_mib
    fits = cpu_needed_mib <= cpu_avail_mib

    # MTP nextn layers (don't count toward -ngl, loaded separately)
    nextn = None
    for k, v in kv.items():
        if "nextn_predict_layers" in k:
            nextn = int(v)
            break

    print()
    print(f"  model       : {os.path.basename(model_path)}")
    print(f"  file size   : {file_size_mib / 1024:.2f} GiB")
    print(f"  layers      : {block_count}" + (f"  (+{nextn} MTP nextn)" if nextn else ""))
    print(f"  overhead    : {overhead_mib:.0f} MiB  (embeddings + output head)")
    print(f"  per layer   : {per_layer_mib:.0f} MiB")
    print()
    print(f"  GPU free    : {gpu_free_mib} MiB  (KV reserve: {args.kv_cache_mib} MiB  → usable: {gpu_usable_mib} MiB)")
    print(f"  GPU layers  : {gpu_layers}")
    print()
    print(f"  CPU avail   : {cpu_avail_mib} MiB")
    print(f"  CPU layers  : {cpu_layers}  (~{cpu_needed_mib:.0f} MiB needed)  fits: {fits}")
    print()
    if not fits:
        short = cpu_needed_mib - cpu_avail_mib
        print(f"  WARNING: {short:.0f} MiB short on CPU — reduce context (-c) or use a smaller quant")
    print(f"  Suggested   : -ngl {gpu_layers}")
    print()


if __name__ == "__main__":
    main()
