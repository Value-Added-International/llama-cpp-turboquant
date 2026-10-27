# Product Context

## Why It Exists / Problems It Solves

llama.cpp's default quantization is MSE-driven, which degrades KV-cache fidelity under high compression. This fork
integrates **TurboQuant+**: Walsh-Hadamard-rotated polar codebook quantization, attention-gated sparse V dequantization,
and layer-aware "Boundary V" protection.

How it should work:
- **Asymmetric K/V policy** — V tolerates aggressive compression (free); K must stay high-precision (everything). Never start
  symmetric; keep K at `f16`/`q8_0` and ratchet V from `turbo4` (lightest) to `turbo2` (heaviest).
- **Boundary V** — auto-enables on layers where aggressive V quantization degrades quality; leaves the rest fully aggressive.
- **Sparse V dequant** — skips V dequant for positions whose softmax attention weight is below a threshold.

It is additive: new types are opt-in via the standard `--cache-type-k` / `--cache-type-v` and `llama-quantize` interfaces.

## Palace references

PALACE-REF | IDENTITY | vai_llama_cpp_turboquant/identity | source README.md | 2026-09-27 | zoocode-tiel-coder-35b-A3B-MTP-262144-ctx | product purpose and asymmetric K/V design
