# Palace Map — vai_llama_cpp_turboquant

## Identity

Wing: vai_llama_cpp_turboquant
Project name: llama.cpp + TurboQuant+ (fork of ggml-org/llama.cpp integrating the TurboQuant+ codec stack)
Previous folder names: none (first onboard)
Last reconciled with palace: 2026-09-27 by zoocode-tiel-coder-35b-A3B-MTP-262144-ctx

## Mine roots

MINE-ROOT | /home/none/git/vai-llama-cpp-turboquant | 2026-09-27 | zoocode-tiel-coder-35b-A3B-MTP-262144-ctx | bootstrap

## Rooms

ROOM | vai_llama_cpp_turboquant/backend | per-backend quant kernels | ggml/
ROOM | vai_llama_cpp_turboquant/benches | benchmark harnesses | benches/
ROOM | vai_llama_cpp_turboquant/cmake | CMake build configs | cmake/
ROOM | vai_llama_cpp_turboquant/conversion | HF→GGUF converters | conversion/
ROOM | vai_llama_cpp_turboquant/frontend | server web UI | tools/server/webui
ROOM | vai_llama_cpp_turboquant/ggml | tensor compute layer + kernels | ggml/
ROOM | vai_llama_cpp_turboquant/planning | planning notes | plans/
ROOM | vai_llama_cpp_turboquant/scripts | build/benchmark scripts | scripts/
ROOM | vai_llama_cpp_turboquant/testing | C++ unit tests + snapshots | tests/
ROOM | vai_llama_cpp_turboquant/vendor | vendored deps | vendor/
ROOM | vai_llama_cpp_turboquant/general | misc | root
backend | backend kernels, quant ops | ggml/ per-backend kernel sources
benches | benchmark harnesses | benches/
cmake | CMake build configs | cmake/
conversion | HF→GGUF converters | conversion/
frontend | server web UI | tools/server/webui
general | misc | root
ggml | tensor compute layer + kernels | ggml/
planning | planning notes | plans/
scripts | build/benchmark scripts | scripts/
testing | C++ unit tests + snapshots | tests/
vendor | vendored deps | vendor/

## Latest handoff

HANDOFF | stream project/vai_llama_cpp_turboquant | event evt_20260927T075215_806690b7d0e3 | 2026-09-27T07:52:15Z | zoocode-tiel-coder-35b-A3B-MTP-262144-ctx | ready | handoff drawer drawer_vai_llama_cpp_turboquant_handoff_4817f757a8564b23d5be113c

## Latest manifest snapshot (machine-local)

SNAPSHOT | 20260927T112555Z | files 9741 | sha256 db42b649c2676f3619bc319922ff79b3c11889e6bc05ccddbebbbfbeae0f7ec9

## Shared dependencies

SHARED | TheTom/turboquant_plus | codec design / calibration / validation papers (separate repo, not a shared wing) | runtime integration lives here; codec theory lives there

## Knowledge graph

KG | vai_llama_cpp_turboquant → documented_in → vai_llama_cpp_turboquant:README.md | valid from 2026-09-27 | canonical source of project identity
KG | vai_llama_cpp_turboquant → depends_on → TheTom/turboquant_plus | valid from 2026-09-27 | codec theory dependency (separate repo)

## Tunnels

TUNNEL | (none) — no shared_* wings used | — | this project has no k3s/infra dependencies; codec papers are in a separate repo

## Drawer index

### IDENTITY
PALACE-REF | IDENTITY | vai_llama_cpp_turboquant/identity | drawer drawer_vai_llama_cpp_turboquant_identity_d8a65b9f5b32b19cd6be4943 | 2026-09-27 | zoocode-tiel-coder-35b-A3B-MTP-262144-ctx | project identity, scope, fingerprint tokens

### PATTERN
PALACE-REF | PATTERN | vai_llama_cpp_turboquant/patterns | drawer drawer_vai_llama_cpp_turboquant_patterns_867211379c96c05da1f9bebd | 2026-09-27 | zoocode-tiel-coder-35b-A3B-MTP-262144-ctx | quant/snapshot/build conventions

### DECISION
PALACE-REF | DECISION | vai_llama_cpp_turboquant/decisions | drawer drawer_vai_llama_cpp_turboquant_decisions_50046beb11098e7e931b2e88 | 2026-09-27 | zoocode-tiel-coder-35b-A3B-MTP-262144-ctx | ROCm preset name corrected to rocm10

### ISSUE
PALACE-REF | ISSUE | vai_llama_cpp_turboquant/issues | drawer drawer_vai_llama_cpp_turboquant_issues_796e5d5bc38a50ac699ccbcd | 2026-09-27 | zoocode-tiel-coder-35b-A3B-MTP-262144-ctx | test needs network; branch/preset mismatch

### MANIFEST
PALACE-REF | MANIFEST | vai_llama_cpp_turboquant/manifest | drawer drawer_vai_llama_cpp_turboquant_manifest_fbb402963b48616397ba9d87 | 2026-09-27 | zoocode-tiel-coder-35b-A3B-MTP-262144-ctx | baseline manifest snapshot

### CHANGELOG
PALACE-REF | CHANGELOG | vai_llama_cpp_turboquant/changelog | drawer drawer_vai_llama_cpp_turboquant_changelog_41cd589575f2a9ca52467a29 | 2026-09-27 | zoocode-tiel-coder-35b-A3B-MTP-262144-ctx | changelog of onboarding + AGENTS.md corrections

### HANDOFF
PALACE-REF | HANDOFF | vai_llama_cpp_turboquant/handoff | drawer drawer_vai_llama_cpp_turboquant_handoff_4817f757a8564b23d5be113c | 2026-09-27 | zoocode-tiel-coder-35b-A3B-MTP-262144-ctx | onboarding handoff state

Last reconciled with palace: 2026-09-27 by zoocode-tiel-coder-35b-A3B-MTP-262144-ctx
