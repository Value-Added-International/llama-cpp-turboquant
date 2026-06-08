#!/usr/bin/env python3
"""
test_runner.py — pytest suite for vai-llama-bench-runner.py

Tests are organised into four areas:
  1. UUID generation         — make_run_uuid() returns a valid UUID string
  2. Log filename helpers    — stem/path/finalise construction
  3. Pre-flight checks       — classify() integration for ROCm, CUDA, Vulkan
  4. Orchestrator (dry-run)  — end-to-end run_benchmarks() with a fake binary,
                               verifying log file creation, naming, and skip logic

Run with:
    pytest vai-bench-mark-scripts/test_runner.py -v
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the runner module under test
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_RUNNER_PATH = _SCRIPT_DIR / "vai-llama-bench-runner.py"

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

runner = _load_module("runner", _RUNNER_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_config(log_dir: str, prompts: list, backend_binary: str = "/bin/true",
                    backend_name: str = "rocm") -> dict:
    """Build the smallest valid config dict for a single backend + model."""
    return {
        "bench_params": {
            "kv_types": [{"ctk": "q8_0", "ctv": "q8_0"}],
            "prompts":      prompts,
            "n_gen":        64,
            "n_batch":      max(prompts),
            "flash_attn":   1,
            "kv_offload":   0,
            "repetitions":  1,
        },
        "backends": {
            backend_name: {
                "binary":       backend_binary,
                "enabled":      True,
                "split_mode":   "layer",
                "vram_monitor": "rocm-smi",
            }
        },
        "models": [
            {
                "name":          "test-model",
                "path":          "/dev/null",   # exists, readable, zero bytes
                "architecture":  "gemma4-dense",
                "total_layers":  48,
                "context_max":   8192,
                "ngl_rocm_candidates":   [1],
                "ngl_cuda_candidates":   [1],
                "ngl_vulkan_candidates": [1],
            }
        ],
        "logging": {
            "log_dir":                log_dir,
            "vram_sample_interval_s": 60,   # won't fire during tests
        },
        "watchdog": {
            "timeout_s":  30,
            "max_retries": 0,
        },
    }


# ===========================================================================
# 1. UUID generation
# ===========================================================================

class TestUUID:
    def test_make_run_uuid_returns_string(self):
        uid = runner.make_run_uuid()
        assert isinstance(uid, str)

    def test_make_run_uuid_is_valid_uuid_format(self):
        import uuid
        uid = runner.make_run_uuid()
        parsed = uuid.UUID(uid)   # raises ValueError if invalid
        assert str(parsed) == uid

    def test_make_run_uuid_is_unique(self):
        uids = {runner.make_run_uuid() for _ in range(50)}
        assert len(uids) == 50, "UUIDs should be unique"

    def test_make_run_uuid_version(self):
        """Should be v7 if uuid_v7 is importable, v4 as fallback — both acceptable."""
        import uuid
        uid = runner.make_run_uuid()
        parsed = uuid.UUID(uid)
        assert parsed.version in (4, 7)


# ===========================================================================
# 2. Log filename helpers
# ===========================================================================

class TestLogFilenameHelpers:
    def test_make_log_stem_format(self):
        stem = runner.make_log_stem(
            "01234567-89ab-7cde-f012-3456789abcde",
            "rocm", "gemma-4-12B", "q8_0", "q8_0", "20260607-120000"
        )
        assert stem == (
            "01234567-89ab-7cde-f012-3456789abcde"
            "-ROCM-gemma-4-12B-q8_0-q8_0-20260607-120000"
        )

    def test_make_log_stem_backend_uppercased(self):
        stem = runner.make_log_stem("uuid", "cuda", "model", "f16", "f16", "ts")
        assert "-CUDA-" in stem

    def test_log_path_for_suffix(self, tmp_path):
        log_dir = tmp_path
        stem    = "uuid-ROCM-model-q8_0-q8_0-20260607-120000"
        for status in ("inprogress", "completed", "skipped", "failed"):
            p = runner.log_path_for(log_dir, stem, status)
            assert p.name == f"{stem}-{status}.jsonl"
            assert p.parent == log_dir

    def test_finalise_log_renames_file(self, tmp_path):
        stem = "uuid-ROCM-model-q8_0-q8_0-20260607-120000"
        inprogress = runner.log_path_for(tmp_path, stem, "inprogress")
        inprogress.write_text('{"event":"test"}\n')

        final = runner.finalise_log(tmp_path, stem, "completed", inprogress)

        assert final.exists()
        assert final.name.endswith("-completed.jsonl")
        assert not inprogress.exists()

    def test_finalise_log_failed_suffix(self, tmp_path):
        stem = "uuid-ROCM-model-q8_0-q8_0-20260607-120000"
        inprogress = runner.log_path_for(tmp_path, stem, "inprogress")
        inprogress.write_text('{"event":"test"}\n')

        final = runner.finalise_log(tmp_path, stem, "failed", inprogress)
        assert final.name.endswith("-failed.jsonl")

    def test_finalise_log_no_op_if_not_exists(self, tmp_path):
        stem = "uuid-ROCM-model-q8_0-q8_0-20260607-120000"
        missing = runner.log_path_for(tmp_path, stem, "inprogress")
        # Should not raise even if file doesn't exist
        final = runner.finalise_log(tmp_path, stem, "completed", missing)
        assert final.name.endswith("-completed.jsonl")


# ===========================================================================
# 3. Pre-flight checks
# ===========================================================================

class TestPreflightCheck:
    """
    Tests for preflight_check() — the single function that wraps classify().
    """

    _model_gemma4 = {
        "name":         "gemma-4-12B",
        "architecture": "gemma4-dense",
        "total_layers": 48,
    }

    # ── ROCm: unsafe prompts ────────────────────────────────────────────────

    def test_rocm_unsafe_512(self):
        result = runner.preflight_check(
            backend="rocm", model=self._model_gemma4,
            n_prompt=512, n_ctx=8192,
            type_k="q8_0", type_v="q8_0", flash_attn=1,
        )
        assert result["safe"] is False
        assert result["rule"] == "rocm_kv_max_oob_threshold"
        assert "512" in result["comment"]
        assert result["structured"]["n_prompt"] == 512

    def test_rocm_unsafe_1023(self):
        result = runner.preflight_check(
            backend="rocm", model=self._model_gemma4,
            n_prompt=1023, n_ctx=8192,
            type_k="turbo3", type_v="turbo3", flash_attn=1,
        )
        assert result["safe"] is False

    def test_rocm_unsafe_turbo4(self):
        result = runner.preflight_check(
            backend="rocm", model=self._model_gemma4,
            n_prompt=512, n_ctx=8192,
            type_k="turbo4", type_v="turbo4", flash_attn=1,
        )
        assert result["safe"] is False
        assert result["structured"]["type_k"] == "turbo4"

    # ── ROCm: safe prompts ──────────────────────────────────────────────────

    def test_rocm_safe_1024(self):
        result = runner.preflight_check(
            backend="rocm", model=self._model_gemma4,
            n_prompt=1024, n_ctx=8192,
            type_k="q8_0", type_v="q8_0", flash_attn=1,
        )
        assert result["safe"] is True
        assert result["rule"] == ""

    def test_rocm_safe_2048(self):
        result = runner.preflight_check(
            backend="rocm", model=self._model_gemma4,
            n_prompt=2048, n_ctx=8192,
            type_k="turbo3", type_v="turbo3", flash_attn=1,
        )
        assert result["safe"] is True

    def test_rocm_safe_flash_attn_disabled(self):
        result = runner.preflight_check(
            backend="rocm", model=self._model_gemma4,
            n_prompt=512, n_ctx=8192,
            type_k="q8_0", type_v="q8_0", flash_attn=0,
        )
        assert result["safe"] is True

    # ── CUDA: always safe currently ─────────────────────────────────────────

    def test_cuda_always_safe_small_prompt(self):
        result = runner.preflight_check(
            backend="cuda", model=self._model_gemma4,
            n_prompt=512, n_ctx=8192,
            type_k="q8_0", type_v="q8_0", flash_attn=1,
        )
        assert result["safe"] is True
        assert result["rule"] == ""

    def test_cuda_safe_turbo4(self):
        result = runner.preflight_check(
            backend="cuda", model=self._model_gemma4,
            n_prompt=512, n_ctx=8192,
            type_k="turbo4", type_v="turbo4", flash_attn=1,
        )
        assert result["safe"] is True

    # ── Vulkan: always safe currently ───────────────────────────────────────

    def test_vulkan_always_safe(self):
        result = runner.preflight_check(
            backend="vulkan", model=self._model_gemma4,
            n_prompt=512, n_ctx=8192,
            type_k="q8_0", type_v="q8_0", flash_attn=1,
        )
        assert result["safe"] is True

    # ── Structured fields present ────────────────────────────────────────────

    def test_structured_has_required_keys(self):
        result = runner.preflight_check(
            backend="rocm", model=self._model_gemma4,
            n_prompt=512, n_ctx=8192,
            type_k="q8_0", type_v="q8_0", flash_attn=1,
        )
        for key in ("backend", "n_prompt", "n_ctx", "type_k", "type_v",
                    "flash_attn", "head_dim", "rule"):
            assert key in result["structured"], f"Missing key: {key}"

    def test_comment_is_human_readable_string(self):
        result = runner.preflight_check(
            backend="rocm", model=self._model_gemma4,
            n_prompt=512, n_ctx=8192,
            type_k="q8_0", type_v="q8_0", flash_attn=1,
        )
        assert isinstance(result["comment"], str)
        assert len(result["comment"]) > 10


# ===========================================================================
# 4. Orchestrator — dry-run / skip integration
# ===========================================================================

class TestOrchestratorDryRun:
    """
    End-to-end tests of run_benchmarks() in dry-run mode.
    We use /bin/true as the binary so it exits 0 immediately.
    """

    def _run(self, tmp_path: Path, prompts: list, backend: str = "rocm",
             binary: str = "/bin/true") -> list[Path]:
        """Run the orchestrator and return all .jsonl files created."""
        cfg = _minimal_config(str(tmp_path), prompts,
                              backend_binary=binary, backend_name=backend)
        runner.run_benchmarks(cfg, [backend], ["all"], dry_run=True)
        return list(tmp_path.glob("*.jsonl"))

    # ── Safe config (prompts >= 1024 on ROCm) ───────────────────────────────

    def test_safe_run_creates_completed_log(self, tmp_path):
        logs = self._run(tmp_path, prompts=[1024, 2048], backend="rocm")
        assert len(logs) == 1
        assert logs[0].name.endswith("-completed.jsonl")

    def test_completed_log_has_uuid_and_backend(self, tmp_path):
        logs = self._run(tmp_path, prompts=[1024], backend="rocm")
        name = logs[0].name
        # Format: {uuid}-ROCM-{model}-{ctk}-{ctv}-{ts}-completed.jsonl
        parts = name.split("-")
        assert "ROCM" in parts
        assert name.endswith("-completed.jsonl")

    def test_completed_log_contains_run_start_event(self, tmp_path):
        logs = self._run(tmp_path, prompts=[1024], backend="rocm")
        events = [json.loads(l) for l in logs[0].read_text().splitlines()
                  if l.strip().startswith("{")]
        event_names = [e.get("event") for e in events]
        assert "run_start" in event_names

    def test_completed_log_contains_preflight_passed_event(self, tmp_path):
        logs = self._run(tmp_path, prompts=[1024], backend="rocm")
        events = [json.loads(l) for l in logs[0].read_text().splitlines()
                  if l.strip().startswith("{")]
        event_names = [e.get("event") for e in events]
        assert "preflight_passed" in event_names

    def test_completed_log_contains_run_end_event(self, tmp_path):
        logs = self._run(tmp_path, prompts=[1024], backend="rocm")
        events = [json.loads(l) for l in logs[0].read_text().splitlines()
                  if l.strip().startswith("{")]
        run_end = [e for e in events if e.get("event") == "run_end"]
        assert len(run_end) == 1
        assert run_end[0]["status"] == "completed"

    # ── Unsafe config (512 in prompts on ROCm) ──────────────────────────────

    def test_unsafe_run_creates_skipped_log(self, tmp_path):
        logs = self._run(tmp_path, prompts=[512, 1024], backend="rocm")
        assert len(logs) == 1
        assert logs[0].name.endswith("-skipped.jsonl")

    def test_skipped_log_has_uuid_and_backend(self, tmp_path):
        logs = self._run(tmp_path, prompts=[512], backend="rocm")
        name = logs[0].name
        assert "ROCM" in name
        assert name.endswith("-skipped.jsonl")

    def test_skipped_log_contains_run_skipped_event(self, tmp_path):
        logs = self._run(tmp_path, prompts=[512], backend="rocm")
        events = [json.loads(l) for l in logs[0].read_text().splitlines()
                  if l.strip().startswith("{")]
        event_names = [e.get("event") for e in events]
        assert "run_skipped" in event_names

    def test_skipped_event_has_comment_string(self, tmp_path):
        logs = self._run(tmp_path, prompts=[512], backend="rocm")
        events = [json.loads(l) for l in logs[0].read_text().splitlines()
                  if l.strip().startswith("{")]
        skip_ev = next(e for e in events if e.get("event") == "run_skipped")
        assert isinstance(skip_ev.get("comment"), str)
        assert len(skip_ev["comment"]) > 10

    def test_skipped_event_has_structured_dict(self, tmp_path):
        logs = self._run(tmp_path, prompts=[512], backend="rocm")
        events = [json.loads(l) for l in logs[0].read_text().splitlines()
                  if l.strip().startswith("{")]
        skip_ev = next(e for e in events if e.get("event") == "run_skipped")
        s = skip_ev.get("structured", {})
        assert isinstance(s, dict)
        assert "unsafe_prompts" in s
        assert len(s["unsafe_prompts"]) >= 1
        up = s["unsafe_prompts"][0]
        assert "n_prompt" in up
        assert "rule" in up
        assert "comment" in up

    def test_skipped_event_unsafe_prompt_has_rule(self, tmp_path):
        logs = self._run(tmp_path, prompts=[512], backend="rocm")
        events = [json.loads(l) for l in logs[0].read_text().splitlines()
                  if l.strip().startswith("{")]
        skip_ev = next(e for e in events if e.get("event") == "run_skipped")
        rule = skip_ev["structured"]["unsafe_prompts"][0]["rule"]
        assert rule == "rocm_kv_max_oob_threshold"

    def test_skipped_event_has_run_uuid(self, tmp_path):
        logs = self._run(tmp_path, prompts=[512], backend="rocm")
        events = [json.loads(l) for l in logs[0].read_text().splitlines()
                  if l.strip().startswith("{")]
        skip_ev = next(e for e in events if e.get("event") == "run_skipped")
        assert "run_uuid" in skip_ev

    def test_no_inprogress_files_left(self, tmp_path):
        self._run(tmp_path, prompts=[1024], backend="rocm")
        inprogress = list(tmp_path.glob("*-inprogress.jsonl"))
        assert len(inprogress) == 0, "No inprogress files should remain"

    # ── CUDA: 512 is safe, run proceeds ─────────────────────────────────────

    def test_cuda_512_is_safe_run_proceeds(self, tmp_path):
        cfg = _minimal_config(str(tmp_path), [512, 1024],
                              backend_binary="/bin/true", backend_name="cuda")
        # Patch vram_monitor to something that won't fail
        cfg["backends"]["cuda"]["vram_monitor"] = "rocm-smi"
        runner.run_benchmarks(cfg, ["cuda"], ["all"], dry_run=True)
        logs = list(tmp_path.glob("*.jsonl"))
        assert len(logs) == 1
        assert logs[0].name.endswith("-completed.jsonl")
        assert "CUDA" in logs[0].name

    # ── Vulkan: 512 is safe, run proceeds ───────────────────────────────────

    def test_vulkan_512_is_safe_run_proceeds(self, tmp_path):
        cfg = _minimal_config(str(tmp_path), [512],
                              backend_binary="/bin/true", backend_name="vulkan")
        cfg["backends"]["vulkan"]["vram_monitor"] = "rocm-smi"
        runner.run_benchmarks(cfg, ["vulkan"], ["all"], dry_run=True)
        logs = list(tmp_path.glob("*.jsonl"))
        assert len(logs) == 1
        assert logs[0].name.endswith("-completed.jsonl")
        assert "VULKAN" in logs[0].name

    # ── Mixed: safe + unsafe prompts in same list on ROCm ───────────────────

    def test_mixed_prompts_any_unsafe_skips_run(self, tmp_path):
        # 512 is unsafe → whole run skipped even though 2048 would be safe
        logs = self._run(tmp_path, prompts=[512, 2048], backend="rocm")
        assert len(logs) == 1
        assert logs[0].name.endswith("-skipped.jsonl")

    # ── Log filename contains all expected components ────────────────────────

    def test_completed_filename_components(self, tmp_path):
        logs = self._run(tmp_path, prompts=[1024], backend="rocm")
        name = logs[0].name
        assert "ROCM"        in name
        assert "test-model"  in name
        assert "q8_0"        in name
        assert "completed"   in name
        assert name.endswith(".jsonl")

    def test_skipped_filename_components(self, tmp_path):
        logs = self._run(tmp_path, prompts=[512], backend="rocm")
        name = logs[0].name
        assert "ROCM"       in name
        assert "test-model" in name
        assert "q8_0"       in name
        assert "skipped"    in name
        assert name.endswith(".jsonl")
