"""Focused unit tests for orchestrator decomposition helpers (#20).

Covers the pure scheduling helpers extracted from ``run_nightshift`` and
``run_marathon``, the deferred-attribute placeholders on ``search._Engine``,
and control-character stripping at the orchestrator announcer boundaries.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

from llamatune import marathon, nightshift, quality, search, ui
from llamatune.session import Session
from llamatune.types import (
    GPUInfo,
    HardwareReport,
    LlamaCppReport,
    ModelReport,
    ProgressEvent,
    TuneOptions,
    WorkItem,
)


def _work_item(**overrides: Any) -> WorkItem:
    values: dict[str, Any] = {
        "kind": "tune",
        "model_path": Path("models/m.gguf"),
        "fingerprint": "f" * 16,
        "session_dir": None,
        "reference_fingerprint": None,
        "estimated_minutes": None,
        "reason": "test",
    }
    values.update(overrides)
    return WorkItem(**values)


def _queues() -> tuple[dict[str, list[WorkItem]], dict[str, int]]:
    queues: dict[str, list[WorkItem]] = {
        "resume": [],
        "tune": [],
        "calibrate": [],
        "retune": [],
    }
    return queues, {}


def _exit_code(
    *,
    stop: bool = False,
    second_signal: bool = False,
    count: int = 0,
    items: list[dict[str, Any]] | None = None,
    queues: dict[str, list[WorkItem]] | None = None,
    consumed: dict[str, int] | None = None,
    failed: bool = False,
) -> int:
    if queues is None:
        queues, consumed = _queues()
    assert consumed is not None
    return nightshift._nightshift_exit_code(
        stop_requested=stop,
        second_signal=second_signal,
        signal_count=count,
        items=items or [],
        phase_queues=queues,
        consumed=consumed,
        failed=failed,
    )


class TestNightshiftExitCode:
    def test_clean_shift_exits_zero(self) -> None:
        assert _exit_code() == 0

    def test_failure_without_interruption_exits_one(self) -> None:
        assert _exit_code(failed=True) == 1

    def test_second_signal_exits_four(self) -> None:
        assert _exit_code(stop=True, second_signal=True, failed=True) == 4

    def test_interrupted_item_exits_four(self) -> None:
        items = [{"outcome": "interrupted"}]
        assert _exit_code(stop=True, count=1, items=items) == 4

    def test_signals_with_unconsumed_plan_exits_four(self) -> None:
        queues, consumed = _queues()
        queues["tune"] = [_work_item()]
        assert _exit_code(stop=True, count=1, queues=queues, consumed=consumed) == 4

    def test_signals_without_unconsumed_plan_do_not_exit_four(self) -> None:
        queues, consumed = _queues()
        queues["tune"] = [_work_item()]
        consumed["tune"] = 1
        assert _exit_code(stop=True, count=1, queues=queues, consumed=consumed) == 0

    def test_stop_alone_does_not_exit_four(self) -> None:
        assert _exit_code(stop=True) == 0


class TestRestoreChampion:
    @staticmethod
    def _config(gpu_layers: int) -> dict[str, Any]:
        return {
            "gpu_layers": gpu_layers,
            "moe_cpu_layers": 0,
            "flash_attn": False,
            "ubatch": 512,
            "batch": 2048,
            "threads": 4,
            "mmap": True,
            "no_kv_offload": False,
            "cache_type_k": "f16",
            "cache_type_v": "f16",
        }

    def test_picks_last_changed_round(self) -> None:
        rounds: list[dict[str, Any]] = [
            {"champion_changed": False},
            {
                "champion_changed": True,
                "winner_config": self._config(10),
                "session_dir": "runs/a",
            },
            {
                "champion_changed": True,
                "winner_config": self._config(33),
                "session_dir": "runs/b",
            },
        ]
        champion, session = marathon._restore_champion(rounds)
        assert champion is not None
        assert champion.gpu_layers == 33
        assert session == Path("runs/b")

    def test_none_when_no_round_changed_champion(self) -> None:
        rounds = [{"champion_changed": False, "winner_config": {"gpu_layers": 1}}]
        assert marathon._restore_champion(rounds) == (None, None)

    def test_none_without_rounds(self) -> None:
        assert marathon._restore_champion([]) == (None, None)


class TestMarathonDefaultConfig:
    def test_fallback_uses_all_cpu_defaults(self) -> None:
        hardware = HardwareReport(
            os_name="Linux",
            arch="x86_64",
            cpu_model="c",
            physical_cores=8,
            logical_cores=8,
            perf_cores=None,
            ram_mb=1024,
            gpus=(),
            warnings=(),
        )
        config = marathon._fallback_default_config(hardware)
        assert config.gpu_layers == 0
        assert config.moe_cpu_layers == 0
        assert config.threads == 8

    def test_fallback_threads_at_least_one(self) -> None:
        hardware = HardwareReport(
            os_name="Linux",
            arch="x86_64",
            cpu_model="c",
            physical_cores=0,
            logical_cores=0,
            perf_cores=None,
            ram_mb=1024,
            gpus=(),
            warnings=(),
        )
        assert marathon._fallback_default_config(hardware).threads == 1

    def test_reference_default_config_wins_when_present(self) -> None:
        reference = {
            "default_config": {
                "gpu_layers": 12,
                "moe_cpu_layers": 3,
                "flash_attn": True,
                "ubatch": 256,
                "batch": 1024,
                "threads": 4,
                "mmap": False,
                "no_kv_offload": True,
                "cache_type_k": "q8_0",
                "cache_type_v": "f16",
            }
        }
        hardware = HardwareReport(
            os_name="Linux",
            arch="x86_64",
            cpu_model="c",
            physical_cores=8,
            logical_cores=8,
            perf_cores=None,
            ram_mb=1024,
            gpus=(),
            warnings=(),
        )
        config = marathon._initial_default_config(reference, hardware)
        assert config.gpu_layers == 12
        assert config.cache_type_k == "q8_0"


class TestEngineDeferredAttributes:
    @pytest.fixture()
    def engine(self, tmp_path: Path) -> search._Engine:
        model = ModelReport(
            path=tmp_path / "m.gguf",
            size_bytes=10,
            architecture="llama",
            n_layer=2,
            ngl_all=4,
            expert_count=0,
            moe=False,
            name="m",
            fingerprint="ab" * 16,
            full_sha256=None,
        )
        llama = LlamaCppReport(
            bench_path=tmp_path / "llama-bench",
            cli_path=None,
            server_path=None,
            capabilities=frozenset(),
            help_sha256="h" * 64,
            build_commit=None,
            build_number=None,
        )
        hardware = HardwareReport(
            os_name="Linux",
            arch="x86_64",
            cpu_model="c",
            physical_cores=4,
            logical_cores=4,
            perf_cores=None,
            ram_mb=1024,
            gpus=(GPUInfo(vendor="test", name="g", vram_mb=100, method="test"),),
            warnings=(),
        )
        options = TuneOptions(
            target="balanced",
            budget_trials=2,
            budget_minutes=None,
            reps_search=1,
            reps_confirm=1,
            baseline_runs=1,
            pp=512,
            tg=128,
            allow_lossy=False,
            cooldown_s=0.0,
            baseline_only=False,
            llama_bin=tmp_path / "llama-bench",
            sessions_dir=tmp_path / "sessions",
            full_hash=False,
        )
        session = Session.create(
            tmp_path / "sessions",
            model=model,
            hardware=hardware,
            llama=llama,
            options=options,
            argv=["llamatune", "tune"],
        )
        return search._Engine(session, hardware, model, llama, options)

    def test_attributes_exist_before_baseline(self, engine: search._Engine) -> None:
        assert engine.baseline.kind == "unestablished"
        assert engine.baseline.runs == 0
        assert engine.default_config.gpu_layers == 0
        assert engine.incumbent_config.gpu_layers == 0

    def test_placeholders_are_inert_but_distinct_from_defaults(
        self, engine: search._Engine
    ) -> None:
        assert engine.measured_default is engine.default_config
        assert engine.baseline.kind != "defaults"

    def test_complete_coverage_skips_before_establishment(self, engine: search._Engine) -> None:
        engine._complete_coverage()
        assert engine.coverage == {}


class _CapturingReporter:
    """Reporter double that records emitted progress events."""

    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []
        self.err = io.StringIO()

    def emit(self, event: ProgressEvent) -> None:
        self.events.append(event)


HOSTILE_NAME = "\x1b[31mho\x07st\x1b(B\x08evil\u202emodel"


class TestAnnouncerSanitization:
    def test_nightshift_announcer_strips_terminal_control_characters(self) -> None:
        err = io.StringIO()
        reporter = ui.PlainReporter(err)
        item = _work_item(model_path=Path(f"{HOSTILE_NAME}.gguf"))
        nightshift._announce_item(reporter, 1, 2, item)
        out = err.getvalue()
        assert "\x1b" not in out
        assert "\x07" not in out
        assert "\x08" not in out
        assert "\u202e" not in out
        assert "[nightshift] item 1/2 tune" in out

    def test_nightshift_announcer_payload_is_sanitized(self) -> None:
        reporter = _CapturingReporter()
        item = _work_item(fingerprint=f"{HOSTILE_NAME}01234abcd")
        nightshift._announce_item(reporter, 1, 1, item)
        payload = reporter.events[0].payload
        message = str(payload["message"])
        assert "\x1b" not in message
        assert "\u202e" not in message

    def test_nightshift_announcer_falls_back_to_sanitized_fingerprint(self) -> None:
        err = io.StringIO()
        item = _work_item(model_path=None, fingerprint="\x1baaaabbbbccccddd")
        nightshift._announce_item(ui.PlainReporter(err), 1, 1, item)
        out = err.getvalue()
        assert "\x1b" not in out
        assert "aaaabbbbccccddd" in out

    def test_marathon_announcer_strips_terminal_control_characters(self) -> None:
        err = io.StringIO()
        marathon._announce(ui.PlainReporter(err), "[marathon] phase re\x1b[31mcon\x07", {})
        out = err.getvalue()
        assert "\x1b" not in out
        assert "\x07" not in out
        assert "[marathon] phase re[31mcon" in out

    def test_quality_task_announcer_strips_suite_and_task_names(self) -> None:
        progress = quality._TaskProgress(total=3)
        err = io.StringIO()
        progress.announce(ui.PlainReporter(err), "evaluated", "suite\x1b[31m", "task\x07id")
        out = err.getvalue()
        assert "\x1b" not in out
        assert "\x07" not in out
        assert "[quality] evaluated task 1/3 taskid (suite[31m)" in out

    def test_quality_progress_counts_each_task_once(self) -> None:
        progress = quality._TaskProgress(total=2)
        reporter = _CapturingReporter()
        progress.announce(reporter, "evaluated", "s", "t")
        progress.announce(reporter, "evaluated", "s", "t")
        assert len(reporter.events) == 1
        assert progress.announced == 1
