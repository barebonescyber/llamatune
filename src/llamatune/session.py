"""Evidence, session layout, and resume (DESIGN §12, §13.2).

Only this module writes inside a session directory, and every write is
path-confined to reject traversal or symlinks that would escape it.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import secrets
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llamatune._version import __version__
from llamatune.types import GPUInfo, HardwareReport, LlamaCppReport, ModelReport, TuneOptions

_SCHEMA_VERSION = 2
_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")
_CREATE_RETRIES = 5
_DERIVED_OUTPUTS = ("analysis.json", "recommended.json", "recommended.sh", "report.md")


class SessionPathError(Exception):
    """Raised when a resolved path would escape the session directory."""


class SessionCorruptionError(Exception):
    """Raised when journal.jsonl contains corruption resume cannot tolerate."""


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _sanitize_stem(stem: str) -> str:
    cleaned = _SAFE_STEM_RE.sub("_", stem).strip("._")
    return cleaned or "model"


def _confine(base: Path, *parts: str) -> Path:
    """Join `parts` onto `base`, rejecting any result that escapes `base`."""
    base_resolved = base.resolve()
    candidate = base
    for part in parts:
        candidate = candidate / part
    candidate_resolved = candidate.resolve()
    try:
        candidate_resolved.relative_to(base_resolved)
    except ValueError:
        msg = f"path {candidate} escapes session directory {base}"
        raise SessionPathError(msg) from None
    return candidate


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


# --------------------------------------------------------------------------
# Serialization: dataclasses <-> JSON-safe dicts
# --------------------------------------------------------------------------


def _gpu_to_dict(gpu: GPUInfo) -> dict[str, Any]:
    return {
        "vendor": gpu.vendor,
        "name": gpu.name,
        "vram_mb": gpu.vram_mb,
        "method": gpu.method,
        "vram_free_mb": gpu.vram_free_mb,
        "vram_free_method": gpu.vram_free_method,
        "vram_free_at": gpu.vram_free_at,
        "driver_version": gpu.driver_version,
    }


def _gpu_from_dict(data: Mapping[str, Any]) -> GPUInfo:
    return GPUInfo(
        vendor=data["vendor"],
        name=data["name"],
        vram_mb=data["vram_mb"],
        method=data["method"],
        vram_free_mb=data.get("vram_free_mb"),
        vram_free_method=data.get("vram_free_method"),
        vram_free_at=data.get("vram_free_at"),
        driver_version=data.get("driver_version"),
    )


def _hardware_to_dict(hardware: HardwareReport) -> dict[str, Any]:
    return {
        "os_name": hardware.os_name,
        "arch": hardware.arch,
        "cpu_model": hardware.cpu_model,
        "physical_cores": hardware.physical_cores,
        "logical_cores": hardware.logical_cores,
        "perf_cores": hardware.perf_cores,
        "ram_mb": hardware.ram_mb,
        "gpus": [_gpu_to_dict(g) for g in hardware.gpus],
        "warnings": list(hardware.warnings),
    }


def _hardware_from_dict(data: Mapping[str, Any]) -> HardwareReport:
    return HardwareReport(
        os_name=data["os_name"],
        arch=data["arch"],
        cpu_model=data["cpu_model"],
        physical_cores=data["physical_cores"],
        logical_cores=data["logical_cores"],
        perf_cores=data["perf_cores"],
        ram_mb=data["ram_mb"],
        gpus=tuple(_gpu_from_dict(g) for g in data["gpus"]),
        warnings=tuple(data["warnings"]),
    )


def _model_to_dict(model: ModelReport) -> dict[str, Any]:
    return {
        "path": str(model.path),
        "size_bytes": model.size_bytes,
        "architecture": model.architecture,
        "n_layer": model.n_layer,
        "ngl_all": model.ngl_all,
        "expert_count": model.expert_count,
        "moe": model.moe,
        "name": model.name,
        "fingerprint": model.fingerprint,
        "full_sha256": model.full_sha256,
        "expert_bytes": model.expert_bytes,
        "dense_bytes": model.dense_bytes,
        "kv_bytes_per_token_f16": model.kv_bytes_per_token_f16,
        "n_kv_layers": model.n_kv_layers,
    }


def _model_from_dict(data: Mapping[str, Any]) -> ModelReport:
    return ModelReport(
        path=Path(data["path"]),
        size_bytes=data["size_bytes"],
        architecture=data["architecture"],
        n_layer=data["n_layer"],
        ngl_all=data["ngl_all"],
        expert_count=data["expert_count"],
        moe=data["moe"],
        name=data["name"],
        fingerprint=data["fingerprint"],
        full_sha256=data["full_sha256"],
        expert_bytes=data.get("expert_bytes"),
        dense_bytes=data.get("dense_bytes"),
        kv_bytes_per_token_f16=data.get("kv_bytes_per_token_f16"),
        n_kv_layers=data.get("n_kv_layers"),
    )


def _llama_to_dict(llama: LlamaCppReport) -> dict[str, Any]:
    return {
        "bench_path": str(llama.bench_path),
        "cli_path": str(llama.cli_path) if llama.cli_path is not None else None,
        "server_path": str(llama.server_path) if llama.server_path is not None else None,
        "capabilities": sorted(llama.capabilities),
        "help_sha256": llama.help_sha256,
        "build_commit": llama.build_commit,
        "build_number": llama.build_number,
        "backends": llama.backends,
        "bench_sha256": llama.bench_sha256,
        "perplexity_path": (
            str(llama.perplexity_path) if llama.perplexity_path is not None else None
        ),
    }


def _llama_from_dict(data: Mapping[str, Any]) -> LlamaCppReport:
    return LlamaCppReport(
        bench_path=Path(data["bench_path"]),
        cli_path=Path(data["cli_path"]) if data["cli_path"] is not None else None,
        server_path=Path(data["server_path"]) if data["server_path"] is not None else None,
        capabilities=frozenset(data["capabilities"]),
        help_sha256=data["help_sha256"],
        build_commit=data["build_commit"],
        build_number=data["build_number"],
        backends=data.get("backends"),
        perplexity_path=(
            Path(data["perplexity_path"]) if data.get("perplexity_path") is not None else None
        ),
        bench_sha256=data.get("bench_sha256"),
    )


def _options_to_dict(options: TuneOptions) -> dict[str, Any]:
    return {
        "target": options.target,
        "budget_trials": options.budget_trials,
        "budget_minutes": options.budget_minutes,
        "reps_search": options.reps_search,
        "reps_confirm": options.reps_confirm,
        "baseline_runs": options.baseline_runs,
        "pp": options.pp,
        "tg": options.tg,
        "allow_lossy": options.allow_lossy,
        "cooldown_s": options.cooldown_s,
        "baseline_only": options.baseline_only,
        "llama_bin": str(options.llama_bin) if options.llama_bin is not None else None,
        "sessions_dir": str(options.sessions_dir),
        "full_hash": options.full_hash,
        "ctx_size": options.ctx_size,
        "ctx_ladder": list(options.ctx_ladder),
        "vram_reserve_mb": options.vram_reserve_mb,
        "initial_gpu_layers": options.initial_gpu_layers,
        "max_gpu_layers": options.max_gpu_layers,
        "initial_cpu_moe": options.initial_cpu_moe,
        "allow_core_dumps": options.allow_core_dumps,
        "quiet_wait_s": options.quiet_wait_s,
        "quiet_load": options.quiet_load,
        "observe_vram": options.observe_vram,
        "validate_with_cli": options.validate_with_cli,
        "quality_corpus": (
            str(options.quality_corpus) if options.quality_corpus is not None else None
        ),
        "batched_trials": options.batched_trials,
        "ot_search": options.ot_search,
        "depth": options.depth,
        "depth_profile": list(options.depth_profile) if options.depth_profile is not None else None,
        "thermal_threshold_c": options.thermal_threshold_c,
        "thermal_wait_cap_s": options.thermal_wait_cap_s,
        "multi_gpu": options.multi_gpu,
    }


def _options_from_dict(data: Mapping[str, Any]) -> TuneOptions:
    return TuneOptions(
        target=data["target"],
        budget_trials=data["budget_trials"],
        budget_minutes=data["budget_minutes"],
        reps_search=data["reps_search"],
        reps_confirm=data["reps_confirm"],
        baseline_runs=data["baseline_runs"],
        pp=data["pp"],
        tg=data["tg"],
        allow_lossy=data["allow_lossy"],
        cooldown_s=data["cooldown_s"],
        baseline_only=data["baseline_only"],
        llama_bin=Path(data["llama_bin"]) if data["llama_bin"] is not None else None,
        sessions_dir=Path(data["sessions_dir"]),
        full_hash=data["full_hash"],
        ctx_size=data.get("ctx_size"),
        ctx_ladder=tuple(int(value) for value in data.get("ctx_ladder", ())),
        vram_reserve_mb=data.get("vram_reserve_mb"),
        initial_gpu_layers=data.get("initial_gpu_layers"),
        max_gpu_layers=data.get("max_gpu_layers"),
        initial_cpu_moe=data.get("initial_cpu_moe"),
        allow_core_dumps=bool(data.get("allow_core_dumps", False)),
        quiet_wait_s=float(data.get("quiet_wait_s", 0.0)),
        quiet_load=(float(data["quiet_load"]) if data.get("quiet_load") is not None else None),
        observe_vram=bool(data.get("observe_vram", True)),
        validate_with_cli=bool(data.get("validate_with_cli", False)),
        quality_corpus=(
            Path(data["quality_corpus"]) if data.get("quality_corpus") is not None else None
        ),
        batched_trials=bool(data.get("batched_trials", True)),
        ot_search=bool(data.get("ot_search", False)),
        depth=(int(data["depth"]) if data.get("depth") is not None else None),
        depth_profile=(
            tuple(int(value) for value in data["depth_profile"])
            if data.get("depth_profile") is not None
            else None
        ),
        thermal_threshold_c=float(data.get("thermal_threshold_c", 75.0)),
        thermal_wait_cap_s=float(data.get("thermal_wait_cap_s", 60.0)),
        multi_gpu=bool(data.get("multi_gpu", False)),
    )


def list_sessions(sessions_dir: Path) -> list[dict[str, Any]]:
    """Summarize session directories, tolerating incomplete and corrupt entries."""
    if not sessions_dir.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for candidate in sorted(path for path in sessions_dir.iterdir() if path.is_dir()):
        if not (candidate / "session.json").is_file():
            if (candidate / "results-matrix.json").is_file():
                continue
            if candidate.name == "nightshift" and any(
                child.is_dir()
                and (child / "run.json").is_file()
                and (child / "nightshift.json").is_file()
                for child in candidate.iterdir()
            ):
                continue
        row: dict[str, Any] = {
            "session_dir": str(candidate),
            "model": None,
            "created": None,
            "exit_code": None,
            "winner_trial_id": None,
            "confirmed": False,
            "status": "in_progress",
        }
        try:
            meta = _read_json(candidate / "session.json")
            model = _read_json(candidate / "model.json")
            row["model"] = model.get("name") or Path(str(model.get("path", ""))).name
            row["created"] = meta.get("created")
            analysis_path = candidate / "analysis.json"
            if analysis_path.is_file():
                analysis = _read_json(analysis_path)
                winner = analysis.get("winner")
                if isinstance(winner, dict):
                    row["winner_trial_id"] = winner.get("trial_id")
                    row["confirmed"] = bool(winner.get("confirmed", False))
                endings: list[dict[str, Any]] = []
                journal_path = candidate / "journal.jsonl"
                if journal_path.is_file():
                    for line in journal_path.read_text(encoding="utf-8").splitlines():
                        entry = json.loads(line)
                        if isinstance(entry, dict) and entry.get("type") == "session_end":
                            endings.append(entry)
                row["exit_code"] = endings[-1].get("exit_code") if endings else None
                row["status"] = "complete"
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            row["status"] = "corrupt"
        rows.append(row)
    return rows


class Session:
    """A tuning session's evidence directory (DESIGN §12).

    Construct via :meth:`create` (new session) or :meth:`load` (existing
    session directory); the `__init__` signature is an implementation
    detail.
    """

    def __init__(
        self,
        *,
        session_dir: Path,
        model: ModelReport,
        hardware: HardwareReport,
        llama: LlamaCppReport,
        options: TuneOptions,
        entries: list[dict[str, Any]],
        resume_warnings: tuple[str, ...] = (),
    ) -> None:
        self._dir = session_dir
        self._model = model
        self._hardware = hardware
        self._llama = llama
        self._options = options
        self._entries = entries
        self._resume_warnings = resume_warnings

    # -- properties ---------------------------------------------------

    @property
    def dir(self) -> Path:
        return self._dir

    @property
    def model(self) -> ModelReport:
        return self._model

    @property
    def hardware(self) -> HardwareReport:
        return self._hardware

    @property
    def llama(self) -> LlamaCppReport:
        return self._llama

    @property
    def options(self) -> TuneOptions:
        return self._options

    @property
    def entries(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._entries)

    @property
    def journaled_trial_ids(self) -> frozenset[str]:
        return frozenset(
            e["trial_id"] for e in self._entries if e.get("type") == "trial" and "trial_id" in e
        )

    @property
    def baseline_runs_completed(self) -> int:
        return sum(1 for e in self._entries if e.get("type") == "baseline_run")

    @property
    def resume_warnings(self) -> tuple[str, ...]:
        """Warnings recorded while loading an existing session (e.g. a
        torn final journal line that was discarded)."""
        return self._resume_warnings

    # -- construction ---------------------------------------------------

    @classmethod
    def create(
        cls,
        sessions_root: Path,
        *,
        model: ModelReport,
        hardware: HardwareReport,
        llama: LlamaCppReport,
        options: TuneOptions,
        argv: Sequence[str],
    ) -> Session:
        sessions_root.mkdir(parents=True, exist_ok=True)
        stem = _sanitize_stem(model.path.stem)

        session_dir: Path | None = None
        last_error: OSError | None = None
        for _ in range(_CREATE_RETRIES):
            timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            suffix = secrets.token_hex(3)
            candidate = sessions_root / f"{stem}-{timestamp}-{suffix}"
            try:
                candidate.mkdir(parents=True, exist_ok=False)
            except FileExistsError as exc:
                last_error = exc
                continue
            session_dir = candidate
            break
        if session_dir is None:
            msg = f"could not allocate a unique session directory under {sessions_root}"
            raise SessionPathError(msg) from last_error

        (session_dir / "baseline").mkdir(exist_ok=True)
        (session_dir / "trials").mkdir(exist_ok=True)

        session = cls(
            session_dir=session_dir,
            model=model,
            hardware=hardware,
            llama=llama,
            options=options,
            entries=[],
        )

        session._write_json(
            "session.json",
            {
                "schema_version": _SCHEMA_VERSION,
                "tool_version": __version__,
                "argv": list(argv),
                "options": _options_to_dict(options),
                "created": _utc_now_iso(),
            },
        )
        session._write_json("hardware.json", _hardware_to_dict(hardware))
        session._write_json("model.json", _model_to_dict(model))
        session._write_json("llamacpp.json", _llama_to_dict(llama))

        session.append(
            {
                "type": "session_start",
                "tool_version": __version__,
                "argv": list(argv),
                "model_fingerprint": model.fingerprint,
                "llama_help_sha256": llama.help_sha256,
            }
        )

        return session

    @classmethod
    def load(cls, session_dir: Path) -> Session:
        session_dir = Path(session_dir)
        # Confine every fixed artifact before reading it so a traversal
        # component or an out-of-tree symlink cannot make resume read (or,
        # for the journal, truncate) a file outside the session directory.
        session_meta = _read_json(_confine(session_dir, "session.json"))
        schema_version = session_meta.get("schema_version", 1)
        if not isinstance(schema_version, int):
            raise SessionCorruptionError("session.json schema_version must be an integer")
        if schema_version > _SCHEMA_VERSION:
            raise SessionCorruptionError(
                f"session schema version {schema_version} is newer than supported version "
                f"{_SCHEMA_VERSION}"
            )
        hardware = _hardware_from_dict(_read_json(_confine(session_dir, "hardware.json")))
        model = _model_from_dict(_read_json(_confine(session_dir, "model.json")))
        llama = _llama_from_dict(_read_json(_confine(session_dir, "llamacpp.json")))
        options = _options_from_dict(session_meta["options"])

        entries, resume_warnings = cls._read_journal(_confine(session_dir, "journal.jsonl"))

        return cls(
            session_dir=session_dir,
            model=model,
            hardware=hardware,
            llama=llama,
            options=options,
            entries=entries,
            resume_warnings=tuple(resume_warnings),
        )

    @staticmethod
    def _read_journal(journal_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
        entries: list[dict[str, Any]] = []
        warnings: list[str] = []
        if not journal_path.is_file():
            return entries, warnings

        raw = journal_path.read_bytes()
        raw_lines = raw.decode("utf-8", errors="replace").splitlines()
        last_index = len(raw_lines) - 1
        for index, line in enumerate(raw_lines):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                if index == last_index:
                    warnings.append("journal.jsonl: discarded a torn final line on resume")
                    # Truncate the torn tail from disk as well: a later
                    # append would otherwise merge with the torn fragment
                    # and corrupt the journal for every future load.
                    cut = raw.rfind(b"\n") + 1
                    if cut >= len(raw):  # corrupt final line is newline-terminated
                        cut = raw.rfind(b"\n", 0, len(raw) - 1) + 1
                    with journal_path.open("r+b") as fh:
                        fh.truncate(cut)
                    continue
                msg = f"journal.jsonl line {index + 1} is corrupt and is not the final line"
                raise SessionCorruptionError(msg) from None
            if not isinstance(entry, dict):
                msg = f"journal.jsonl line {index + 1} is corrupt: expected a JSON object"
                raise SessionCorruptionError(msg)
            entries.append(entry)
        return entries, warnings

    # -- writes -----------------------------------------------------------

    def _write_json(self, name: str, data: dict[str, Any]) -> None:
        path = _confine(self._dir, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")

    def append(self, entry: dict[str, Any]) -> None:
        """Append one journal entry; adds a timestamp, flushes, and fsyncs."""
        record = dict(entry)
        record.setdefault("ts", _utc_now_iso())
        line = json.dumps(record, sort_keys=True)
        journal_path = _confine(self._dir, "journal.jsonl")
        with journal_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._entries.append(record)

    def trial_dir(self, trial_id: str) -> Path:
        path = _confine(self._dir, "trials", trial_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def baseline_dir(self, n: int) -> Path:
        path = _confine(self._dir, "baseline", f"run-{n}")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def probe_dir(self, probe_id: str) -> Path:
        """Return the confined artifact directory for a feasibility probe."""
        path = _confine(self._dir, "probes", probe_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def batch_dir(self, batch_id: str) -> Path:
        """Create and return a confined artifact directory for a batched trial."""
        path = _confine(self._dir, "batches", batch_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_analysis(self, analysis: dict[str, Any]) -> None:
        self._write_json("analysis.json", analysis)

    def read_json(self, name: str) -> dict[str, Any]:
        """Read one JSON artifact after confining it to this session."""
        return _read_json(_confine(self._dir, name))

    def write_json(self, name: str, data: dict[str, Any]) -> None:
        """Write one JSON artifact after confining it to this session."""
        self._write_json(name, data)

    def write_text(self, name: str, text: str) -> None:
        path = _confine(self._dir, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def invalidate_derived_outputs(self) -> None:
        """Remove stale generated outputs while preserving their journal evidence."""
        for name in _DERIVED_OUTPUTS:
            path = self._dir / name
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.exists():
                raise OSError(f"derived output is not a regular file: {path}")

    def record_build_info(
        self, commit: str | None, number: int | None, backends: str | None
    ) -> None:
        """Append llama.cpp build and backend identity to llamacpp.json (DESIGN §6)."""
        self._llama = dataclasses.replace(
            self._llama, build_commit=commit, build_number=number, backends=backends
        )
        self._write_json("llamacpp.json", _llama_to_dict(self._llama))
