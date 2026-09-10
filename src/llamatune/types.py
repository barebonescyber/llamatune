"""Frozen dataclasses shared by all llamatune modules.

This module imports nothing above the standard library (see DESIGN.md
§13 layering rules).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


def _freeze_json(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class GPUInfo:
    """A single detected GPU."""

    vendor: str
    name: str
    vram_mb: int | None
    method: str
    vram_free_mb: int | None = None
    vram_free_method: str | None = None
    vram_free_at: str | None = None
    driver_version: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class GpuSample:
    """One best-effort observation of GPU memory and utilization."""

    ts: str
    vram_used_mb: int
    vram_free_mb: int
    utilization_pct: float
    power_w: float | None = None
    temperature_c: float | None = None
    clocks_sm_mhz: float | None = None
    device_index: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class HardwareReport:
    """Best-effort hardware and environment assessment (DESIGN §4)."""

    os_name: str
    arch: str
    cpu_model: str
    physical_cores: int
    logical_cores: int
    perf_cores: int | None
    ram_mb: int
    gpus: tuple[GPUInfo, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelReport:
    """GGUF model metadata and fingerprint (DESIGN §5)."""

    path: Path
    size_bytes: int
    architecture: str
    n_layer: int
    ngl_all: int
    expert_count: int
    moe: bool
    name: str | None
    fingerprint: str
    full_sha256: str | None
    expert_bytes: int | None = None
    dense_bytes: int | None = None
    kv_bytes_per_token_f16: int | None = None
    n_kv_layers: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class LlamaCppReport:
    """llama.cpp binary discovery and capability probe result."""

    bench_path: Path
    cli_path: Path | None
    server_path: Path | None
    capabilities: frozenset[str]
    help_sha256: str
    build_commit: str | None
    build_number: int | None
    backends: str | None = None
    perplexity_path: Path | None = None
    bench_sha256: str | None = None


# Capability keys gated behind a `llama-bench --help` probe; flags without an
# entry here (-ngl, -ub, -b, -t) are treated as always-supported core flags.
_GATED_FLAGS: tuple[tuple[str, str], ...] = (
    ("ncmoe", "-ncmoe"),
    ("fa", "-fa"),
    ("mmp", "-mmp"),
    ("nkvo", "-nkvo"),
    ("ctk", "-ctk"),
    ("ctv", "-ctv"),
    ("tb", "-tb"),
    ("ot", "-ot"),
    ("ts", "-ts"),
    ("sm", "-sm"),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class TrialConfig:
    """One point in the search space: the tunable llama-bench dimensions."""

    gpu_layers: int
    moe_cpu_layers: int
    flash_attn: bool
    ubatch: int
    batch: int
    threads: int
    mmap: bool
    no_kv_offload: bool
    cache_type_k: str
    cache_type_v: str
    threads_batch: int | None = None
    ot_spec: str | None = None
    tensor_split: tuple[float, ...] | None = None
    split_mode: str | None = None

    @property
    def trial_id(self) -> str:
        """Deterministic trial identifier: sha256(canonical JSON)[:16]."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "gpu_layers": self.gpu_layers,
            "moe_cpu_layers": self.moe_cpu_layers,
            "flash_attn": self.flash_attn,
            "ubatch": self.ubatch,
            "batch": self.batch,
            "threads": self.threads,
            "mmap": self.mmap,
            "no_kv_offload": self.no_kv_offload,
            "cache_type_k": self.cache_type_k,
            "cache_type_v": self.cache_type_v,
        }
        if self.threads_batch is not None:
            data["threads_batch"] = self.threads_batch
        if self.ot_spec is not None:
            data["ot_spec"] = self.ot_spec
        if self.tensor_split is not None:
            data["tensor_split"] = list(self.tensor_split)
        if self.split_mode is not None:
            data["split_mode"] = self.split_mode
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrialConfig:
        return cls(
            gpu_layers=int(data["gpu_layers"]),
            moe_cpu_layers=int(data["moe_cpu_layers"]),
            flash_attn=bool(data["flash_attn"]),
            ubatch=int(data["ubatch"]),
            batch=int(data["batch"]),
            threads=int(data["threads"]),
            mmap=bool(data["mmap"]),
            no_kv_offload=bool(data["no_kv_offload"]),
            cache_type_k=str(data["cache_type_k"]),
            cache_type_v=str(data["cache_type_v"]),
            threads_batch=(
                int(data["threads_batch"]) if data.get("threads_batch") is not None else None
            ),
            ot_spec=str(data["ot_spec"]) if data.get("ot_spec") is not None else None,
            tensor_split=(
                tuple(float(value) for value in data["tensor_split"])
                if data.get("tensor_split") is not None
                else None
            ),
            split_mode=(str(data["split_mode"]) if data.get("split_mode") is not None else None),
        )

    def bench_args(self, capabilities: frozenset[str]) -> tuple[str, ...]:
        """Build the llama-bench flag list for this config.

        Only flags whose capability key is present in ``capabilities`` are
        emitted (an unsupported flag is silently dropped rather than passed
        to a llama-bench build that would reject it). ``-ngl``/``-ub``/
        ``-b``/``-t`` have no gating capability key and are always
        applicable, so they are always emitted explicitly. Every other
        applicable (capability-supported) dimension is likewise always
        emitted explicitly, even when its value equals llama-bench's own
        default, so the trial's evidence unambiguously reflects the
        requested configuration.
        """
        args: list[str] = ["-ngl", str(self.gpu_layers)]

        gated_values: dict[str, str] = {
            "ncmoe": str(self.moe_cpu_layers),
            "fa": "1" if self.flash_attn else "0",
            "mmp": "1" if self.mmap else "0",
            "nkvo": "1" if self.no_kv_offload else "0",
            "ctk": self.cache_type_k,
            "ctv": self.cache_type_v,
        }
        if self.threads_batch is not None:
            gated_values["tb"] = str(self.threads_batch)
        if self.ot_spec is not None:
            gated_values["ot"] = self.ot_spec
        if self.tensor_split is not None:
            gated_values["ts"] = ",".join(f"{value:g}" for value in self.tensor_split)
        if self.split_mode is not None:
            gated_values["sm"] = self.split_mode

        args += ["-ub", str(self.ubatch)]
        args += ["-b", str(self.batch)]
        args += ["-t", str(self.threads)]

        for cap_key, flag in _GATED_FLAGS:
            if cap_key == "ncmoe" and self.ot_spec is not None:
                continue
            if cap_key in capabilities and cap_key in gated_values:
                args += [flag, gated_values[cap_key]]

        return tuple(args)


@dataclass(frozen=True, slots=True, kw_only=True)
class MetricStats:
    """Mean/stdev/coefficient-of-variation summary of repeated measurements."""

    mean: float
    stdev: float
    cv: float
    n: int


@dataclass(frozen=True, slots=True, kw_only=True)
class TrialResult:
    """The outcome of executing (or pruning) one trial."""

    trial_id: str
    config: TrialConfig
    status: str
    pp: MetricStats | None
    tg: MetricStats | None
    wall_s: float
    exit_code: int | None
    oom_pattern: str | None
    artifact_dir: Path | None
    flags: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class BaselineResult:
    """Aggregated baseline measurement and the resolved default config."""

    runs: int
    pp: MetricStats
    tg: MetricStats
    noise_floor_cv: float
    fallback: str | None
    resolved_defaults: dict[str, Any]
    kind: str = "defaults"


@dataclass(frozen=True, slots=True, kw_only=True)
class TuneOptions:
    """Resolved CLI options for the `tune` command."""

    target: str
    budget_trials: int
    budget_minutes: float | None
    reps_search: int
    reps_confirm: int
    baseline_runs: int
    pp: int
    tg: int
    allow_lossy: bool
    cooldown_s: float
    baseline_only: bool
    llama_bin: Path | None
    sessions_dir: Path
    full_hash: bool
    ctx_size: int | None = None
    ctx_ladder: tuple[int, ...] = ()
    vram_reserve_mb: int | None = None
    initial_gpu_layers: int | None = None
    max_gpu_layers: int | None = None
    initial_cpu_moe: int | None = None
    allow_core_dumps: bool = False
    quiet_wait_s: float = 0.0
    quiet_load: float | None = None
    observe_vram: bool = True
    validate_with_cli: bool = False
    quality_corpus: Path | None = None
    batched_trials: bool = True
    ot_search: bool = False
    depth: int | None = None
    depth_profile: tuple[int, ...] | None = None
    thermal_threshold_c: float = 75.0
    thermal_wait_cap_s: float = 60.0
    multi_gpu: bool = False
    probe_timeout_scale: float = 2.5
    probe_timeout_s: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class VramCalibration:
    """Persisted correction factors fitted from observed memory evidence."""

    schema_version: int
    fitted_at: str
    samples: int
    weights_scale: float
    kv_scale: float
    compute_scale: float


@dataclass(frozen=True, slots=True, kw_only=True)
class TuneOutcome:
    """The result of a (possibly resumed) tuning run."""

    session_dir: Path
    analysis: dict[str, Any]
    exit_code: int
    failure_stage: str | None = None
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class VramEstimate:
    """Advisory decomposition of estimated GPU memory use."""

    weights_mb: float
    kv_mb: float
    compute_mb: float
    total_mb: float
    reserve_mb: float
    budget_mb: float | None
    kv_basis: str = "heuristic"
    calibrated: bool = False
    budget_basis: str = "total-reserve"


@dataclass(frozen=True, slots=True, kw_only=True)
class RamEstimate:
    """Advisory decomposition of estimated host memory use."""

    weights_mb: float
    kv_mb: float
    total_mb: float


@dataclass(frozen=True, slots=True, kw_only=True)
class FeasibilityBoundary:
    """Observed fitting/failing GPU-layer edge for one MoE placement."""

    moe_cpu_layers: int
    max_ok_ngl: int
    min_fail_ngl: int | None
    probes: int
    cap_ngl: int | None = None
    cap_source: str = "model"
    spill_suspected: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class ProgressEvent:
    """Ephemeral progress notification emitted by the tuning engine."""

    kind: str
    ts: str
    payload: dict[str, Any]


class Reporter(Protocol):
    """Consumer of ephemeral tuning progress events."""

    def emit(self, event: ProgressEvent) -> None:
        """Render or otherwise consume one progress event."""


@dataclass(frozen=True, slots=True, kw_only=True)
class DiscoveredModel:
    """One benchmarkable GGUF entry discovered for Night Shift."""

    path: Path
    report: ModelReport
    shard_paths: tuple[Path, ...]
    group_key: str | None
    representative: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class RegistryRecord:
    """Latest completed tuning evidence for one model fingerprint."""

    fingerprint: str
    session_dir: Path
    created: str
    outcome: str
    reference_config: TrialConfig | None
    reference_pp: float
    reference_tg: float
    noise_floor_cv: float
    pp_workload: int
    tg_workload: int
    reps_confirm: int
    target: str
    build_commit: str | None
    help_sha256: str
    median_trial_wall_s: float | None
    session_wall_s: float | None
    depth_workload: int | None = None
    bench_sha256: str | None = None
    ctx_size: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class CalibrationResult:
    """Night Shift verification result for a recorded recommendation."""

    fingerprint: str
    reference_session: Path
    verdict: str
    pp: MetricStats | None
    tg: MetricStats | None
    drift_pp: float | None
    drift_tg: float | None
    threshold: float
    runs: int
    reason: str | None
    transfer_from: str | None
    build_changed: bool
    artifact_dir: Path | None


@dataclass(frozen=True, slots=True, kw_only=True)
class NightshiftOptions:
    """Validated options for one unattended Night Shift run."""

    models_dir: Path
    llama_bin: Path | None
    sessions_dir: Path
    until: str | None
    max_hours: float | None
    profile: str
    drift_threshold: float
    calibration_runs: int
    duplicates: str
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    dry_run: bool
    target: str
    allow_lossy: bool
    ctx_size: int | None
    vram_reserve_mb: int | None
    cooldown_s: float | None
    full_hash: bool
    budget_trials: int | None
    reps_search: int | None
    reps_confirm: int | None
    baseline_runs: int | None
    depth: int | None = None
    ctx_ladder: tuple[int, ...] = ()
    probe_timeout_scale: float = 2.5
    probe_timeout_s: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkItem:
    """One serial Night Shift scheduling unit."""

    kind: str
    model_path: Path | None
    fingerprint: str | None
    session_dir: Path | None
    reference_fingerprint: str | None
    estimated_minutes: float | None
    reason: str
    depth_workload: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class NightshiftOutcome:
    """Final Night Shift summary and process exit code."""

    run_dir: Path
    summary: dict[str, Any]
    exit_code: int


@dataclass(frozen=True, slots=True, kw_only=True)
class MarathonOptions:
    """Validated options for one exhaustive single-model Marathon run."""

    model_path: Path
    llama_bin: Path | None
    sessions_dir: Path
    until: str | None
    max_hours: float | None
    rounds_max: int
    converge_rounds: int
    ab_blocks: int
    depth_grid: tuple[int, ...]
    ctx_size: int | None
    ctx_ladder: tuple[int, ...]
    matrix_refine: bool
    drift_threshold: float
    dry_run: bool
    target: str
    allow_lossy: bool
    vram_reserve_mb: int | None
    cooldown_s: float | None
    full_hash: bool
    pp: int
    tg: int
    quality_corpus: Path | None
    ot_search: bool
    budget_trials: int | None
    reps_search: int | None
    reps_confirm: int | None
    baseline_runs: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class TierCoverage:
    """Derived coverage accounting for one Marathon search tier."""

    enumerated: int
    executed: int
    pruned: int
    remaining: int
    remaining_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class CoverageLedger:
    """Coverage and responsive dimensions derived from session evidence."""

    tiers: dict[str, TierCoverage]
    responsive: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class RoundRecord:
    """Summary of one tuning round within a Marathon."""

    index: int
    session_dir: Path
    exit_code: int
    winner_config: TrialConfig | None
    champion_changed: bool
    wall_s: float
    coverage_pct: float


@dataclass(frozen=True, slots=True, kw_only=True)
class MatrixCell:
    """One measured or deferred context-by-depth operating point."""

    ctx: int
    depth: int
    status: str
    config: TrialConfig | None
    pp: float | None
    tg: float | None
    refined: bool
    evidence: Path | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ABResult:
    """Result of an interleaved Marathon A/B comparison."""

    label: str
    blocks: int
    a_wins: int
    b_wins: int
    ties: int
    a_pp: float
    a_tg: float
    b_pp: float
    b_tg: float
    margin: float
    verdict: str


@dataclass(frozen=True, slots=True, kw_only=True)
class MarathonOutcome:
    """Final Marathon summary and process exit code."""

    run_dir: Path
    summary: dict[str, Any]
    exit_code: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ResultRow:
    """One normalized measurement in the Results Matrix."""

    row_id: str
    kind: str
    current: bool
    model_fingerprint: str
    model_name: str | None
    model_path: str
    quant: str | None
    hardware_hash: str
    hardware_signature: tuple[Any, ...]
    build_discriminator: str
    build_commit: str | None
    config: TrialConfig | None
    ctx: int | None
    depth: int | None
    pp_workload: int | None
    tg_workload: int | None
    suite_id: str | None
    metrics: dict[str, float]
    status: str
    confirmed: bool
    replicated: bool | None
    reps: int | None
    noise_floor_cv: float | None
    source_root: Path
    evidence_dir: Path
    ts: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "kind": self.kind,
            "current": self.current,
            "model_fingerprint": self.model_fingerprint,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "quant": self.quant,
            "hardware_hash": self.hardware_hash,
            "hardware_signature": list(self.hardware_signature),
            "build_discriminator": self.build_discriminator,
            "build_commit": self.build_commit,
            "config": self.config.to_dict() if self.config is not None else None,
            "ctx": self.ctx,
            "depth": self.depth,
            "pp_workload": self.pp_workload,
            "tg_workload": self.tg_workload,
            "suite_id": self.suite_id,
            "metrics": dict(self.metrics),
            "status": self.status,
            "confirmed": self.confirmed,
            "replicated": self.replicated,
            "reps": self.reps,
            "noise_floor_cv": self.noise_floor_cv,
            "source_root": str(self.source_root),
            "evidence_dir": str(self.evidence_dir),
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResultRow:
        raw_config = data.get("config")
        raw_metrics = data.get("metrics", {})
        if not isinstance(raw_metrics, dict):
            raise ValueError("metrics must be an object")
        return cls(
            row_id=str(data["row_id"]),
            kind=str(data["kind"]),
            current=bool(data["current"]),
            model_fingerprint=str(data["model_fingerprint"]),
            model_name=(str(data["model_name"]) if data.get("model_name") is not None else None),
            model_path=str(data["model_path"]),
            quant=str(data["quant"]) if data.get("quant") is not None else None,
            hardware_hash=str(data["hardware_hash"]),
            hardware_signature=tuple(
                _freeze_json(item) for item in data.get("hardware_signature", ())
            ),
            build_discriminator=str(data["build_discriminator"]),
            build_commit=(
                str(data["build_commit"]) if data.get("build_commit") is not None else None
            ),
            config=(TrialConfig.from_dict(raw_config) if isinstance(raw_config, dict) else None),
            ctx=int(data["ctx"]) if data.get("ctx") is not None else None,
            depth=int(data["depth"]) if data.get("depth") is not None else None,
            pp_workload=(int(data["pp_workload"]) if data.get("pp_workload") is not None else None),
            tg_workload=(int(data["tg_workload"]) if data.get("tg_workload") is not None else None),
            suite_id=str(data["suite_id"]) if data.get("suite_id") is not None else None,
            metrics={str(key): float(value) for key, value in raw_metrics.items()},
            status=str(data["status"]),
            confirmed=bool(data["confirmed"]),
            replicated=(bool(data["replicated"]) if data.get("replicated") is not None else None),
            reps=int(data["reps"]) if data.get("reps") is not None else None,
            noise_floor_cv=(
                float(data["noise_floor_cv"]) if data.get("noise_floor_cv") is not None else None
            ),
            source_root=Path(str(data["source_root"])),
            evidence_dir=Path(str(data["evidence_dir"])),
            ts=str(data["ts"]),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ResultsMatrix:
    """A deterministic aggregation of normalized evidence rows."""

    schema_version: int
    generated: str
    roots: tuple[Path, ...]
    rows: tuple[ResultRow, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class MatrixQuerySpec:
    """Filters and ordering requested by one matrix query."""

    use_case: str | None
    sort: str | None
    ascending: bool
    model: str | None
    quant: str | None
    kinds: tuple[str, ...]
    suite: str | None
    ctx_min: int | None
    depth: int | None
    min_pp: float | None
    min_tg: float | None
    confirmed_only: bool
    current_only: bool
    compat: str
    limit: int


@dataclass(frozen=True, slots=True, kw_only=True)
class QualityOptions:
    """Resolved options for one deterministic quality evaluation."""

    model_path: Path
    llama_bin: Path | None
    sessions_dir: Path
    config_mode: str
    config_session: Path | None
    strict_config: bool
    compare_lossless: bool
    suites: tuple[str, ...]
    task_filters: tuple[str, ...]
    exec_enabled: bool
    ctx_size: int
    quality_corpus: Path | None
    reps: int
    max_tokens: int
    request_timeout_s: float
    server_start_timeout_s: float
    seed: int
    dry_run: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskGrade:
    """A deterministic grade for one quality-suite task."""

    task_id: str
    score: float
    status: str
    reason: str | None
    unstable: bool
    grader_results: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class SuiteResult:
    """Aggregated task grades and metrics for one suite."""

    suite_id: str
    name: str
    kind: str
    metrics: dict[str, float]
    tasks: tuple[TaskGrade, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class QualityOutcome:
    """Completed quality run location, summary, and CLI exit code."""

    run_dir: Path
    summary: dict[str, Any]
    exit_code: int
