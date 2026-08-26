"""Unit tests for llamatune.bench (argv construction and output parsing)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llamatune import bench
from llamatune.types import TrialConfig

_ALL_CAPS = frozenset({"fa", "mmp", "nkvo", "ctk", "ctv", "ncmoe", "r", "o", "d"})


@pytest.mark.parametrize(
    ("commit", "number", "expected"),
    [
        ("unknown", 0, (None, None)),
        ("UNKNOWN", None, (None, None)),
        ("", 0, (None, None)),
        (None, None, (None, None)),
        ("abc1234", 0, ("abc1234", None)),
        ("unknown", 4211, (None, 4211)),
        ("abc1234", 4211, ("abc1234", 4211)),
    ],
)
def test_normalize_build_info(
    commit: str | None, number: int | None, expected: tuple[str | None, int | None]
) -> None:
    assert bench.normalize_build_info(commit, number) == expected


def _config(**overrides: Any) -> TrialConfig:
    fields: dict[str, Any] = {
        "gpu_layers": 33,
        "moe_cpu_layers": 0,
        "flash_attn": False,
        "ubatch": 512,
        "batch": 2048,
        "threads": 8,
        "mmap": True,
        "no_kv_offload": False,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
    }
    fields.update(overrides)
    return TrialConfig(**fields)


def _entry(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "build_commit": "fake1234",
        "build_number": 9999,
        "n_prompt": 512,
        "n_gen": 0,
        "avg_ts": 600.0,
        "stddev_ts": 3.0,
    }
    entry.update(overrides)
    return entry


def _valid_output() -> str:
    return json.dumps(
        [
            _entry(n_prompt=512, n_gen=0, avg_ts=600.0, stddev_ts=3.0),
            _entry(n_prompt=0, n_gen=128, avg_ts=40.0, stddev_ts=0.4),
        ]
    )


class TestBuildArgv:
    def test_depth_is_capability_gated_on_trial_baseline_and_batch(self) -> None:
        trial = bench.build_bench_argv(
            bench_path=Path("bench"),
            model_path=Path("m.gguf"),
            pp=512,
            tg=128,
            reps=3,
            config=_config(),
            capabilities=_ALL_CAPS,
            depth=32768,
        )
        baseline = bench.build_baseline_argv(
            bench_path=Path("bench"),
            model_path=Path("m.gguf"),
            pp=512,
            tg=128,
            reps=3,
            capabilities=_ALL_CAPS,
            depth=32768,
        )
        batch = bench.build_bench_batch_argv(
            bench_path=Path("bench"),
            model_path=Path("m.gguf"),
            pp=512,
            tg=128,
            reps=3,
            configs=[_config(threads=4), _config(threads=8)],
            capabilities=_ALL_CAPS,
            depth=32768,
        )
        assert all(
            argv[argv.index("-d") : argv.index("-d") + 2] == ("-d", "32768")
            for argv in (trial, baseline, batch)
        )
        unsupported = bench.build_bench_argv(
            bench_path=Path("bench"),
            model_path=Path("m.gguf"),
            pp=512,
            tg=128,
            reps=3,
            config=_config(),
            capabilities=frozenset(),
            depth=32768,
        )
        assert "-d" not in unsupported

    def test_trial_argv_layout(self) -> None:
        argv = bench.build_bench_argv(
            bench_path=Path("/bin/llama-bench"),
            model_path=Path("/models/m.gguf"),
            pp=512,
            tg=128,
            reps=3,
            config=_config(flash_attn=True),
            capabilities=_ALL_CAPS,
        )
        assert argv[:11] == (
            str(Path("/bin/llama-bench")),
            "-m",
            str(Path("/models/m.gguf")),
            "-p",
            "512",
            "-n",
            "128",
            "-r",
            "3",
            "-o",
            "json",
        )
        rest = argv[11:]
        assert rest[rest.index("-ngl") : rest.index("-ngl") + 2] == ("-ngl", "33")
        assert rest[rest.index("-fa") : rest.index("-fa") + 2] == ("-fa", "1")

    def test_baseline_argv_has_no_tuning_flags(self) -> None:
        argv = bench.build_baseline_argv(
            bench_path=Path("/bin/llama-bench"),
            model_path=Path("/models/m.gguf"),
            pp=256,
            tg=64,
            reps=5,
        )
        assert argv == (
            str(Path("/bin/llama-bench")),
            "-m",
            str(Path("/models/m.gguf")),
            "-p",
            "256",
            "-n",
            "64",
            "-r",
            "5",
            "-o",
            "json",
        )

    def test_baseline_argv_cpu_fallback_appends_ngl(self) -> None:
        argv = bench.build_baseline_argv(
            bench_path=Path("/bin/llama-bench"),
            model_path=Path("/models/m.gguf"),
            pp=256,
            tg=64,
            reps=5,
            ngl=0,
        )
        assert argv[-2:] == ("-ngl", "0")

    def test_context_probe_argv_uses_full_prompt_and_config(self) -> None:
        argv = bench.build_context_probe_argv(
            bench_path=Path("/bin/llama-bench"),
            model_path=Path("/models/m.gguf"),
            ctx=8192,
            config=_config(gpu_layers=22),
            capabilities=_ALL_CAPS,
        )
        assert argv[argv.index("-p") : argv.index("-p") + 2] == ("-p", "8192")
        assert argv[argv.index("-n") : argv.index("-n") + 2] == ("-n", "16")
        assert argv[argv.index("-r") : argv.index("-r") + 2] == ("-r", "1")
        assert argv[argv.index("-ngl") : argv.index("-ngl") + 2] == ("-ngl", "22")
        assert "-d" not in argv

    def test_cli_context_argv_emits_moe_runtime_flags(self) -> None:
        argv = bench.build_cli_context_argv(
            cli_path=Path("/bin/llama-cli"),
            model_path=Path("/models/m.gguf"),
            config=_config(
                gpu_layers=22,
                moe_cpu_layers=10,
                flash_attn=True,
                mmap=False,
                no_kv_offload=True,
                cache_type_k="q8_0",
                cache_type_v="q8_0",
            ),
            ctx=8192,
            moe=True,
        )
        assert argv[:5] == (
            str(Path("/bin/llama-cli")),
            "-m",
            str(Path("/models/m.gguf")),
            "-ngl",
            "22",
        )
        assert argv[argv.index("--n-cpu-moe") :][:2] == ("--n-cpu-moe", "10")
        assert argv[argv.index("-c") :][:2] == ("-c", "8192")
        assert argv[-2:] == ("llamatune context probe", "--no-display-prompt")

    def test_cli_context_argv_non_moe_omits_n_cpu_moe(self) -> None:
        argv = bench.build_cli_context_argv(
            cli_path=Path("llama-cli"),
            model_path=Path("m.gguf"),
            config=_config(moe_cpu_layers=12),
            ctx=4096,
            n_predict=4,
            moe=False,
        )
        assert "--n-cpu-moe" not in argv
        assert argv[argv.index("-n") : argv.index("-n") + 2] == ("-n", "4")

    def test_batch_argv_uses_one_comma_list_dimension(self) -> None:
        configs = [_config(threads=value) for value in (4, 8, 12)]
        argv = bench.build_bench_batch_argv(
            bench_path=Path("llama-bench"),
            model_path=Path("m.gguf"),
            pp=512,
            tg=128,
            reps=1,
            configs=configs,
            capabilities=_ALL_CAPS,
        )
        assert argv[argv.index("-t") : argv.index("-t") + 2] == ("-t", "4,8,12")
        assert argv[argv.index("-b") : argv.index("-b") + 2] == ("-b", "2048")

    def test_batch_argv_rejects_two_varying_dimensions(self) -> None:
        with pytest.raises(ValueError, match="one dimension"):
            bench.build_bench_batch_argv(
                bench_path=Path("llama-bench"),
                model_path=Path("m.gguf"),
                pp=512,
                tg=128,
                reps=1,
                configs=[_config(threads=4, batch=512), _config(threads=8, batch=1024)],
                capabilities=_ALL_CAPS,
            )

    def test_batch_argv_rejects_tensor_split_comma_ambiguity(self) -> None:
        with pytest.raises(ValueError, match="tensor-split"):
            bench.build_bench_batch_argv(
                bench_path=Path("llama-bench"),
                model_path=Path("m.gguf"),
                pp=512,
                tg=128,
                reps=1,
                configs=[
                    _config(tensor_split=(1.0, 1.0), threads=4),
                    _config(tensor_split=(1.0, 1.0), threads=8),
                ],
                capabilities=_ALL_CAPS | {"ts", "sm"},
            )

    def test_batch_argv_rejects_empty_and_mismatched_flag_sets(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            bench.build_bench_batch_argv(
                bench_path=Path("llama-bench"),
                model_path=Path("m.gguf"),
                pp=512,
                tg=128,
                reps=1,
                configs=[],
                capabilities=_ALL_CAPS | {"tb"},
            )
        with pytest.raises(ValueError, match="different flag sets"):
            bench.build_bench_batch_argv(
                bench_path=Path("llama-bench"),
                model_path=Path("m.gguf"),
                pp=512,
                tg=128,
                reps=1,
                configs=[_config(threads_batch=None), _config(threads_batch=8)],
                capabilities=_ALL_CAPS | {"tb"},
            )

    def test_perplexity_argv_and_parser(self) -> None:
        argv = bench.build_perplexity_argv(
            perplexity_path=Path("llama-perplexity"),
            model_path=Path("m.gguf"),
            config=_config(flash_attn=True, cache_type_k="q8_0"),
            corpus=Path("quality.txt"),
            ctx=4096,
            capabilities=_ALL_CAPS,
        )
        assert argv[argv.index("-f") : argv.index("-f") + 2] == ("-f", "quality.txt")
        assert argv[argv.index("-c") : argv.index("-c") + 2] == ("-c", "4096")
        assert bench.parse_perplexity_output("Final estimate: PPL = 7.4142 +/- 0.04") == 7.4142
        assert bench.parse_perplexity_output("garbage") is None

    def test_perplexity_argv_preserves_llamacpp_defaults_without_config(self) -> None:
        argv = bench.build_perplexity_argv(
            perplexity_path=Path("llama-perplexity"),
            model_path=Path("m.gguf"),
            config=None,
            corpus=Path("quality.txt"),
            ctx=4096,
            capabilities=_ALL_CAPS,
        )

        assert argv == (
            "llama-perplexity",
            "-m",
            "m.gguf",
            "-c",
            "4096",
            "-f",
            "quality.txt",
        )

    @pytest.mark.parametrize(
        ("output", "expected"),
        [
            ("PPL = 8.25", 8.25),
            ("ppl=8", 8.0),
            ("perplexity: 9.5", 9.5),
            ("Perplexity = 10", 10.0),
            (b"progress\nperplexity: 6.75\n", 6.75),
        ],
    )
    def test_perplexity_parser_fallback_variants(
        self, output: bytes | str, expected: float
    ) -> None:
        assert bench.parse_perplexity_output(output) == expected

    def test_perplexity_primary_wins_over_progress_values(self) -> None:
        output = "chunk PPL = 99\nFinal estimate: PPL = 7.25\nchunk PPL = 3"
        assert bench.parse_perplexity_output(output) == 7.25

    @pytest.mark.parametrize(
        ("output", "expected"),
        [
            ("Final estimate: PPL = 8\nFinal estimate: PPL = 7", 7.0),
            ("PPL = 8\nPPL = 7", 7.0),
            ("perplexity: 8\nperplexity: 7", 7.0),
            ("PPL = 8\nperplexity: 9", 8.0),
        ],
    )
    def test_perplexity_parser_last_match_within_selected_pattern(
        self, output: str, expected: float
    ) -> None:
        assert bench.parse_perplexity_output(output) == expected

    @pytest.mark.parametrize(
        "output",
        [
            "XPPL = 7",
            "nonperplexity: 7",
            "perplexity improved from 8 to 7",
            "PPL delta = 1",
            "PPL = NaN",
            "PPL = -2",
            "ordinary garbage",
        ],
    )
    def test_perplexity_parser_rejects_false_positives(self, output: str) -> None:
        assert bench.parse_perplexity_output(output) is None

    def test_perplexity_argv_emits_all_supported_runtime_flags(self) -> None:
        argv = bench.build_perplexity_argv(
            perplexity_path=Path("llama-perplexity"),
            model_path=Path("m.gguf"),
            config=_config(
                moe_cpu_layers=10,
                flash_attn=True,
                mmap=False,
                no_kv_offload=True,
                cache_type_k="q8_0",
                cache_type_v="q4_0",
                threads_batch=6,
            ),
            corpus=Path("quality.txt"),
            ctx=8192,
            capabilities=_ALL_CAPS | {"tb"},
        )
        for flag in ("--n-cpu-moe", "-fa", "--no-mmap", "--no-kv-offload", "-ctk", "-ctv", "-tb"):
            assert flag in argv
        # The parser is stream-agnostic: callers may pass captured stderr bytes.
        stderr = b"progress\nFinal estimate: PPL = 9.25 +/- 0.10\n"
        assert bench.parse_perplexity_output(stderr) == 9.25


class TestParseBenchOutput:
    def test_valid_output(self) -> None:
        sample = bench.parse_bench_output(_valid_output())
        assert sample.pp_avg == 600.0
        assert sample.pp_stddev == 3.0
        assert sample.tg_avg == 40.0
        assert sample.tg_stddev == 0.4
        assert sample.pp_entry["n_prompt"] == 512
        assert sample.tg_entry["n_gen"] == 128

    def test_accepts_bytes(self) -> None:
        sample = bench.parse_bench_output(_valid_output().encode("utf-8"))
        assert sample.pp_avg == 600.0

    def test_unknown_extra_fields_ignored(self) -> None:
        data = json.loads(_valid_output())
        data[0]["some_future_field"] = {"nested": True}
        sample = bench.parse_bench_output(json.dumps(data))
        assert sample.pp_avg == 600.0

    def test_non_dict_entries_skipped(self) -> None:
        data = ["noise", *json.loads(_valid_output())]
        sample = bench.parse_bench_output(json.dumps(data))
        assert sample.tg_avg == 40.0

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(bench.BenchParseError, match="not valid JSON"):
            bench.parse_bench_output(b"not json at all")

    def test_non_array_raises(self) -> None:
        with pytest.raises(bench.BenchParseError, match="not a JSON array"):
            bench.parse_bench_output(json.dumps({"n_prompt": 512}))

    def test_missing_pp_entry_raises(self) -> None:
        only_tg = json.dumps([_entry(n_prompt=0, n_gen=128)])
        with pytest.raises(bench.BenchParseError, match="missing a pp"):
            bench.parse_bench_output(only_tg)

    def test_missing_tg_entry_raises(self) -> None:
        only_pp = json.dumps([_entry(n_prompt=512, n_gen=0)])
        with pytest.raises(bench.BenchParseError, match="missing a pp"):
            bench.parse_bench_output(only_pp)

    def test_missing_avg_ts_raises(self) -> None:
        data = json.loads(_valid_output())
        del data[0]["avg_ts"]
        with pytest.raises(bench.BenchParseError, match="missing required field"):
            bench.parse_bench_output(json.dumps(data))

    def test_non_numeric_stddev_ts_raises(self) -> None:
        data = json.loads(_valid_output())
        data[1]["stddev_ts"] = "0.4"
        with pytest.raises(bench.BenchParseError, match="not numeric"):
            bench.parse_bench_output(json.dumps(data))

    def test_boolean_avg_ts_rejected(self) -> None:
        data = json.loads(_valid_output())
        data[0]["avg_ts"] = True
        with pytest.raises(bench.BenchParseError, match="not numeric"):
            bench.parse_bench_output(json.dumps(data))

    def test_duplicate_keys_rejected(self) -> None:
        raw = '[{"n_prompt": 512, "n_prompt": 512, "n_gen": 0, "avg_ts": 1.0, "stddev_ts": 0.0}]'
        with pytest.raises(bench.BenchParseError, match="duplicate key"):
            bench.parse_bench_output(raw)

    def test_boolean_n_prompt_not_a_pp_entry(self) -> None:
        data = [_entry(n_prompt=True, n_gen=0), _entry(n_prompt=0, n_gen=128)]
        with pytest.raises(bench.BenchParseError, match="missing a pp"):
            bench.parse_bench_output(json.dumps(data))

    def test_multi_output_preserves_config_attribution(self) -> None:
        data: list[dict[str, Any]] = []
        for threads in (4, 8, 12):
            config_fields = {
                "n_gpu_layers": 20,
                "n_threads": threads,
                "n_threads_batch": 6,
                "n_batch": 2048,
                "n_ubatch": 512,
                "use_mmap": threads != 8,
            }
            data.extend(
                [
                    _entry(n_prompt=512, n_gen=0, **config_fields),
                    _entry(n_prompt=0, n_gen=128, **config_fields),
                ]
            )
        samples = bench.parse_bench_output_multi(json.dumps(data))
        assert [sample.config_fields["n_threads"] for sample in samples] == [4, 8, 12]
        assert samples[1].config_fields["use_mmap"] is False
        with pytest.raises(bench.BenchParseError, match="3 configurations"):
            bench.parse_bench_output(json.dumps(data))

    def test_multi_output_rejects_incomplete_or_unattributable_pairs(self) -> None:
        incomplete = [_entry(n_prompt=512, n_gen=0, n_threads=4)]
        with pytest.raises(bench.BenchParseError, match="missing a pp"):
            bench.parse_bench_output_multi(json.dumps(incomplete))
        with pytest.raises(bench.BenchParseError, match="missing a pp"):
            bench.parse_bench_output_multi(json.dumps(["noise", 1, None]))


class TestClassifyFailure:
    @pytest.mark.parametrize(
        "stderr_text",
        [
            "ggml_backend_cuda_buffer_type_alloc_buffer: failed to allocate",
            "CUDA error: OUT OF MEMORY",
            "cudaMalloc failed",
            "kIOGPUCommandBufferCallbackErrorOutOfMemory something",
            "ggml_backend_metal: alloc heap fail",
            "Vulkan error: ErrorOutOfDeviceMemory",
            "vk::OutOfDeviceMemoryError while allocating",
            "hipMalloc failed for tensor buffer",
        ],
    )
    def test_oom_signatures(self, stderr_text: str) -> None:
        assert bench.classify_failure(stderr_text) == "oom"
        assert bench.failure_pattern(stderr_text, "oom") is not None

    def test_case_insensitive(self) -> None:
        assert bench.failure_pattern("FAILED TO ALLOCATE buffer", "oom") == ("failed to allocate")

    def test_hip_error_is_cuda_class_error_and_oom_still_precedes(self) -> None:
        assert bench.classify_failure("HIP error: invalid device function") == "cuda_error"
        assert bench.classify_failure("HIP error after hipMalloc failed") == "oom"

    def test_sycl_resource_signature(self) -> None:
        assert bench.classify_failure("PI_ERROR_OUT_OF_RESOURCES") == "gpu_resource"

    @pytest.mark.parametrize(
        "stderr_text",
        [
            "ggml-cuda.cu:106: CUDA error",
            "CUDA error: invalid argument",
            "cuBLAS error during matmul",
            "CUBLAS_STATUS_EXECUTION_FAILED",
        ],
    )
    def test_cuda_error_signatures(self, stderr_text: str) -> None:
        assert bench.classify_failure(stderr_text) == "cuda_error"
        assert bench.failure_pattern(stderr_text, "cuda_error") is not None

    def test_oom_precedes_cuda_error(self) -> None:
        assert bench.classify_failure("CUDA error after out of memory") == "oom"

    @pytest.mark.parametrize(
        "stderr_text",
        ["failed to load model", "FAILED TO CREATE CONTEXT with model", "unable to load model"],
    )
    def test_ambiguous_resource_signatures(self, stderr_text: str) -> None:
        assert bench.classify_failure(stderr_text) == "gpu_resource"

    def test_oom_precedes_resource_signature(self) -> None:
        text = "failed to load model after CUDA out of memory"
        assert bench.classify_failure(text) == "oom"
        assert bench.failure_pattern(text, "oom") == "out of memory"

    def test_unknown_is_none_and_parse_fact_is_independent(self) -> None:
        assert bench.classify_failure("segmentation fault") is None
        assert bench.failure_pattern("segmentation fault (core dumped)", "oom") is None
        with pytest.raises(bench.BenchParseError):
            bench.parse_bench_output("[")
        assert bench.classify_failure("failed to load model") == "gpu_resource"


class TestResolvedConfig:
    def test_parses_baseline_entry(self) -> None:
        entry = {
            "n_gpu_layers": 33,
            "flash_attn": False,
            "n_ubatch": 512,
            "n_batch": 2048,
            "n_threads": 8,
            "use_mmap": True,
            "no_kv_offload": False,
            "type_k": "f16",
            "type_v": "f16",
        }
        config = bench.resolved_config(entry)
        assert config == _config()

    def test_force_gpu_layers(self) -> None:
        entry = {"n_gpu_layers": 33, "n_threads": 8}
        config = bench.resolved_config(entry, force_gpu_layers=0)
        assert config.gpu_layers == 0

    def test_missing_fields_take_defaults(self) -> None:
        config = bench.resolved_config({})
        assert config.gpu_layers == 0
        assert config.ubatch == 512
        assert config.batch == 2048
        assert config.threads == 1
        assert config.mmap is True
        assert config.cache_type_k == "f16"

    def test_parses_multi_gpu_defaults(self) -> None:
        entry = {"tensor_split": "16,24", "split_mode": "row"}
        single = bench.resolved_config(entry)
        assert single.tensor_split is None
        assert single.split_mode is None
        config = bench.resolved_config(entry, multi_gpu=True)
        assert config.tensor_split == (16.0, 24.0)
        assert config.split_mode == "row"
