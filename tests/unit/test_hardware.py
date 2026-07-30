"""Unit tests for llamatune.hardware.

External probes are exercised by monkeypatching `hardware._run_probe` (the
sole subprocess boundary) rather than invoking real system tools, so these
tests run identically in CI with no GPU and no vendor tools installed.
"""

from __future__ import annotations

import json
import os
import platform
import struct
import sys
import types
from pathlib import Path

import pytest

from llamatune import hardware
from llamatune.types import GPUInfo, GpuSample

_CPUINFO_TWO_SOCKET_HT = """\
processor\t: 0
physical id\t: 0
core id\t: 0

processor\t: 1
physical id\t: 0
core id\t: 0

processor\t: 2
physical id\t: 0
core id\t: 1

processor\t: 3
physical id\t: 1
core id\t: 0

model name\t: Fake CPU Model X
"""


def test_physical_cores_linux_counts_unique_physical_core_pairs() -> None:
    assert hardware._physical_cores_linux(_CPUINFO_TWO_SOCKET_HT) == 3


def test_physical_cores_linux_missing_fields_returns_none() -> None:
    assert hardware._physical_cores_linux("processor\t: 0\n") is None


def test_physical_cores_linux_skips_incomplete_and_malformed_records() -> None:
    text = """\
physical id: 0

malformed line
physical id: 1
core id: 2"""
    assert hardware._physical_cores_linux(text) == 1


def test_cpu_model_linux_parses_model_name() -> None:
    assert hardware._cpu_model_linux(_CPUINFO_TWO_SOCKET_HT) == "Fake CPU Model X"


def test_cpu_model_linux_missing_returns_none() -> None:
    assert hardware._cpu_model_linux("processor\t: 0\n") is None


def test_detect_cpu_linux_falls_back_without_sched_getaffinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 7)
    assert hardware._detect_cpu_linux()[2] == 7


def test_parse_mem_total_mb() -> None:
    text = "MemTotal:       16384000 kB\nMemAvailable:   8000000 kB\n"
    assert hardware._parse_mem_total_mb(text) == 16384000 // 1024


def test_parse_mem_total_mb_missing_key() -> None:
    assert hardware._parse_mem_total_mb("SomeOtherKey: 123 kB\n") == 0


def test_parse_mem_total_mb_ignores_malformed_total() -> None:
    assert hardware._parse_mem_total_mb("MemTotal: unavailable\n") == 0


def test_detect_load_handles_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> tuple[float, float, float]:
        raise OSError

    monkeypatch.setattr(os, "getloadavg", _raise, raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    assert hardware.detect_load() is None


def test_run_probe_missing_binary_returns_none() -> None:
    assert hardware._run_probe(["definitely-not-a-real-binary-xyz"]) is None


def test_run_probe_nonzero_exit_returns_none() -> None:
    assert hardware._run_probe(["false"]) is None


def test_run_probe_success_returns_stdout() -> None:
    out = hardware._run_probe(["printf", "hello"])
    assert out == "hello"


# -- GPU detection --------------------------------------------------------------


def test_detect_gpus_nvidia_parses_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    csv_output = (
        "NVIDIA GeForce RTX 4090, 24564 MiB, 20000 MiB, 595.80, 17 %\n"
        "NVIDIA GeForce RTX 3060, 12288 MiB, 10000 MiB, 595.80, 2 %\n"
    )
    monkeypatch.setattr(hardware, "_run_probe", lambda argv: csv_output)
    gpus = hardware._detect_gpus_nvidia()
    assert [gpu.vram_mb for gpu in gpus] == [24564, 12288]
    assert [gpu.vram_free_mb for gpu in gpus] == [20000, 10000]
    assert all(gpu.vram_free_method == "nvidia-smi" for gpu in gpus)
    assert all(gpu.driver_version == "595.80" for gpu in gpus)
    assert all(gpu.vram_free_at is not None and "+00:00" in gpu.vram_free_at for gpu in gpus)


def test_detect_gpus_nvidia_legacy_csv_keeps_total(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "_run_probe", lambda argv: "Legacy GPU, 8192 MiB\n")
    gpu = hardware._detect_gpus_nvidia()[0]
    assert gpu.vram_mb == 8192
    assert gpu.vram_free_mb is None


def test_detect_gpus_nvidia_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "_run_probe", lambda argv: None)
    assert hardware._detect_gpus_nvidia() == []


def test_sample_gpu_state_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "_run_probe", lambda argv: "1024, 14336, 37, 125.5, 68, 2415\n")
    sample = hardware.sample_gpu_state()
    assert sample is not None
    assert sample.vram_used_mb == 1024
    assert sample.vram_free_mb == 14336
    assert sample.utilization_pct == 37.0
    assert sample.power_w == 125.5
    assert sample.temperature_c == 68
    assert sample.clocks_sm_mhz == 2415
    assert sample.device_index == 0


def test_sample_gpu_states_attributes_each_nvidia_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hardware,
        "_run_probe",
        lambda argv: "1024, 14336, 37, 125, 68, 2415\n2048, 22528, 42, 175, 71, 2500\n",
    )
    samples = hardware.sample_gpu_states()
    assert [(sample.device_index, sample.vram_used_mb) for sample in samples] == [
        (0, 1024),
        (1, 2048),
    ]
    first = hardware.sample_gpu_state()
    assert first is not None
    assert (first.device_index, first.vram_used_mb) == (0, 1024)


def test_sample_gpu_state_partial_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "_run_probe", lambda argv: "1024, 14336, 37, N/A, 65, N/A\n")
    sample = hardware.sample_gpu_state()
    assert sample is not None
    assert sample.power_w is None
    assert sample.temperature_c == 65
    assert sample.clocks_sm_mhz is None


def test_optional_gpu_telemetry_missing_column() -> None:
    assert hardware._optional_float(["1"], 1) is None


def test_sample_gpu_state_skips_malformed_rows_then_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hardware, "_run_probe", lambda argv: "short,row\nN/A, 100, 10\n")
    monkeypatch.setattr(hardware, "_sample_gpu_states_amd_sysfs", lambda: ())
    assert hardware.sample_gpu_state() is None


def test_detect_throttle() -> None:
    def samples(clocks: list[float]) -> list[GpuSample]:
        return [
            GpuSample(
                ts=str(index),
                vram_used_mb=1,
                vram_free_mb=1,
                utilization_pct=1,
                clocks_sm_mhz=clock,
            )
            for index, clock in enumerate(clocks)
        ]

    assert hardware.detect_throttle(samples([2400, 2400, 2300, 2200, 2100, 2000, 2000, 2000]))
    assert not hardware.detect_throttle(samples([2400] * 8))
    assert not hardware.detect_throttle(samples([2400] * 7))


def test_detect_throttle_ignores_partial_clock_samples() -> None:
    samples = [
        GpuSample(ts=str(i), vram_used_mb=1, vram_free_mb=1, utilization_pct=1) for i in range(8)
    ]
    assert not hardware.detect_throttle(samples)


def test_sample_gpu_state_falls_back_to_amd_sysfs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    device = tmp_path / "card0" / "device"
    device.mkdir(parents=True)
    (device / "mem_info_vram_total").write_text(str(8 * 1024 * 1024 * 1024))
    (device / "mem_info_vram_used").write_text(str(2 * 1024 * 1024 * 1024))
    (device / "gpu_busy_percent").write_text("42")
    monkeypatch.setattr(hardware, "_run_probe", lambda argv: None)
    sample_sysfs = hardware._sample_gpu_states_amd_sysfs
    monkeypatch.setattr(
        hardware,
        "_sample_gpu_states_amd_sysfs",
        lambda: sample_sysfs(tmp_path),
    )
    sample = hardware.sample_gpu_state()
    assert sample is not None
    assert sample.vram_used_mb == 2048
    assert sample.vram_free_mb == 6144
    assert sample.utilization_pct == 42.0


def test_sample_gpu_state_missing_probes_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "_run_probe", lambda argv: None)
    monkeypatch.setattr(hardware, "_sample_gpu_states_amd_sysfs", lambda: ())
    assert hardware.sample_gpu_state() is None


def test_sample_gpu_states_amd_handles_missing_and_malformed_cards(tmp_path: Path) -> None:
    assert hardware._sample_gpu_states_amd_sysfs(tmp_path / "missing") == ()
    bad_device = tmp_path / "card0" / "device"
    bad_device.mkdir(parents=True)
    (bad_device / "mem_info_vram_total").write_text("not-a-number")
    (bad_device / "mem_info_vram_used").write_text("1")
    good_device = tmp_path / "card1" / "device"
    good_device.mkdir(parents=True)
    (good_device / "mem_info_vram_total").write_text(str(4 * 1024 * 1024))
    (good_device / "mem_info_vram_used").write_text(str(1024 * 1024))
    samples = hardware._sample_gpu_states_amd_sysfs(tmp_path)
    assert len(samples) == 1
    assert samples[0].device_index == 1
    assert samples[0].utilization_pct == 0.0


def test_sample_gpu_states_amd_uses_default_sysfs_path() -> None:
    assert isinstance(hardware._sample_gpu_states_amd_sysfs(), tuple)


def test_detect_gpus_amd_rocm_smi_parses_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    rocm_output = "GPU[0]\t\t: VRAM Total Memory (B): 17163091968\n"
    monkeypatch.setattr(hardware, "_run_probe", lambda argv: rocm_output)
    gpus = hardware._detect_gpus_amd_rocm_smi()
    assert len(gpus) == 1
    assert gpus[0].vendor == "amd"
    assert gpus[0].vram_mb == 17163091968 // (1024 * 1024)


def test_detect_gpus_amd_sysfs_fallback(tmp_path: Path) -> None:
    card_dir = tmp_path / "card0" / "device"
    card_dir.mkdir(parents=True)
    (card_dir / "mem_info_vram_total").write_text(str(8 * 1024 * 1024 * 1024))
    (card_dir / "mem_info_vram_used").write_text(str(2 * 1024 * 1024 * 1024))
    gpus = hardware._detect_gpus_amd_sysfs(drm_dir=tmp_path)
    assert len(gpus) == 1
    assert gpus[0].vram_mb == 8192
    assert gpus[0].method == "sysfs:mem_info_vram_total"
    assert gpus[0].vram_free_mb == 6144
    assert gpus[0].vram_free_method == "sysfs:mem_info_vram_used"


def test_detect_gpus_amd_sysfs_missing_dir(tmp_path: Path) -> None:
    assert hardware._detect_gpus_amd_sysfs(drm_dir=tmp_path / "nonexistent") == []


def test_detect_gpus_amd_prefers_rocm_smi_over_sysfs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        hardware,
        "_detect_gpus_amd_rocm_smi",
        lambda: [GPUInfo(vendor="amd", name="rocm-gpu", vram_mb=1, method="rocm-smi")],
    )
    monkeypatch.setattr(
        hardware,
        "_detect_gpus_amd_sysfs",
        lambda: [GPUInfo(vendor="amd", name="sysfs-gpu", vram_mb=2, method="sysfs")],
    )
    gpus = hardware._detect_gpus_amd()
    assert len(gpus) == 1
    assert gpus[0].name == "rocm-gpu"


def test_detect_gpus_apple_uses_unified_memory_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps({"SPDisplaysDataType": [{"sppci_model": "Apple M2 Max"}]})
    monkeypatch.setattr(hardware, "_run_probe", lambda argv: payload)
    gpus = hardware._detect_gpus_apple(ram_mb=32768, arch="arm64")
    assert len(gpus) == 1
    assert gpus[0].vendor == "apple"
    assert gpus[0].name == "Apple M2 Max"
    assert gpus[0].vram_mb == round(0.75 * 32768)
    assert gpus[0].method == "estimate"


def test_detect_gpus_apple_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "_run_probe", lambda argv: "not json")
    assert hardware._detect_gpus_apple(ram_mb=1024, arch="arm64") == []


@pytest.mark.parametrize(
    ("entry", "expected_vram_mb"),
    [
        ({"spdisplays_vram": "2 GB"}, 2048),
        ({"spdisplays_vram_shared": "1536 MB"}, 1536),
        ({}, None),
    ],
)
def test_detect_gpus_apple_intel_uses_reported_vram(
    monkeypatch: pytest.MonkeyPatch, entry: dict[str, str], expected_vram_mb: int | None
) -> None:
    payload = json.dumps({"SPDisplaysDataType": [{"sppci_model": "AMD Radeon", **entry}]})
    monkeypatch.setattr(hardware, "_run_probe", lambda argv: payload)

    gpus = hardware._detect_gpus_apple(ram_mb=32768, arch="x86_64")

    assert len(gpus) == 1
    assert gpus[0].vram_mb == expected_vram_mb
    assert gpus[0].method == "system_profiler"


# -- macOS CPU/memory via sysctl -------------------------------------------------


def test_detect_cpu_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "machdep.cpu.brand_string": "Apple M2 Max",
        "hw.physicalcpu": "12",
        "hw.logicalcpu": "12",
        "hw.perflevel0.physicalcpu": "8",
    }
    monkeypatch.setattr(hardware, "_sysctl", lambda name: values.get(name))
    model, physical, logical, perf, method = hardware._detect_cpu_macos()
    assert model == "Apple M2 Max"
    assert physical == 12
    assert logical == 12
    assert perf == 8
    assert method == "sysctl"


def test_detect_memory_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "_sysctl", lambda name: str(64 * 1024 * 1024 * 1024))
    assert hardware._detect_memory_macos() == 64 * 1024


def test_detect_memory_macos_rejects_nonnumeric_sysctl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hardware, "_sysctl", lambda name: "unavailable")
    assert hardware._detect_memory_macos() == 0


def test_count_windows_processor_cores_parses_variable_records() -> None:
    raw = struct.pack("<II", 0, 8) + struct.pack("<II", 1, 8) + struct.pack("<II", 0, 8)
    assert hardware._count_windows_processor_cores(raw) == 2
    assert hardware._count_windows_processor_cores(struct.pack("<II", 0, 999)) is None


def test_windows_cpu_model_uses_registry_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class Key:
        def __enter__(self) -> Key:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    fake = types.SimpleNamespace(
        HKEY_LOCAL_MACHINE=object(),
        OpenKey=lambda root, path: Key(),
        QueryValueEx=lambda key, name: (" Fake Windows CPU ", 1),
    )
    monkeypatch.setitem(sys.modules, "winreg", fake)
    assert hardware._windows_cpu_model() == "Fake Windows CPU"
    fake.QueryValueEx = lambda key, name: (_ for _ in ()).throw(OSError())
    monkeypatch.setattr(platform, "processor", lambda: "Fallback CPU")
    assert hardware._windows_cpu_model() == "Fallback CPU"


def test_detect_gpus_windows_wmi_names_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        hardware,
        "_run_probe",
        lambda argv: "NVIDIA RTX 4090\nAMD Radeon RX 7900\nIntel Arc A770\n",
    )
    gpus = hardware._detect_gpus_windows_wmi({"NVIDIA RTX 4090"})
    assert [(gpu.vendor, gpu.name, gpu.vram_mb) for gpu in gpus] == [
        ("amd", "AMD Radeon RX 7900", None),
        ("intel", "Intel Arc A770", None),
    ]
    assert all(gpu.method == "wmi:name-only" for gpu in gpus)


def test_assess_hardware_windows_uses_windows_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(
        hardware, "_detect_cpu_windows", lambda: ("Windows CPU", 8, 16, None, "win32")
    )
    monkeypatch.setattr(hardware, "_detect_memory_windows", lambda: 32768)
    monkeypatch.setattr(
        hardware,
        "_detect_gpus",
        lambda os_name, ram_mb, arch: [
            GPUInfo(vendor="intel", name="Arc", vram_mb=None, method="wmi:name-only")
        ],
    )
    report = hardware.assess_hardware()
    assert report.os_name == "Windows"
    assert report.cpu_model == "Windows CPU"
    assert report.gpus[0].vram_mb is None
    assert "load monitoring is unavailable on Windows" in report.warnings


def test_detect_load_windows_is_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    assert hardware.detect_load() is None


def test_sysctl_returns_none_when_probe_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "_run_probe", lambda argv: None)
    assert hardware._sysctl("hw.memsize") is None


# -- assess_hardware orchestration -----------------------------------------------


def test_assess_hardware_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(hardware, "_detect_cpu_linux", lambda: ("Fake CPU", 8, 16, None, "method"))
    monkeypatch.setattr(hardware, "_detect_memory_linux", lambda: 32768)
    monkeypatch.setattr(hardware, "_detect_gpus", lambda os_name, ram_mb, arch: [])
    monkeypatch.setattr(hardware, "detect_load", lambda: (0.5, 0.5, 0.5))

    report = hardware.assess_hardware()
    assert report.os_name == "Linux"
    assert report.cpu_model == "Fake CPU"
    assert report.physical_cores == 8
    assert report.ram_mb == 32768
    assert report.gpus == ()
    assert report.warnings == ()


def test_assess_hardware_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(hardware, "_detect_cpu_macos", lambda: ("Apple M2", 12, 12, 8, "sysctl"))
    monkeypatch.setattr(hardware, "_detect_memory_macos", lambda: 65536)
    monkeypatch.setattr(hardware, "_detect_gpus", lambda os_name, ram_mb, arch: [])
    monkeypatch.setattr(hardware, "detect_load", lambda: None)

    report = hardware.assess_hardware()
    assert report.os_name == "Darwin"
    assert report.arch == "arm64"
    assert report.cpu_model == "Apple M2"
    assert report.perf_cores == 8


def test_assess_hardware_warns_on_high_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(hardware, "_detect_cpu_linux", lambda: ("Fake CPU", 4, 8, None, "method"))
    monkeypatch.setattr(hardware, "_detect_memory_linux", lambda: 8192)
    monkeypatch.setattr(hardware, "_detect_gpus", lambda os_name, ram_mb, arch: [])
    monkeypatch.setattr(hardware, "detect_load", lambda: (10.0, 5.0, 2.0))

    report = hardware.assess_hardware()
    assert any("load average" in w for w in report.warnings)


def test_assess_hardware_unrecognized_platform_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "FreeBSD")
    monkeypatch.setattr(platform, "machine", lambda: "amd64")
    monkeypatch.setattr(hardware, "_detect_cpu_linux", lambda: ("Fake CPU", 4, 8, None, "method"))
    monkeypatch.setattr(hardware, "_detect_memory_linux", lambda: 8192)
    monkeypatch.setattr(hardware, "_detect_gpus", lambda os_name, ram_mb, arch: [])
    monkeypatch.setattr(hardware, "detect_load", lambda: None)

    report = hardware.assess_hardware()
    assert any("unrecognized platform" in w for w in report.warnings)
