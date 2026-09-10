"""Hardware and environment assessment (DESIGN §4).

Best-effort and layered: detection failure of any single probe never
aborts the tool. External probes run through the bounded, allowlisted
executor contract in DESIGN §7.
"""

from __future__ import annotations

import json
import os
import platform
import re
import statistics
import struct
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from llamatune import executor
from llamatune.types import GPUInfo, GpuSample, HardwareReport

_PROBE_TIMEOUT_S = 10.0
_PROBE_MAX_BYTES = 65536


def _run_probe(argv: list[str]) -> str | None:
    """Run an external probe with a bounded timeout and capture.

    Returns decoded, truncated stdout on a zero exit; None on any failure
    (missing binary, timeout, non-zero exit, decode error). A probe that
    cannot run must never raise -- callers treat None as "absent".
    """
    result = executor.run_probe(argv, timeout_s=_PROBE_TIMEOUT_S, max_output_bytes=_PROBE_MAX_BYTES)
    if result is None or result.timed_out or result.exit_code != 0:
        return None
    return result.stdout.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
# CPU
# --------------------------------------------------------------------------


def _physical_cores_linux(cpuinfo_text: str) -> int | None:
    entries: set[tuple[str, str]] = set()
    physical_id: str | None = None
    core_id: str | None = None
    for line in cpuinfo_text.splitlines():
        if line.strip() == "":
            if physical_id is not None and core_id is not None:
                entries.add((physical_id, core_id))
            physical_id = None
            core_id = None
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        if key == "physical id":
            physical_id = value.strip()
        elif key == "core id":
            core_id = value.strip()
    if physical_id is not None and core_id is not None:
        entries.add((physical_id, core_id))
    return len(entries) or None


def _cpu_model_linux(cpuinfo_text: str) -> str | None:
    for line in cpuinfo_text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        if key.strip().lower() == "model name":
            return value.strip()
    return None


def _detect_cpu_linux() -> tuple[str, int, int, None, str]:
    """Returns (cpu_model, physical_cores, logical_cores, perf_cores, method)."""
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    try:
        logical = (
            len(sched_getaffinity(0)) if sched_getaffinity is not None else os.cpu_count() or 1
        )
    except OSError:
        logical = os.cpu_count() or 1

    cpuinfo_path = Path("/proc/cpuinfo")
    model: str | None = None
    physical: int | None = None
    try:
        text = cpuinfo_path.read_text(encoding="utf-8", errors="replace")
        model = _cpu_model_linux(text)
        physical = _physical_cores_linux(text)
    except OSError:
        pass

    return (
        model or "unknown",
        physical or logical,
        logical,
        None,
        "/proc/cpuinfo+os.sched_getaffinity",
    )


def _sysctl(name: str) -> str | None:
    out = _run_probe(["sysctl", "-n", name])
    return out.strip() if out is not None else None


def _detect_cpu_macos() -> tuple[str, int, int, int | None, str]:
    """Returns (cpu_model, physical_cores, logical_cores, perf_cores, method)."""
    model = _sysctl("machdep.cpu.brand_string") or "unknown"
    logical_raw = _sysctl("hw.logicalcpu")
    physical_raw = _sysctl("hw.physicalcpu")
    perf_raw = _sysctl("hw.perflevel0.physicalcpu")

    default_count = os.cpu_count() or 1
    logical = int(logical_raw) if logical_raw and logical_raw.isdigit() else default_count
    physical = int(physical_raw) if physical_raw and physical_raw.isdigit() else logical
    perf = int(perf_raw) if perf_raw and perf_raw.isdigit() else None

    return model, physical, logical, perf, "sysctl"


def _windows_cpu_model() -> str:
    try:
        import winreg

        registry: Any = winreg
        key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"  # pragma: allowlist secret
        with registry.OpenKey(registry.HKEY_LOCAL_MACHINE, key_path) as key:
            value, _kind = registry.QueryValueEx(key, "ProcessorNameString")
        if value:
            return str(value).strip()
    except (ImportError, OSError):
        pass
    return platform.processor() or "unknown"


def _count_windows_processor_cores(raw: bytes) -> int | None:
    """Count RelationProcessorCore records in a Win32 variable-size buffer."""
    offset = 0
    count = 0
    while offset + 8 <= len(raw):
        relationship, size = struct.unpack_from("<II", raw, offset)
        if size < 8 or offset + size > len(raw):
            return None
        if relationship == 0:
            count += 1
        offset += size
    return count or None


def _windows_physical_cores() -> int | None:
    try:
        import ctypes

        ctypes_api: Any = ctypes
        size = ctypes.c_ulong(0)
        kernel32 = ctypes_api.windll.kernel32
        kernel32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))
        if size.value <= 0:
            return None
        buffer = ctypes.create_string_buffer(size.value)
        if not kernel32.GetLogicalProcessorInformationEx(0, buffer, ctypes.byref(size)):
            return None
        return _count_windows_processor_cores(buffer.raw[: size.value])
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _detect_cpu_windows() -> tuple[str, int, int, None, str]:
    logical = os.cpu_count() or 1
    physical = _windows_physical_cores() or logical
    return _windows_cpu_model(), physical, logical, None, "winreg+GetLogicalProcessorInformationEx"


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------


def _parse_mem_total_mb(meminfo_text: str) -> int:
    """Parse the `MemTotal:` line (kB) of /proc/meminfo text into MiB."""
    for line in meminfo_text.splitlines():
        if line.startswith("MemTotal:"):
            match = re.search(r"(\d+)", line)
            if match:
                return int(match.group(1)) // 1024
    return 0


def _detect_memory_linux() -> int:
    """Total RAM in MiB, from /proc/meminfo (MemTotal)."""
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    return _parse_mem_total_mb(text)


def _detect_memory_macos() -> int:
    """Total RAM in MiB, from `sysctl hw.memsize` (bytes)."""
    raw = _sysctl("hw.memsize")
    if raw and raw.isdigit():
        return int(raw) // (1024 * 1024)
    return 0


def _detect_memory_windows() -> int:
    try:
        import ctypes

        ctypes_api: Any = ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes_api.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return 0
        return int(status.ullTotalPhys) // (1024 * 1024)
    except (AttributeError, OSError, TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------


def _detect_gpus_nvidia() -> list[GPUInfo]:
    out = _run_probe(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version,utilization.gpu",
            "--format=csv,noheader",
        ]
    )
    if out is None:
        return []
    gpus: list[GPUInfo] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        name = parts[0].strip()
        mem_match = re.search(r"(\d+)", parts[1])
        vram_mb = int(mem_match.group(1)) if mem_match else None
        free_match = re.search(r"(\d+)", parts[2]) if len(parts) >= 3 else None
        vram_free_mb = int(free_match.group(1)) if free_match else None
        driver_version = parts[3].strip() if len(parts) >= 4 and parts[3].strip() else None
        gpus.append(
            GPUInfo(
                vendor="nvidia",
                name=name,
                vram_mb=vram_mb,
                method="nvidia-smi",
                vram_free_mb=vram_free_mb,
                vram_free_method="nvidia-smi" if vram_free_mb is not None else None,
                vram_free_at=(datetime.now(UTC).isoformat() if vram_free_mb is not None else None),
                driver_version=driver_version,
            )
        )
    return gpus


def _detect_gpus_amd_rocm_smi() -> list[GPUInfo]:
    out = _run_probe(["rocm-smi", "--showmeminfo", "vram"])
    if out is None:
        return []
    gpus: list[GPUInfo] = []
    for match in re.finditer(
        r"GPU\[(\d+)\].*?VRAM Total Memory \(B\):\s*(\d+)", out, re.IGNORECASE | re.DOTALL
    ):
        index, total_bytes = match.groups()
        vram_mb = int(total_bytes) // (1024 * 1024)
        gpus.append(
            GPUInfo(vendor="amd", name=f"AMD GPU {index}", vram_mb=vram_mb, method="rocm-smi")
        )
    return gpus


def _detect_gpus_amd_sysfs(drm_dir: Path | None = None) -> list[GPUInfo]:
    gpus: list[GPUInfo] = []
    if drm_dir is None:
        drm_dir = Path("/sys/class/drm")
    if not drm_dir.is_dir():
        return gpus
    for card_dir in sorted(drm_dir.glob("card[0-9]*")):
        vram_file = card_dir / "device" / "mem_info_vram_total"
        if not vram_file.is_file():
            continue
        try:
            total_bytes = int(vram_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        used_file = card_dir / "device" / "mem_info_vram_used"
        used_bytes: int | None = None
        with suppress(OSError, ValueError):
            used_bytes = int(used_file.read_text(encoding="utf-8").strip())
        free_mb = (
            max(0, total_bytes - used_bytes) // (1024 * 1024) if used_bytes is not None else None
        )
        gpus.append(
            GPUInfo(
                vendor="amd",
                name=card_dir.name,
                vram_mb=total_bytes // (1024 * 1024),
                method="sysfs:mem_info_vram_total",
                vram_free_mb=free_mb,
                vram_free_method="sysfs:mem_info_vram_used" if free_mb is not None else None,
                vram_free_at=datetime.now(UTC).isoformat() if free_mb is not None else None,
            )
        )
    return gpus


def _detect_gpus_amd() -> list[GPUInfo]:
    return _detect_gpus_amd_rocm_smi() or _detect_gpus_amd_sysfs()


def _parse_apple_vram_mb(value: object) -> int | None:
    """Parse a system_profiler display-memory value into MiB."""
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(GB|MB)\s*", value, re.IGNORECASE)
    if match is None:
        return None
    amount = float(match.group(1))
    multiplier = 1024 if match.group(2).upper() == "GB" else 1
    return round(amount * multiplier)


def _detect_gpus_apple(ram_mb: int, arch: str) -> list[GPUInfo]:
    out = _run_probe(["system_profiler", "SPDisplaysDataType", "-json"])
    if out is None:
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    displays = data.get("SPDisplaysDataType", [])
    if not isinstance(displays, list):
        return []
    gpus: list[GPUInfo] = []
    for entry in displays:
        if not isinstance(entry, dict):
            continue
        name = entry.get("sppci_model") or entry.get("_name") or "Apple GPU"
        vram_mb: int | None
        if arch == "arm64":
            vram_mb = round(0.75 * ram_mb)
            method = "estimate"
        else:
            raw_vram = entry.get("spdisplays_vram") or entry.get("spdisplays_vram_shared")
            vram_mb = _parse_apple_vram_mb(raw_vram)
            method = "system_profiler"
        gpus.append(GPUInfo(vendor="apple", name=str(name), vram_mb=vram_mb, method=method))
    return gpus


_LLAMA_BENCH_DEVICE_RE = re.compile(
    r"^\s*(?P<backend>\w+\d*):\s+(?P<name>.+?)\s+"
    r"\((?P<total>\d+)\s+MiB(?:,\s*(?P<free>\d+)\s+MiB\s+free)?\)\s*$"
)


def _gpu_vendor_from_name(name: str) -> str:
    lowered = name.casefold()
    if "nvidia" in lowered or "geforce" in lowered:
        return "nvidia"
    if "amd" in lowered or "radeon" in lowered:
        return "amd"
    return "unknown"


def _detect_gpus_llama_bench(bench_path: Path) -> list[GPUInfo]:
    """Detect GPUs via `llama-bench --list-devices` (NVK/Mesa fallback).

    llama.cpp enumerates the devices its own build supports, so this finds
    GPUs that vendor tooling misses (e.g. NVIDIA under NVK/Mesa). Lines look
    like `Vulkan0: NVIDIA GeForce RTX 5070 (NVK GB205) (12227 MiB, 2226 MiB
    free)`; the free-MiB clause is optional. Failure of any kind yields []
    and never raises.
    """
    out = _run_probe([str(bench_path), "--list-devices"])
    if out is None:
        return []
    gpus: list[GPUInfo] = []
    for line in out.splitlines():
        match = _LLAMA_BENCH_DEVICE_RE.match(line)
        if match is None:
            continue
        name = match.group("name").strip()
        vram_mb = int(match.group("total"))
        free_match = match.group("free")
        vram_free_mb = int(free_match) if free_match is not None else None
        gpus.append(
            GPUInfo(
                vendor=_gpu_vendor_from_name(name),
                name=name,
                vram_mb=vram_mb,
                method="llama-bench --list-devices",
                vram_free_mb=vram_free_mb,
                vram_free_method="llama-bench --list-devices" if vram_free_mb is not None else None,
                vram_free_at=(datetime.now(UTC).isoformat() if vram_free_mb is not None else None),
            )
        )
    return gpus


def _resolve_llama_bench(llama_bin: Path | None) -> Path | None:
    """Resolve llama-bench for GPU discovery, reusing llama.py's resolution."""
    from llamatune.llama import _resolve_binary

    return _resolve_binary("llama-bench", llama_bin)


def _detect_gpus(
    os_name: str, ram_mb: int, arch: str, llama_bench: Path | None = None
) -> tuple[list[GPUInfo], list[str]]:
    gpus: list[GPUInfo] = [*_detect_gpus_nvidia()]
    if os_name == "Windows":
        gpus.extend(_detect_gpus_windows_wmi({gpu.name for gpu in gpus}))
    else:
        gpus.extend(_detect_gpus_amd())
    if os_name == "Darwin":
        gpus.extend(_detect_gpus_apple(ram_mb, arch))
    if gpus:
        return gpus, []
    if llama_bench is None:
        return [], ["llama-bench not found; GPU fallback skipped"]
    llama_gpus = _detect_gpus_llama_bench(llama_bench)
    if llama_gpus:
        return llama_gpus, []
    return [], [
        "vendor GPU probes found no devices and 'llama-bench --list-devices' "
        "found none either; if this is a CPU-only llama.cpp build, GPU "
        "detection cannot see a GPU the build does not support"
    ]


def _detect_gpus_windows_wmi(existing_names: set[str] | None = None) -> list[GPUInfo]:
    """Detect non-NVIDIA Windows adapters without trusting WMI ``AdapterRAM``.

    ``AdapterRAM`` is a 32-bit field and cannot honestly represent modern
    adapters above 4 GiB. Unknown capacity is safer than a wrapped or saturated
    value because runtime placement probes remain the source of truth.
    """
    out = _run_probe(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }",
        ]
    )
    if out is None:
        return []
    existing = {name.casefold() for name in (existing_names or set())}
    gpus: list[GPUInfo] = []
    for line in out.splitlines():
        name = line.strip()
        lowered = name.casefold()
        if not name or lowered in existing or "nvidia" in lowered:
            continue
        vendor = "amd" if any(token in lowered for token in ("amd", "radeon")) else "intel"
        gpus.append(GPUInfo(vendor=vendor, name=name, vram_mb=None, method="wmi:name-only"))
    return gpus


def _sample_gpu_states_amd_sysfs(drm_dir: Path | None = None) -> tuple[GpuSample, ...]:
    if drm_dir is None:
        drm_dir = Path("/sys/class/drm")
    if not drm_dir.is_dir():
        return ()
    samples: list[GpuSample] = []
    for device_index, card_dir in enumerate(sorted(drm_dir.glob("card[0-9]*"))):
        device = card_dir / "device"
        try:
            total_bytes = int((device / "mem_info_vram_total").read_text().strip())
            used_bytes = int((device / "mem_info_vram_used").read_text().strip())
        except (OSError, ValueError):
            continue
        utilization: float = 0.0
        with suppress(OSError, ValueError):
            utilization = float((device / "gpu_busy_percent").read_text().strip())
        samples.append(
            GpuSample(
                ts=datetime.now(UTC).isoformat(),
                vram_used_mb=used_bytes // (1024 * 1024),
                vram_free_mb=max(0, total_bytes - used_bytes) // (1024 * 1024),
                utilization_pct=utilization,
                device_index=device_index,
            )
        )
    return tuple(samples)


def _sample_gpu_state_amd_sysfs(drm_dir: Path | None = None) -> GpuSample | None:
    """Compatibility wrapper returning the first AMD device sample."""
    samples = _sample_gpu_states_amd_sysfs(drm_dir)
    return samples[0] if samples else None


def sample_gpu_states() -> tuple[GpuSample, ...]:
    """Return bounded best-effort observations for every visible GPU."""
    out = _run_probe(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu,clocks.sm",
            "--format=csv,noheader,nounits",
        ]
    )
    if out is not None:
        samples: list[GpuSample] = []
        for device_index, line in enumerate(out.splitlines()):
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                samples.append(
                    GpuSample(
                        ts=datetime.now(UTC).isoformat(),
                        vram_used_mb=int(float(parts[0])),
                        vram_free_mb=int(float(parts[1])),
                        utilization_pct=float(parts[2]),
                        power_w=_optional_float(parts, 3),
                        temperature_c=_optional_float(parts, 4),
                        clocks_sm_mhz=_optional_float(parts, 5),
                        device_index=device_index,
                    )
                )
            except ValueError:
                continue
        if samples:
            return tuple(samples)
    return _sample_gpu_states_amd_sysfs()


def sample_gpu_state() -> GpuSample | None:
    """Compatibility wrapper returning the first visible GPU observation."""
    samples = sample_gpu_states()
    return samples[0] if samples else None


def _optional_float(parts: list[str], index: int) -> float | None:
    if index >= len(parts):
        return None
    try:
        return float(parts[index])
    except ValueError:
        return None


def detect_throttle(samples: list[GpuSample] | tuple[GpuSample, ...]) -> bool:
    """Detect a sustained >10% SM-clock decline between first/last quartiles."""
    clocks = [sample.clocks_sm_mhz for sample in samples if sample.clocks_sm_mhz is not None]
    if len(clocks) < 8:
        return False
    quartile = max(1, len(clocks) // 4)
    first = statistics.median(clocks[:quartile])
    last = statistics.median(clocks[-quartile:])
    return first > 0 and last < 0.9 * first


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------


def detect_load() -> tuple[float, float, float] | None:
    getloadavg = cast(
        Callable[[], tuple[float, float, float]] | None,
        getattr(os, "getloadavg", None),
    )
    if platform.system() == "Windows" or getloadavg is None:
        return None
    try:
        return getloadavg()
    except OSError:
        return None


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------


def assess_hardware(llama_bin: Path | None = None) -> HardwareReport:
    """Best-effort hardware and environment assessment (DESIGN §4).

    `llama_bin` is the optional `--llama-bin` directory; when given, the
    llama-bench GPU-detection fallback resolves the binary from it instead
    of PATH.
    """
    warnings: list[str] = []
    os_name = platform.system()
    arch = platform.machine()

    if os_name == "Darwin":
        cpu_model, physical_cores, logical_cores, perf_cores, _method = _detect_cpu_macos()
        ram_mb = _detect_memory_macos()
    elif os_name == "Windows":
        cpu_model, physical_cores, logical_cores, perf_cores, _method = _detect_cpu_windows()
        ram_mb = _detect_memory_windows()
        warnings.append("load monitoring is unavailable on Windows")
    else:
        if os_name != "Linux":
            warnings.append(f"unrecognized platform {os_name!r}; using Linux-style probes")
        cpu_model, physical_cores, logical_cores, perf_cores, _method = _detect_cpu_linux()
        ram_mb = _detect_memory_linux()

    gpus, gpu_warnings = _detect_gpus(
        os_name, ram_mb, arch, llama_bench=_resolve_llama_bench(llama_bin)
    )
    warnings.extend(gpu_warnings)

    load = detect_load()
    if load is not None and physical_cores > 0 and load[0] > physical_cores / 2:
        warnings.append(
            f"1-minute load average {load[0]:.2f} exceeds physical_cores/2 "
            f"({physical_cores / 2:.1f})"
        )

    return HardwareReport(
        os_name=os_name,
        arch=arch,
        cpu_model=cpu_model,
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        perf_cores=perf_cores,
        ram_mb=ram_mb,
        gpus=tuple(gpus),
        warnings=tuple(warnings),
    )
