"""Enforce branch coverage for every module in the beta-supported core."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MINIMUM_PERCENT = 80.0

CORE_MODULES = (
    "bench.py",
    "cli.py",
    "config.py",
    "coverage.py",
    "discovery.py",
    "executor.py",
    "hardware.py",
    "llama.py",
    "matrix.py",
    "matrixquery.py",
    "matrixreport.py",
    "model.py",
    "recommend.py",
    "registry.py",
    "report.py",
    "resultsmatrix.py",
    "search.py",
    "session.py",
    "stats.py",
    "types.py",
    "ui.py",
)


def _coverage_percent(entry: object, module: str) -> float:
    if not isinstance(entry, dict):
        raise ValueError(f"coverage entry for {module} is not an object")
    summary = entry.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"coverage entry for {module} omits summary")
    value = summary.get("percent_covered")
    if not isinstance(value, (int, float)):
        raise ValueError(f"coverage entry for {module} omits numeric percent_covered")
    return float(value)


def validate(document: object) -> tuple[tuple[str, float], ...]:
    if not isinstance(document, dict):
        raise ValueError("coverage document is not an object")
    files = document.get("files")
    if not isinstance(files, dict):
        raise ValueError("coverage document omits files")

    results: list[tuple[str, float]] = []
    failures: list[str] = []
    for module in CORE_MODULES:
        path = f"src/llamatune/{module}"
        if path not in files:
            failures.append(f"{path}: missing from coverage report")
            continue
        percent = _coverage_percent(files[path], module)
        results.append((path, percent))
        if percent + 1e-9 < MINIMUM_PERCENT:
            failures.append(f"{path}: {percent:.2f}% < {MINIMUM_PERCENT:.2f}%")

    if failures:
        raise ValueError("core module coverage gate failed:\n" + "\n".join(failures))
    return tuple(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("coverage_json", type=Path)
    args = parser.parse_args()
    document: Any = json.loads(args.coverage_json.read_text(encoding="utf-8"))
    results = validate(document)
    for path, percent in results:
        print(f"{path}: {percent:.2f}%")
    print(f"core module coverage: {len(results)} modules at or above {MINIMUM_PERCENT:.2f}%")


if __name__ == "__main__":
    main()
