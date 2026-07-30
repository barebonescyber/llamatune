"""Tests for release-only validation scripts."""

from __future__ import annotations

import hashlib
import re
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

SCRIPTS = Path(__file__).parents[2] / ".github" / "scripts"
WORKFLOWS = Path(__file__).parents[2] / ".github" / "workflows"

coverage_validate = cast(
    Callable[[object], tuple[tuple[str, float], ...]],
    runpy.run_path(str(SCRIPTS / "check_module_coverage.py"))["validate"],
)
verify_sha256s = cast(
    Callable[[Path], tuple[Path, ...]],
    runpy.run_path(str(SCRIPTS / "verify_sha256s.py"))["verify"],
)
core_modules = cast(
    tuple[str, ...],
    runpy.run_path(str(SCRIPTS / "check_module_coverage.py"))["CORE_MODULES"],
)


def _coverage_document(percent: float = 80.0) -> dict[str, Any]:
    return {
        "files": {
            f"src/llamatune/{module}": {"summary": {"percent_covered": percent}}
            for module in core_modules
        }
    }


def test_core_module_coverage_accepts_exact_floor() -> None:
    results = coverage_validate(_coverage_document())
    assert len(results) == len(core_modules)
    assert {percent for _, percent in results} == {80.0}


def test_core_module_coverage_reports_missing_and_low_modules() -> None:
    document = _coverage_document()
    del document["files"]["src/llamatune/bench.py"]
    document["files"]["src/llamatune/cli.py"]["summary"]["percent_covered"] = 79.99
    with pytest.raises(
        ValueError,
        match=r"(?s)bench\.py: missing.*cli\.py: 79\.99%",
    ):
        coverage_validate(document)


def test_checksum_manifest_verifies_exact_files(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.whl"
    artifact.write_bytes(b"wheel")
    digest = hashlib.sha256(b"wheel").hexdigest()
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(f"{digest}  artifact.whl\n", encoding="utf-8")
    assert verify_sha256s(manifest) == (artifact,)


@pytest.mark.parametrize(
    "line",
    [
        "not-a-checksum  artifact.whl\n",
        f"{'0' * 64}  ../artifact.whl\n",
        f"{'0' * 64}  missing.whl\n",
    ],
)
def test_checksum_manifest_rejects_invalid_entries(tmp_path: Path, line: str) -> None:
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(line, encoding="utf-8")
    with pytest.raises(ValueError):
        verify_sha256s(manifest)


def test_workflow_checkouts_do_not_persist_credentials() -> None:
    secured_checkout = re.compile(
        r"(?m)^(?P<indent>\s*)- uses: actions/checkout@[^\n]+\n"
        r"(?P=indent)  with:\n"
        r"(?P=indent)    persist-credentials: false(?:\n|$)"
    )
    for workflow in sorted(WORKFLOWS.glob("*.yml")):
        text = workflow.read_text(encoding="utf-8")
        checkout_count = text.count("- uses: actions/checkout@")
        assert len(secured_checkout.findall(text)) == checkout_count, (
            f"{workflow} has a checkout step that persists credentials"
        )
