"""Enforce the Phase 3 baseline contract for built release artifacts."""

from __future__ import annotations

import argparse
import email.policy
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

REQUIRED_RUNTIME_FILES = {
    "llamatune/__init__.py",
    "llamatune/cli.py",
    "llamatune/evaltasks/agentic.json",
    "llamatune/evaltasks/coding.json",
    "llamatune/evaltasks/filler.txt",
    "llamatune/evaltasks/ifollow.json",
    "llamatune/evaltasks/tooluse.json",
}

ALLOWED_RUNTIME_DATA = {
    "llamatune/evaltasks/agentic.json",
    "llamatune/evaltasks/coding.json",
    "llamatune/evaltasks/filler.txt",
    "llamatune/evaltasks/ifollow.json",
    "llamatune/evaltasks/tooluse.json",
}

REQUIRED_SDIST_FILES = {
    "CHANGELOG.md",
    "DESIGN.md",
    "LICENSE",
    "README.md",
    "SECURITY.md",
    "SUPPORT.md",
    "docs/beta-acceptance-checklist.md",
    "docs/beta-contract.md",
    "docs/compatibility-policy.md",
    "pyproject.toml",
    "src/llamatune/__init__.py",
    "src/llamatune/cli.py",
    "src/llamatune/evaltasks/agentic.json",
    "uv.lock",
}

ALLOWED_SDIST_ROOT_FILES = {
    ".gitignore",
    "CHANGELOG.md",
    "DESIGN.md",
    "LICENSE",
    "PKG-INFO",
    "README.md",
    "SECURITY.md",
    "SUPPORT.md",
    "pyproject.toml",
    "uv.lock",
}

ALLOWED_SDIST_DOCS = {
    "docs/beta-acceptance-checklist.md",
    "docs/beta-contract.md",
    "docs/compatibility-policy.md",
}

FORBIDDEN_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "llamatune-sessions",
}

FORBIDDEN_SUFFIXES = {".gguf", ".pyc", ".session"}

REQUIRED_CLASSIFIERS = {
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: MIT License",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.14",
}

REQUIRED_PROJECT_URLS = {"Changelog", "Documentation", "Issues", "Repository"}


def _validate_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"unsafe archive member: {name}")
    return path


def _validate_forbidden(relative_names: set[str]) -> None:
    for name in sorted(relative_names):
        path = _validate_member(name)
        if FORBIDDEN_PARTS.intersection(path.parts):
            raise ValueError(f"generated or local directory included in release artifact: {name}")
        if path.suffix in FORBIDDEN_SUFFIXES:
            raise ValueError(f"generated or model file included in release artifact: {name}")


def _project_metadata() -> dict[str, object]:
    repository_root = Path(__file__).resolve().parents[2]
    with (repository_root / "pyproject.toml").open("rb") as stream:
        document = tomllib.load(stream)
    project = document.get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml omits [project] metadata")
    return project


def _validate_wheel_metadata(raw: bytes) -> None:
    expected = _project_metadata()
    metadata = BytesParser(policy=email.policy.default).parsebytes(raw)
    expected_name = expected.get("name")
    expected_version = expected.get("version")
    if metadata["Name"] != expected_name:
        raise ValueError(f"wheel name {metadata['Name']!r} does not match {expected_name!r}")
    if metadata["Version"] != expected_version:
        raise ValueError(
            f"wheel version {metadata['Version']!r} does not match {expected_version!r}"
        )
    if metadata["Requires-Python"] != expected.get("requires-python"):
        raise ValueError("wheel Requires-Python does not match pyproject.toml")
    if not metadata["Maintainer"]:
        raise ValueError("wheel metadata omits Maintainer")
    if not metadata["Keywords"]:
        raise ValueError("wheel metadata omits Keywords")

    classifiers = set(metadata.get_all("Classifier", []))
    missing_classifiers = sorted(REQUIRED_CLASSIFIERS - classifiers)
    if missing_classifiers:
        raise ValueError(
            f"wheel metadata omits required classifiers: {', '.join(missing_classifiers)}"
        )

    project_urls = {
        value.split(",", 1)[0].strip()
        for value in metadata.get_all("Project-URL", [])
        if "," in value
    }
    missing_urls = sorted(REQUIRED_PROJECT_URLS - project_urls)
    if missing_urls:
        raise ValueError(f"wheel metadata omits project URLs: {', '.join(missing_urls)}")


def inspect_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = {item.filename.rstrip("/") for item in archive.infolist() if not item.is_dir()}
        for name in names:
            _validate_member(name)
        _validate_forbidden(names)

        missing = sorted(REQUIRED_RUNTIME_FILES - names)
        if missing:
            raise ValueError(f"wheel omits required runtime files: {', '.join(missing)}")

        dist_info_roots = {PurePosixPath(name).parts[0] for name in names if ".dist-info/" in name}
        if len(dist_info_roots) != 1:
            raise ValueError(
                f"expected one wheel .dist-info directory, found {len(dist_info_roots)}"
            )
        dist_info = next(iter(dist_info_roots))
        required_metadata = {
            f"{dist_info}/METADATA",
            f"{dist_info}/RECORD",
            f"{dist_info}/WHEEL",
            f"{dist_info}/entry_points.txt",
        }
        missing_metadata = sorted(required_metadata - names)
        if missing_metadata:
            raise ValueError(f"wheel omits required metadata: {', '.join(missing_metadata)}")

        unexpected_runtime = sorted(
            name
            for name in names
            if name.startswith("llamatune/")
            and not name.endswith(".py")
            and name not in ALLOWED_RUNTIME_DATA
        )
        if unexpected_runtime:
            raise ValueError(
                f"wheel contains runtime files outside the public allowlist: "
                f"{', '.join(unexpected_runtime)}"
            )

        unexpected_roots = {
            PurePosixPath(name).parts[0] for name in names if not name.startswith("llamatune/")
        } - {dist_info}
        if unexpected_roots:
            raise ValueError(
                f"wheel contains unexpected top-level paths: {sorted(unexpected_roots)}"
            )
        allowed_dist_info = required_metadata | {f"{dist_info}/licenses/LICENSE"}
        unexpected_metadata = sorted(
            name
            for name in names
            if name.startswith(f"{dist_info}/") and name not in allowed_dist_info
        )
        if unexpected_metadata:
            raise ValueError(
                f"wheel contains metadata files outside the public allowlist: "
                f"{', '.join(unexpected_metadata)}"
            )
        _validate_wheel_metadata(archive.read(f"{dist_info}/METADATA"))

    print(f"wheel: {path} ({len(names)} files)")
    for name in sorted(names):
        print(f"  {name}")


def inspect_sdist(path: Path) -> None:
    with tarfile.open(path, mode="r:gz") as archive:
        unsupported = [
            item.name for item in archive.getmembers() if not (item.isfile() or item.isdir())
        ]
        if unsupported:
            raise ValueError(
                f"source distribution contains links or special files: {', '.join(unsupported)}"
            )
        raw_names = {item.name.rstrip("/") for item in archive.getmembers() if item.isfile()}
    validated = {_validate_member(name) for name in raw_names}
    roots = {name.parts[0] for name in validated if name.parts}
    if len(roots) != 1:
        raise ValueError(f"expected one source-distribution root, found {sorted(roots)}")
    root = next(iter(roots))
    names = {str(PurePosixPath(*name.parts[1:])) for name in validated}
    _validate_forbidden(names)

    missing = sorted(REQUIRED_SDIST_FILES - names)
    if missing:
        raise ValueError(f"source distribution omits required files: {', '.join(missing)}")

    unexpected = sorted(
        name
        for name in names
        if name not in ALLOWED_SDIST_ROOT_FILES
        and name not in ALLOWED_SDIST_DOCS
        and not name.startswith("src/")
    )
    if unexpected:
        raise ValueError(
            f"source distribution contains paths outside the public allowlist: "
            f"{', '.join(unexpected)}"
        )

    print(f"source distribution: {path} ({len(names)} files, root {root})")
    for name in sorted(names):
        print(f"  {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("sdist", type=Path)
    args = parser.parse_args()

    if args.wheel.suffix != ".whl":
        raise ValueError(f"expected a wheel, got {args.wheel}")
    if not args.sdist.name.endswith(".tar.gz"):
        raise ValueError(f"expected a .tar.gz source distribution, got {args.sdist}")
    inspect_wheel(args.wheel)
    inspect_sdist(args.sdist)


if __name__ == "__main__":
    main()
