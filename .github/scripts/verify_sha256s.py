"""Verify the deterministic SHA-256 manifest for release archives."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(manifest: Path) -> tuple[Path, ...]:
    checked: list[Path] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line:
            continue
        try:
            expected, marker_and_name = raw_line.split("  ", 1)
        except ValueError as error:
            raise ValueError(f"{manifest}:{line_number}: malformed checksum line") from error
        if len(expected) != 64 or any(
            character not in "0123456789abcdef" for character in expected
        ):
            raise ValueError(f"{manifest}:{line_number}: invalid SHA-256")
        if marker_and_name.startswith(("*", " ")):
            raise ValueError(f"{manifest}:{line_number}: unsupported filename marker")
        name = marker_and_name
        if name in seen:
            raise ValueError(f"{manifest}:{line_number}: duplicate filename {name!r}")
        seen.add(name)
        path = manifest.parent / name
        if path.parent != manifest.parent or path.is_symlink() or not path.is_file():
            raise ValueError(f"{manifest}:{line_number}: unsafe or missing file {name!r}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"{name}: checksum mismatch ({actual} != {expected})")
        checked.append(path)
    if not checked:
        raise ValueError(f"{manifest}: no checksums")
    return tuple(checked)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    checked = verify(args.manifest)
    for path in checked:
        print(f"{path.name}: OK")


if __name__ == "__main__":
    main()
