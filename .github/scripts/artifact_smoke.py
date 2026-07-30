"""Create and verify fixtures for the installed-wheel CI smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def create_model(path: Path) -> None:
    """Write the smallest useful GGUF fixture with wheel-installed dependencies."""
    import gguf
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_block_count(32)
    writer.add_name("artifact-smoke-model")
    writer.add_tensor("dummy.weight", np.zeros((4,), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def verify(root: Path) -> None:
    """Verify CLI output and evidence produced exclusively by the installed wheel."""
    root = root.resolve()
    venv = (root / "venv").resolve()
    module_path = Path((root / "module-path.txt").read_text(encoding="utf-8").strip()).resolve()
    if not module_path.is_relative_to(venv):
        raise ValueError(f"llamatune imported outside the clean environment: {module_path}")

    if "Usage:" not in (root / "help.txt").read_text(encoding="utf-8"):
        raise ValueError("installed `llamatune --help` output is incomplete")

    scan = _load_object(root / "scan.json")
    if scan.get("llama_error") is not None:
        raise ValueError(f"installed scan failed to discover fake llama.cpp: {scan['llama_error']}")
    if not isinstance(scan.get("hardware"), dict) or not isinstance(scan.get("llama"), dict):
        raise ValueError("installed scan JSON lacks hardware or llama.cpp evidence")

    tune = _load_object(root / "tune.json")
    if not isinstance(tune.get("baseline"), dict):
        raise ValueError("installed tune JSON lacks baseline evidence")
    tune_exit = int((root / "tune-exit.txt").read_text(encoding="utf-8").strip())
    if tune_exit not in {0, 1}:
        raise ValueError(f"installed tune returned fatal exit code {tune_exit}")

    session_dirs = [
        path for path in (root / "sessions").iterdir() if (path / "session.json").is_file()
    ]
    if len(session_dirs) != 1:
        raise ValueError(f"expected one installed-wheel session, found {len(session_dirs)}")

    required_evidence = {
        "analysis.json",
        "hardware.json",
        "journal.jsonl",
        "llamacpp.json",
        "model.json",
        "recommended.json",
        "recommended.sh",
        "report.md",
        "session.json",
    }
    missing = sorted(name for name in required_evidence if not (session_dirs[0] / name).is_file())
    if missing:
        raise ValueError(f"installed tune omitted required evidence: {', '.join(missing)}")

    analysis = _load_object(session_dirs[0] / "analysis.json")
    if analysis != tune:
        raise ValueError("installed tune stdout does not match its analysis.json evidence")

    print(f"installed module: {module_path}")
    print(f"validated session: {session_dirs[0]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create-model")
    create_parser.add_argument("path", type=Path)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("root", type=Path)
    args = parser.parse_args()

    if args.command == "create-model":
        create_model(args.path)
    else:
        verify(args.root)


if __name__ == "__main__":
    main()
