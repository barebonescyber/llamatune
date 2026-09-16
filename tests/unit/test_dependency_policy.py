import tomllib
from pathlib import Path


def test_runtime_dependency_floors_match_supported_baseline() -> None:
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["requires-python"] == ">=3.11"
    assert set(project["project"]["dependencies"]) == {
        "typer>=0.27.0,<1",
        "gguf>=0.19.0,<1",
        "rich>=15.0.0,<16",
    }
