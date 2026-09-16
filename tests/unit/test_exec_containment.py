from pathlib import Path
from typing import Any, cast

import pytest

from llamatune import sandbox


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_run_python_refuses_before_spawn(
    platform: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sandbox_module = cast(Any, sandbox)
    monkeypatch.setattr(sandbox_module.sys, "platform", platform)
    monkeypatch.setattr(
        sandbox_module.subprocess, "Popen", lambda *a, **k: pytest.fail("child started")
    )
    monkeypatch.setattr(
        sandbox_module.tempfile, "mkdtemp", lambda *a, **k: pytest.fail("scratch created")
    )
    with pytest.raises(ValueError, match="generated-code execution is disabled"):
        sandbox.run_python("pass", timeout_s=1.0)
    assert list(tmp_path.iterdir()) == []
