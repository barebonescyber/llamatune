from pathlib import Path


def test_security_audit_has_unprivileged_pr_trigger() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / ".github/workflows/security.yml").read_text(encoding="utf-8")
    assert "on:\n  pull_request:\n  workflow_dispatch:" in text
    assert "permissions:\n  contents: read\n" in text
    assert "pull_request_target" not in text
    assert "id-token: write" not in text
    assert "persist-credentials: false" in text
    assert "timeout-minutes: 15" in text
    assert "enable-cache: false" in text
    assert "github.event_name == 'workflow_dispatch' && inputs.run_codeql" in text
