"""Fail when a detect-secrets JSON report contains findings."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_secret_scan.py REPORT.json", file=sys.stderr)
        return 2
    report_path = Path(argv[1])
    try:
        report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        results = report["results"]
        if not isinstance(results, dict):
            raise TypeError("results must be an object")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"invalid detect-secrets report: {exc}", file=sys.stderr)
        return 2

    findings = 0
    for filename, records in sorted(results.items()):
        if not isinstance(records, list):
            print(f"invalid findings list for {filename}", file=sys.stderr)
            return 2
        for record in records:
            if not isinstance(record, dict):
                print(f"invalid finding for {filename}", file=sys.stderr)
                return 2
            findings += 1
            print(
                f"{filename}:{record.get('line_number', '?')}: "
                f"{record.get('type', 'potential secret')}",
                file=sys.stderr,
            )
    if findings:
        print(f"detect-secrets found {findings} potential secret(s)", file=sys.stderr)
        return 1
    print("detect-secrets found no potential secrets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
