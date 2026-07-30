from llamatune.marathonreport import render


def test_report_has_every_required_section_and_non_replication_label() -> None:
    text = render(
        {
            "stop_reason": "rounds_max",
            "rounds": [],
            "champion": {"config": {"gpu_layers": 4}, "pp": 2, "tg": 3},
            "coverage": {
                "tiers": {"A": {"enumerated": 1, "executed": 1, "pruned": 0, "remaining": 0}},
                "responsive": ["ubatch"],
            },
            "matrix": [],
            "verification": {"verdict": "tie"},
            "warnings": ["warmup drift"],
        }
    )
    for heading in (
        "Summary",
        "Champion",
        "Round history",
        "Coverage",
        "Operating points",
        "Environment",
        "A/B verification",
        "Deferred and failed",
        "Warnings",
    ):
        assert f"## {heading}" in text
    assert "NOT REPLICATED" in text


def test_report_tolerates_absent_phases() -> None:
    assert "# llamatune Marathon report" in render({})
