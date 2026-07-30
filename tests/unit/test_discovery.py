"""Night Shift model-discovery contracts."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from llamatune import discovery
from llamatune.model import inspect_model
from llamatune.types import ModelReport


def test_nested_discovery_and_identical_fingerprint_dedup(tmp_path: Path, tiny_gguf: Path) -> None:
    models = tmp_path / "models"
    nested = models / "nested"
    nested.mkdir(parents=True)
    first = models / "a.gguf"
    second = nested / "b.gguf"
    shutil.copy2(tiny_gguf, first)
    shutil.copy2(tiny_gguf, second)

    with pytest.warns(RuntimeWarning, match="deduplicating"):
        found = discovery.discover_models(models, (), ())

    assert [model.path for model in found] == [first]
    assert found[0].representative


def test_include_then_exclude_filters_basename(tmp_path: Path, tiny_gguf: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    keep = models / "keep-Q4.gguf"
    excluded = models / "skip-Q4.gguf"
    shutil.copy2(tiny_gguf, keep)
    shutil.copy2(tiny_gguf, excluded)

    found = discovery.discover_models(models, ("*-Q4.gguf",), ("skip-*",))
    assert [model.path for model in found] == [keep]


def test_incomplete_shard_group_is_skipped(tmp_path: Path, tiny_gguf: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    shutil.copy2(tiny_gguf, models / "model-00001-of-00003.gguf")
    shutil.copy2(tiny_gguf, models / "model-00003-of-00003.gguf")

    with pytest.warns(RuntimeWarning, match="model-00002-of-00003[.]gguf"):
        assert discovery.discover_models(models, (), ()) == ()


def test_incomplete_unpadded_shard_group_uses_natural_index_width(
    tmp_path: Path, tiny_gguf: Path
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    shutil.copy2(tiny_gguf, models / "model-1-of-10.gguf")
    shutil.copy2(tiny_gguf, models / "model-10-of-10.gguf")

    with pytest.warns(RuntimeWarning) as caught:
        assert discovery.discover_models(models, (), ()) == ()

    message = str(caught[0].message)
    assert "model-2-of-10.gguf" in message
    assert "model-02-of-10.gguf" not in message


def test_complete_shards_expose_only_head(tmp_path: Path, tiny_gguf: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    paths = tuple(models / f"model-{index:05d}-of-00002.gguf" for index in (1, 2))
    for path in paths:
        shutil.copy2(tiny_gguf, path)

    found = discovery.discover_models(models, (), ())
    assert len(found) == 1
    assert found[0].path == paths[0]
    assert found[0].shard_paths == paths


def test_four_digit_shards_are_one_numeric_group(tmp_path: Path, tiny_gguf: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    paths = tuple(models / f"model-{index:04d}-of-0002.gguf" for index in (1, 2))
    for path in paths:
        shutil.copy2(tiny_gguf, path)

    found = discovery.discover_models(models, (), ())
    assert len(found) == 1
    assert found[0].path == paths[0]
    assert found[0].shard_paths == paths


def test_variable_width_shards_are_sorted_numerically(tmp_path: Path, tiny_gguf: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    paths = tuple(models / f"model-{index}-of-10.gguf" for index in range(1, 11))
    for path in reversed(paths):
        shutil.copy2(tiny_gguf, path)

    found = discovery.discover_models(models, (), ())
    assert len(found) == 1
    assert found[0].shard_paths == paths


def test_mixed_width_groups_remain_separate(tmp_path: Path, tiny_gguf: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    four_digit = tuple(models / f"four-{index:04d}-of-0002.gguf" for index in (1, 2))
    five_digit = tuple(models / f"five-{index:05d}-of-00002.gguf" for index in (1, 2))
    for offset, path in enumerate((*four_digit, *five_digit)):
        shutil.copy2(tiny_gguf, path)
        with path.open("ab") as handle:
            handle.write(bytes([offset]))

    found = discovery.discover_models(models, (), ())
    assert [model.path for model in found] == [five_digit[0], four_digit[0]]
    assert [model.shard_paths for model in found] == [five_digit, four_digit]


def test_underscore_separator_is_a_shard_group(tmp_path: Path, tiny_gguf: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    paths = tuple(models / f"model_{index:05d}-of-00002.gguf" for index in (1, 2))
    for path in paths:
        shutil.copy2(tiny_gguf, path)

    found = discovery.discover_models(models, (), ())
    assert len(found) == 1
    assert found[0].path == paths[0]
    assert found[0].shard_paths == paths


def test_incomplete_four_digit_group_warns_and_is_skipped(tmp_path: Path, tiny_gguf: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    shutil.copy2(tiny_gguf, models / "foo-0001-of-0002.gguf")

    with pytest.warns(RuntimeWarning, match="foo-0002-of-0002"):
        assert discovery.discover_models(models, (), ()) == ()


def test_mixed_declared_totals_warn_and_are_skipped(tmp_path: Path, tiny_gguf: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    shutil.copy2(tiny_gguf, models / "foo-0001-of-0002.gguf")
    shutil.copy2(tiny_gguf, models / "foo-0002-of-0003.gguf")

    with pytest.warns(RuntimeWarning, match="mixed declared totals"):
        assert discovery.discover_models(models, (), ()) == ()


def test_symlink_directory_cycle_does_not_duplicate_or_hang(
    tmp_path: Path, tiny_gguf: Path
) -> None:
    models = tmp_path / "models"
    nested = models / "nested"
    nested.mkdir(parents=True)
    shutil.copy2(tiny_gguf, nested / "model.gguf")
    try:
        (nested / "cycle").symlink_to(models, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable")
    found = discovery.discover_models(models, (), ())
    assert len(found) == 1


def test_duplicate_policy_marks_one_or_both_representatives(
    tmp_path: Path, tiny_gguf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    shard = models / "same-00001-of-00001.gguf"
    merged = models / "same-merged.gguf"
    shutil.copy2(tiny_gguf, shard)
    shutil.copy2(tiny_gguf, merged)
    with merged.open("ab") as handle:
        handle.write(b"x")
    monkeypatch.setattr(discovery, "_tensor_table_sha", lambda _paths: "a" * 64)

    one = discovery.discover_models(models, (), (), duplicates="one")
    assert [model.path for model in one if model.representative] == [merged]
    assert {model.group_key for model in one} == {"a" * 16}
    both = discovery.discover_models(models, (), (), duplicates="both")
    assert all(model.representative for model in both)


def test_zero_byte_decoy_is_warning_and_skip(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "broken.gguf").touch()
    with pytest.warns(RuntimeWarning, match="unreadable GGUF"):
        assert discovery.discover_models(models, (), ()) == ()


def test_model_removed_after_inspection_is_warned_and_skipped(
    tmp_path: Path, tiny_gguf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    vanished = models / "a-vanished.gguf"
    retained = models / "b-retained.gguf"
    shutil.copy2(tiny_gguf, vanished)
    shutil.copy2(tiny_gguf, retained)

    def inspect_then_remove(path: Path, *, full_hash: bool = False) -> ModelReport:
        report = inspect_model(path, full_hash=full_hash)
        if path == vanished:
            path.unlink()
        return report

    monkeypatch.setattr(discovery, "inspect_model", inspect_then_remove)

    with pytest.warns(RuntimeWarning, match="skipping unreadable GGUF.*a-vanished"):
        found = discovery.discover_models(models, (), ())

    assert [model.path for model in found] == [retained]


def _layout_pair(tmp_path: Path, tiny_gguf: Path) -> tuple[Path, Path, Path]:
    models = tmp_path / "models"
    models.mkdir()
    shard = models / "same-00001-of-00001.gguf"
    merged = models / "same-merged.gguf"
    shutil.copy2(tiny_gguf, shard)
    shutil.copy2(tiny_gguf, merged)
    with merged.open("ab") as handle:
        handle.write(b"x")
    return models, shard, merged


def test_different_tensor_table_or_type_is_not_grouped(
    tmp_path: Path, tiny_gguf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models, _shard, merged = _layout_pair(tmp_path, tiny_gguf)
    monkeypatch.setattr(
        discovery,
        "_tensor_table_sha",
        lambda paths: "a" * 64 if paths[0] == merged else "b" * 64,
    )
    found = discovery.discover_models(models, (), ())
    assert len(found) == 2
    assert all(model.group_key is None and model.representative for model in found)


def test_unavailable_tensor_table_is_not_grouped(
    tmp_path: Path, tiny_gguf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models, _shard, merged = _layout_pair(tmp_path, tiny_gguf)
    monkeypatch.setattr(
        discovery,
        "_tensor_table_sha",
        lambda paths: None if paths[0] == merged else "a" * 64,
    )
    found = discovery.discover_models(models, (), ())
    assert len(found) == 2
    assert all(model.group_key is None and model.representative for model in found)


def test_payload_size_difference_over_one_percent_is_not_grouped(
    tmp_path: Path, tiny_gguf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models, _shard, merged = _layout_pair(tmp_path, tiny_gguf)
    with merged.open("ab") as handle:
        handle.write(b"padding" * max(1, tiny_gguf.stat().st_size // 50))
    monkeypatch.setattr(discovery, "_tensor_table_sha", lambda _paths: "a" * 64)
    found = discovery.discover_models(models, (), ())
    assert len(found) == 2
    assert all(model.group_key is None and model.representative for model in found)


def test_excluding_merged_shifts_representative_to_shard_head(
    tmp_path: Path, tiny_gguf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models, shard, _merged = _layout_pair(tmp_path, tiny_gguf)
    monkeypatch.setattr(discovery, "_tensor_table_sha", lambda _paths: "a" * 64)
    found = discovery.discover_models(models, (), ("*-merged.gguf",))
    assert len(found) == 1
    assert found[0].path == shard
    assert found[0].representative
