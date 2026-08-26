"""Guard rails for CLI help coverage and experimental labeling (#23, UX-012)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from typer.core import TyperGroup
from typer.main import get_command

from llamatune.cli import app

_EXPERIMENTAL_PREFIX = "Experimental: "

# (command path, primary option spelling) for every flag whose help must carry
# the experimental prefix. tune --allow-lossy/--multi-gpu/--quality-corpus were
# labeled before #23; the rest were labeled by #32/UX-012.
_EXPERIMENTAL_PARAMS = (
    ("tune", "--allow-lossy"),
    ("tune", "--multi-gpu"),
    ("tune", "--quality-corpus"),
    ("tune", "--ot-search"),
    ("nightshift", "--allow-lossy"),
    ("marathon", "--allow-lossy"),
    ("marathon", "--quality-corpus"),
    ("marathon", "--ot-search"),
)


def _iter_commands(command: Any) -> Iterator[Any]:
    yield command
    for subcommand in getattr(command, "commands", {}).values():
        yield from _iter_commands(subcommand)


def _param_label(param: Any) -> str:
    opts = tuple(getattr(param, "opts", ()) or ())
    return "/".join(opts) if opts else str(getattr(param, "name", "<argument>"))


def _all_params() -> list[tuple[str, str, Any]]:
    root = get_command(app)
    assert isinstance(root, TyperGroup)
    entries: list[tuple[str, str, Any]] = []
    for command in _iter_commands(root):
        for param in command.params:
            label = _param_label(param)
            entries.append((str(command.name), label, param))
    return entries


_ALL_PARAMS = _all_params()


def _find_param(command_name: str, option: str) -> Any:
    for name, label, param in _ALL_PARAMS:
        if name == command_name and label == option:
            return param
    msg = f"no parameter {option} on command {command_name}"
    raise AssertionError(msg)


def test_cli_exposes_parameters() -> None:
    assert len(_ALL_PARAMS) > 100


@pytest.mark.parametrize(
    ("command_name", "label", "param"),
    _ALL_PARAMS,
    ids=[f"{name}:{label}" for name, label, _ in _ALL_PARAMS],
)
def test_every_option_and_argument_has_help(command_name: str, label: str, param: Any) -> None:
    help_text = param.help
    assert isinstance(help_text, str), f"{command_name} {label} lacks a help string"
    assert help_text.strip() == help_text, f"{command_name} {label} help has stray whitespace"
    assert help_text, f"{command_name} {label} help is empty"
    assert "\n" not in help_text, f"{command_name} {label} help spans multiple lines"


@pytest.mark.parametrize(("command_name", "option"), _EXPERIMENTAL_PARAMS)
def test_experimental_options_carry_prefix(command_name: str, option: str) -> None:
    param = _find_param(command_name, option)
    message = f"{command_name} {option} must start with '{_EXPERIMENTAL_PREFIX}'"
    assert str(param.help).startswith(_EXPERIMENTAL_PREFIX), message


@pytest.mark.parametrize(
    ("command_name", "label", "param"),
    [
        entry
        for entry in _ALL_PARAMS
        if entry[2].help and str(entry[2].help).startswith("Experimental:")
    ],
    ids=[
        f"{name}:{label}"
        for name, label, param in _ALL_PARAMS
        if param.help and str(param.help).startswith("Experimental:")
    ],
)
def test_no_unlisted_experimental_labels(command_name: str, label: str, param: Any) -> None:
    listed = {(command, option) for command, option in _EXPERIMENTAL_PARAMS}
    assert (command_name, label.split("/")[0]) in listed, (
        f"{command_name} {label} is labeled experimental but missing from _EXPERIMENTAL_PARAMS"
    )
