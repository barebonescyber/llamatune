"""Output-boundary sanitizers for untrusted strings (SEC-003, SEC-004).

Model files, llama-server stderr, and parser exceptions are data that end up
embedded in generated Markdown reports, terminal echoes, ``warnings.warn``
messages, and JSON evidence summaries. This module is the single low-level
choke point for making those strings presentation-safe:

- :func:`strip_control_chars` removes C0/C1 control characters (keeping a
  configurable set such as ``\\n`` and ``\\t``) plus Unicode bidirectional
  formatting characters, so terminal echo and warning text cannot carry
  ANSI escapes, carriage-return spoofing, or direction overrides.
- :func:`markdown_text` makes a string safe for Markdown prose: HTML
  special characters are escaped into entities so raw HTML/link payloads
  render as literal text in common Markdown pipelines, and backslash
  escapes are neutralized so the escaping itself cannot be undone.
- :func:`markdown_cell` additionally applies the table-cell discipline
  (pipe escaping, newlines flattened to spaces) used by every Markdown
  table renderer.

The module deliberately imports nothing so it can sit below every renderer.
Bounded on-disk capture artifacts (stderr logs) are out of scope: raw bytes
there are intentional evidence, sanitized only when surfaced.
"""

from __future__ import annotations

#: Control characters preserved by :func:`strip_control_chars` by default.
DEFAULT_KEEP = "\n\t"

#: Unicode bidirectional/format control characters stripped alongside C0/C1
#: so echoed text cannot flip rendering direction (terminal spoofing).
_BIDI_CONTROLS = frozenset("\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")


def strip_control_chars(value: str, *, keep: str = DEFAULT_KEEP) -> str:
    """Return ``value`` without C0/C1 control characters or bidi controls.

    Characters listed in ``keep`` survive untouched; the default keeps
    newline and tab so multi-line evidence stays readable. Printable
    Unicode passes through unchanged.
    """
    kept = frozenset(keep)

    def _is_control(char: str) -> bool:
        if char < " ":  # C0 range
            return True
        return "\x7f" <= char <= "\u009f" or char in _BIDI_CONTROLS  # DEL, C1, bidi

    return "".join(char for char in value if char in kept or not _is_control(char))


def markdown_text(value: object) -> str:
    """Escape ``value`` for embedding as Markdown prose.

    Control characters are stripped, backslashes are doubled so no later
    escape sequence can be neutralized, and ``& < > " '`` become HTML
    entities so injected markup renders as text. Plain benign names
    (letters, digits, spaces, dashes, dots, parentheses, slashes) pass
    through byte-for-byte unchanged.
    """
    text = value if isinstance(value, str) else str(value)
    return (
        strip_control_chars(text)
        .replace("\\", "\\\\")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def markdown_cell(value: object) -> str:
    """Escape ``value`` for one Markdown table cell.

    Builds on :func:`markdown_text`, then escapes ``|`` and flattens
    newlines to spaces so a hostile cell cannot break out of its column
    or row. Existing renderers' pipe/newline discipline is preserved.
    """
    return markdown_text(value).replace("|", "\\|").replace("\n", " ")
