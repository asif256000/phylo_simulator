from __future__ import annotations

from typing import Any, Iterable


def format_tuple(value: Any) -> str:
    """Render tensor-friendly tuple or scalar values as strings for logging."""
    if value is None:
        return "None"
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return "(" + ", ".join(str(v) for v in value) + ")"
    return str(value)


__all__ = ["format_tuple"]
