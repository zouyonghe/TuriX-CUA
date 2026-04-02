from __future__ import annotations

import os
import re
from typing import Any


_ENV_PLACEHOLDER_RE = re.compile(
    r"^\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))$"
)


def resolve_env_placeholders(value: Any) -> Any:
    """Recursively expand shell-style env placeholders like $API_KEY or ${API_KEY}."""
    if isinstance(value, dict):
        return {key: resolve_env_placeholders(val) for key, val in value.items()}

    if isinstance(value, list):
        return [resolve_env_placeholders(item) for item in value]

    if not isinstance(value, str):
        return value

    match = _ENV_PLACEHOLDER_RE.fullmatch(value)
    if not match:
        return value

    env_name = match.group("braced") or match.group("plain")
    return os.getenv(env_name)
