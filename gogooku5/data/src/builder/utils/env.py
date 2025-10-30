"""Environment management helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def load_local_env(env_file: str | Path = ".env") -> bool:
    """Load a local dotenv file if present.

    Returns True when the file exists and is loaded, False otherwise.
    """

    path = Path(env_file)
    if not path.exists():
        return False
    load_dotenv(dotenv_path=path)
    return True


def require_env_var(name: str) -> str:
    """Fetch an environment variable and raise a descriptive error if missing."""

    from os import getenv

    value = getenv(name)
    if value is None or value == "":
        raise RuntimeError(f"Environment variable '{name}' is required but missing.")
    return value


def ensure_env_loaded(env_file: Optional[str | Path] = None) -> None:
    """Best-effort load for tooling scripts before settings are instantiated."""

    candidate = env_file or ".env"
    load_local_env(candidate)
