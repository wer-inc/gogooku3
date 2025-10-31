"""APEX-Ranker package (v0 baseline)."""

from importlib import metadata
from typing import Final

from . import api, backtest, data, losses, models, utils  # noqa: F401

__all__: Final = ["__version__", "api", "backtest", "data", "losses", "models", "utils"]


def __version__() -> str:
    """Return the installed package version or a fallback when running from source."""

    try:
        return metadata.version("gogooku5-apex-ranker")
    except metadata.PackageNotFoundError:
        return "0.0.0"
