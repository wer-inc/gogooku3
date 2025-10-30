"""APEX-Ranker modular package."""

from importlib import metadata
from typing import Final

__all__: Final = ["__version__"]


def __version__() -> str:
    """Return the installed package version or "0.0.0".`"""

    try:
        return metadata.version("gogooku5-apex-ranker")
    except metadata.PackageNotFoundError:
        return "0.0.0"
