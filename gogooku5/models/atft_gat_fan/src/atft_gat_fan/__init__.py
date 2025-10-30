"""ATFT-GAT-FAN modular package.

Expose public training utilities once migration completes.
"""

from importlib import metadata
from typing import Final

__all__: Final = ["__version__"]


def __version__() -> str:
    """Return the installed package version.

    Falls back to "0.0.0" when the package metadata is unavailable.
    """

    try:
        return metadata.version("gogooku5-atft-gat-fan")
    except metadata.PackageNotFoundError:
        return "0.0.0"
