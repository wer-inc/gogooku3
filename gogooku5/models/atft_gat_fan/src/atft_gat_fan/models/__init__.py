"""Neural network architectures for ATFT-GAT-FAN.

Phase 2 Bug #32 fix: Add stub class to prevent import errors when models are not implemented.
"""
from __future__ import annotations


class _StubModel:
    """Stub model class for gogooku5 Phase 2 migration.

    ATFT/GAT/FAN models are not implemented in gogooku5 Phase 2.
    This stub prevents import errors during the transition period.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "ATFT/GAT/FAN models are not provided in this build. "
            "Use the gogooku3 package for full ATFT-GAT-FAN implementation."
        )


__all__ = ["_StubModel"]
