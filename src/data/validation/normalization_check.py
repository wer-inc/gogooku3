"""
NormalizationValidator - Stub for backward compatibility.
"""

import logging

logger = logging.getLogger(__name__)


class NormalizationValidator:
    """Stub NormalizationValidator for compatibility."""

    def __init__(self):
        """Initialize validator."""
        pass

    def validate(self, data):
        """Validate normalization (stub implementation)."""
        logger.debug("NormalizationValidator.validate called (stub)")
        return True

    def check_normalization(self, data):
        """Check normalization (stub implementation)."""
        logger.debug("NormalizationValidator.check_normalization called (stub)")
        return {"status": "ok", "warnings": []}
