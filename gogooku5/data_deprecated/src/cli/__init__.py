"""
Unified CLI for gogooku5 dataset builder.

This module provides a consolidated command-line interface for:
- Dataset building with automatic chunking
- Chunk merging
- Environment validation
- Progress monitoring

Entry point: `gogooku5-dataset` or `python -m gogooku5_data.cli`
"""

__version__ = "1.0.0"
__all__ = ["main"]

from .main import main
