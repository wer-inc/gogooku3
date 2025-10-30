#!/usr/bin/env python3
# DEPRECATED: Moved to scripts/_archive on 2025-09-04
"""
JQuants Pipeline実行用のエントリーポイント

使用例:
    python run_jquants_pipeline.py --start-date 2024-12-16 --end-date 2024-12-20
"""

import os
import sys

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the pipeline
from jquants_pipeline.pipeline import main

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
