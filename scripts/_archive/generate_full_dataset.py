#!/usr/bin/env python3
# DEPRECATED: Moved to scripts/_archive on 2025-09-04
"""
Automated Full Historical Dataset Generation
Generates dataset without user interaction
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import time
import asyncio

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from jquants_pipeline.pipeline import JQuantsPipelineV4Optimized
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Generate full historical dataset automatically"""
    
    start_time = time.time()
    
    # Date range: All available historical data
    # J-Quants API typically has data from 2017 onwards
    end_date = datetime.now()
    start_date = datetime(2017, 1, 1)  # From 2017 (earliest available)
    
    logger.info(f"""
    ========================================
    Starting Full Historical Dataset Generation
    ========================================
    Start Date: {start_date.strftime('%Y-%m-%d')}
    End Date: {end_date.strftime('%Y-%m-%d')}
    Period: ALL AVAILABLE DATA (2017-present)
    Output: /home/ubuntu/gogooku3-standalone/output/
    ========================================
    """)
    
    # Initialize pipeline
    pipeline = JQuantsPipelineV4Optimized(
        output_dir=Path('output')
    )
    
    try:
        # Execute pipeline
        import asyncio
        async def run_pipeline():
            return await pipeline.run(
                use_jquants=True,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
        
        df, metadata = asyncio.run(run_pipeline())
        result = {'output_file': str(metadata.get('output_file', 'ml_dataset_latest.parquet')),
                  'total_records': len(df) if df is not None else 0,
                  'date_range': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"}
        
        elapsed_time = time.time() - start_time
        
        if result:
            logger.info(f"""
            ========================================
            ✅ DATASET GENERATION COMPLETE
            ========================================
            Output file: {result['output_file']}
            Total records: {result.get('total_records', 'N/A')}
            Date range: {result.get('date_range', 'N/A')}
            Execution time: {elapsed_time:.2f} seconds
            
            Files created:
            - ml_dataset_latest.parquet (symlink)
            - ml_dataset_latest_metadata.json (symlink)
            - ml_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet
            - performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json
            ========================================
            """)
            return 0
        else:
            logger.error("Pipeline execution failed")
            return 1
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
