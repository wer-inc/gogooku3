#!/usr/bin/env python3
# DEPRECATED: Moved to scripts/_archive on 2025-09-04
"""
Full Historical Dataset Generation Script
Fetches all available historical data from J-Quants API (2017-present)
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from jquants_pipeline.pipeline import JQuantsPipelineV4
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Generate full historical dataset from J-Quants API"""
    
    # Calculate date range
    # J-Quants typically has data from 2017 onwards
    # We'll fetch from 2018-01-01 to be safe
    start_date = datetime(2018, 1, 1)
    end_date = datetime.now()
    
    # Calculate the number of days
    days_diff = (end_date - start_date).days
    
    logger.info(f"""
    ====================================
    Full Historical Dataset Generation
    ====================================
    Start Date: {start_date.strftime('%Y-%m-%d')}
    End Date: {end_date.strftime('%Y-%m-%d')}
    Total Days: {days_diff} days
    
    This will fetch approximately:
    - 6.8 years of data
    - ~1,700 trading days
    - ~6.8 million records (assuming 4,000 stocks)
    
    Estimated time: 10-15 minutes
    Estimated API calls: ~1,700
    ====================================
    """)
    
    # Confirm with user
    response = input("Do you want to proceed? (yes/no): ")
    if response.lower() != 'yes':
        logger.info("Cancelled by user")
        return
    
    # Initialize pipeline
    pipeline = JQuantsPipelineV4(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        output_dir='output',
        use_cache=True
    )
    
    try:
        logger.info("Starting pipeline execution...")
        
        # Execute pipeline
        result = pipeline.run()
        
        if result:
            logger.info(f"""
            ====================================
            ✅ FULL HISTORICAL DATASET GENERATED
            ====================================
            Output file: {result['output_file']}
            Total records: {result.get('total_records', 'N/A')}
            Date range: {result.get('date_range', 'N/A')}
            Execution time: {result.get('execution_time', 'N/A')}
            ====================================
            """)
        else:
            logger.error("Pipeline execution failed")
            
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
