"""
Upload rescored results to S3.
"""
import os
import sys
import logging
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.storage.s3_handler import S3Handler

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def upload_latest_rescored():
    """Upload the most recent rescored results to S3."""
    logger = logging.getLogger(__name__)
    setup_logging()
    
    logger.info("Looking for latest rescored results...")
    
    # Find the latest rescored file
    results_dir = 'results'
    parquet_files = [
        f for f in os.listdir(results_dir) 
        if f.endswith('.parquet') 
        and 'rescored' in f 
        and 'backup' not in f
    ]
    
    if not parquet_files:
        logger.error("No rescored files found!")
        return
    
    # Get most recent by modification time
    latest_file = max(parquet_files, key=lambda f: os.path.getmtime(os.path.join(results_dir, f)))
    filepath = os.path.join(results_dir, latest_file)
    
    logger.info(f"Found latest rescored file: {latest_file}")
    logger.info(f"Loading from: {filepath}")
    
    # Load the data
    df = pd.read_parquet(filepath)
    logger.info(f"Loaded {len(df)} evaluation results")
    
    # Show summary
    logger.info(f"\nDataFrame summary:")
    logger.info(f"  Models: {df['model_name'].unique().tolist()}")
    logger.info(f"  Charts: {df['chart_id'].nunique()}")
    logger.info(f"  Total evaluations: {len(df)}")
    logger.info(f"  Average score: {df['score'].mean():.3f}")
    
    # Upload to S3
    try:
        s3_handler = S3Handler()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = s3_handler.upload_results(df, timestamp)
        logger.info(f"\n✅ Successfully uploaded to S3: {s3_key}")
        logger.info(f"Dashboard will now use these results when you refresh!")
    except Exception as e:
        logger.error(f"\n❌ Failed to upload to S3: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    upload_latest_rescored()

