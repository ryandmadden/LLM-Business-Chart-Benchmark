"""
Manually upload evaluation results to S3.
Use this after fixing S3 permissions to upload existing results.
"""
import os
import sys
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from storage.s3_handler import upload_results

def upload_existing_results():
    """Upload the most recent evaluation results to S3."""
    load_dotenv()
    
    results_dir = "results"
    
    # Find the most recent results file
    parquet_files = [f for f in os.listdir(results_dir) if f.endswith('.parquet') and f.startswith('scored_results_')]
    
    if not parquet_files:
        print("❌ No evaluation results found in results/ directory")
        print("Please run the evaluation first: python -m scripts.run_evaluation")
        return False
    
    # Get the most recent file (by filename timestamp)
    latest_file = sorted(parquet_files)[-1]
    file_path = os.path.join(results_dir, latest_file)
    
    print(f"Found results file: {latest_file}")
    print(f"File size: {os.path.getsize(file_path) / 1024:.1f} KB")
    
    try:
        # Load the results
        print("Loading results...")
        df = pd.read_parquet(file_path)
        print(f"Loaded {len(df)} evaluation records")
        
        # Upload to S3
        print("Uploading to S3...")
        upload_results(df)
        print("✅ Results uploaded successfully to S3!")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

if __name__ == "__main__":
    success = upload_existing_results()
    sys.exit(0 if success else 1)
