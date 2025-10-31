"""
S3 storage integration for uploading and downloading evaluation results.
"""
import os
import boto3
import pandas as pd
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from botocore.exceptions import ClientError, NoCredentialsError
from config import S3_CONFIG


class S3Handler:
    """Handler for S3 operations."""
    
    def __init__(self):
        """Initialize S3 client."""
        self.bucket_name = S3_CONFIG["bucket_name"]
        self.region = S3_CONFIG["region"]
        self.results_prefix = S3_CONFIG["results_prefix"]
        
        try:
            self.s3_client = boto3.client(
                's3',
                region_name=self.region
            )
            self.logger = logging.getLogger(__name__)
        except NoCredentialsError:
            self.logger.error("AWS credentials not found. Please configure your credentials.")
            raise
    
    def upload_results(self, df: pd.DataFrame, run_id: Optional[str] = None) -> str:
        """
        Upload scored results as Parquet to S3 with timestamp.
        
        Args:
            df: DataFrame containing evaluation results
            run_id: Optional run identifier
            
        Returns:
            S3 key of uploaded file
            
        Raises:
            ClientError: If upload fails
        """
        try:
            # Generate timestamp and filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if run_id:
                filename = f"scored_results_{run_id}_{timestamp}.parquet"
            else:
                filename = f"scored_results_{timestamp}.parquet"
            
            key = f"{self.results_prefix}{filename}"
            
            # Convert DataFrame to Parquet bytes
            parquet_buffer = df.to_parquet(index=False)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=parquet_buffer,
                ContentType='application/octet-stream'
            )
            
            self.logger.info(f"Results uploaded to S3: s3://{self.bucket_name}/{key}")
            return key
            
        except ClientError as e:
            error_msg = f"Failed to upload results to S3: {str(e)}"
            self.logger.error(error_msg)
            raise ClientError(error_msg)
    
    def download_latest_results(self) -> Optional[pd.DataFrame]:
        """
        Download the most recent evaluation results from S3.
        
        Returns:
            DataFrame with latest results or None if no results found
            
        Raises:
            ClientError: If download fails
        """
        try:
            # List objects in results prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.results_prefix
            )
            
            if 'Contents' not in response:
                self.logger.warning("No results found in S3")
                return None
            
            # Sort by last modified date (most recent first)
            objects = sorted(response['Contents'], 
                           key=lambda x: x['LastModified'], 
                           reverse=True)
            
            # Get the most recent file
            latest_object = objects[0]
            key = latest_object['Key']
            
            # Download the file
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            
            # Read Parquet data - need to read bytes first to avoid seek errors
            import io
            parquet_buffer = io.BytesIO(response['Body'].read())
            df = pd.read_parquet(parquet_buffer)
            
            self.logger.info(f"Downloaded latest results from S3: s3://{self.bucket_name}/{key}")
            return df
            
        except ClientError as e:
            error_msg = f"Failed to download results from S3: {str(e)}"
            self.logger.error(error_msg)
            raise ClientError(error_msg)
    
    def list_results(self, limit: int = 10) -> list[Dict[str, Any]]:
        """
        List available result files in S3.
        
        Args:
            limit: Maximum number of files to return
            
        Returns:
            List of file metadata dictionaries
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.results_prefix
            )
            
            if 'Contents' not in response:
                return []
            
            # Sort by last modified date (most recent first)
            objects = sorted(response['Contents'], 
                           key=lambda x: x['LastModified'], 
                           reverse=True)
            
            # Return metadata for each file
            results = []
            for obj in objects[:limit]:
                results.append({
                    'key': obj['Key'],
                    'last_modified': obj['LastModified'],
                    'size': obj['Size'],
                    'url': f"s3://{self.bucket_name}/{obj['Key']}"
                })
            
            return results
            
        except ClientError as e:
            error_msg = f"Failed to list results in S3: {str(e)}"
            self.logger.error(error_msg)
            return []
    
    def download_results_by_key(self, key: str) -> Optional[pd.DataFrame]:
        """
        Download specific results file by S3 key.
        
        Args:
            key: S3 key of the file to download
            
        Returns:
            DataFrame with results or None if download fails
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            
            # Read Parquet data - need to read bytes first to avoid seek errors
            import io
            parquet_buffer = io.BytesIO(response['Body'].read())
            df = pd.read_parquet(parquet_buffer)
            
            self.logger.info(f"Downloaded results from S3: s3://{self.bucket_name}/{key}")
            return df
            
        except ClientError as e:
            error_msg = f"Failed to download results from S3: {str(e)}"
            self.logger.error(error_msg)
            return None
    
    def delete_results(self, key: str) -> bool:
        """
        Delete results file from S3.
        
        Args:
            key: S3 key of the file to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            self.logger.info(f"Deleted results from S3: s3://{self.bucket_name}/{key}")
            return True
            
        except ClientError as e:
            error_msg = f"Failed to delete results from S3: {str(e)}"
            self.logger.error(error_msg)
            return False
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about S3 storage usage.
        
        Returns:
            Dictionary with storage information
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.results_prefix
            )
            
            if 'Contents' not in response:
                return {
                    'total_files': 0,
                    'total_size_bytes': 0,
                    'total_size_mb': 0,
                    'bucket_name': self.bucket_name,
                    'region': self.region
                }
            
            total_size = sum(obj['Size'] for obj in response['Contents'])
            total_files = len(response['Contents'])
            
            return {
                'total_files': total_files,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'bucket_name': self.bucket_name,
                'region': self.region
            }
            
        except ClientError as e:
            error_msg = f"Failed to get storage info: {str(e)}"
            self.logger.error(error_msg)
            return {
                'total_files': 0,
                'total_size_bytes': 0,
                'total_size_mb': 0,
                'bucket_name': self.bucket_name,
                'region': self.region,
                'error': error_msg
            }


# Convenience functions for backward compatibility
def upload_results(df: pd.DataFrame, run_id: Optional[str] = None) -> str:
    """Upload results to S3."""
    handler = S3Handler()
    return handler.upload_results(df, run_id)


def download_latest_results() -> Optional[pd.DataFrame]:
    """Download latest results from S3."""
    handler = S3Handler()
    return handler.download_latest_results()
