"""
Cloudflare R2 Storage Module
Handles reading/writing articles to R2 bucket
"""
import logging
import json
import os
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class R2Storage:
    """Cloudflare R2 storage handler using S3-compatible API"""
    
    def __init__(self):
        self.bucket_name = os.getenv('R2_BUCKET_NAME')
        self.account_id = os.getenv('R2_ACCOUNT_ID')
        self.access_key = os.getenv('R2_ACCESS_KEY_ID')
        self.secret_key = os.getenv('R2_SECRET_ACCESS_KEY')
        
        # Optional: Allow custom endpoint URL (useful for multiple accounts)
        self.endpoint_url = os.getenv('R2_ENDPOINT_URL')
        if not self.endpoint_url and self.account_id:
            self.endpoint_url = f'https://{self.account_id}.r2.cloudflarestorage.com'
        
        if not all([self.bucket_name, self.endpoint_url, self.access_key, self.secret_key]):
            raise ValueError("Missing R2 credentials in environment variables")
        
        # Initialize S3 client for R2
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name='auto'
        )
        
        logger.info(f"✅ R2 Storage initialized - Bucket: {self.bucket_name}")
    
    def get_today_key(self) -> str:
        """Get the R2 key for today's aggregated articles (EST timezone)"""
        from datetime import timezone, timedelta
        
        # Always use EST timezone for consistency
        est = timezone(timedelta(hours=-5))
        today = datetime.now(est).date().strftime("%Y-%m-%d")
        return f"articles/{today}.json"
    
    def load_today_articles(self) -> Optional[List[Dict]]:
        """
        Load today's accumulated articles from R2
        
        Returns:
            List of articles, or empty list if not found
        """
        key = self.get_today_key()
        
        try:
            logger.info(f"Loading existing articles from R2: {key}")
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            data = json.loads(response['Body'].read().decode('utf-8'))
            
            articles = data.get('articles', [])
            logger.info(f"✅ Loaded {len(articles)} existing articles from R2")
            return articles
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.info(f"No existing file for today - starting fresh")
                return []
            else:
                logger.error(f"R2 read error: {e}")
                return []
        except Exception as e:
            logger.error(f"Failed to load from R2: {e}")
            return []
    
    def save_today_articles(self, articles: List[Dict], run_number: int = 1) -> bool:
        """
        Save accumulated articles to R2 for today
        
        Args:
            articles: List of articles with scores
            run_number: Which run of the day (1=9am, 2=noon, 3=4pm)
        
        Returns:
            True if successful
        """
        key = self.get_today_key()
        
        # Calculate stats
        with_desc = sum(1 for a in articles if a.get('description'))
        graded = sum(1 for a in articles if 'overall_score' in a)
        
        # Calculate dimensional stats for graded articles
        graded_articles = [a for a in articles if 'overall_score' in a]
        if graded_articles:
            avg_macro = sum(a.get('macro_score', 0) for a in graded_articles) / len(graded_articles)
            avg_equities = sum(a.get('equities_score', 0) for a in graded_articles) / len(graded_articles)
            avg_sector = sum(a.get('sector_score', 0) for a in graded_articles) / len(graded_articles)
            avg_sentiment = sum(a.get('sentiment', 0) for a in graded_articles) / len(graded_articles)
        else:
            avg_macro = avg_equities = avg_sector = avg_sentiment = 0
        
        # Count by primary dimension
        dimension_counts = {}
        for art in graded_articles:
            dim = art.get('dimension_primary', 'unknown')
            dimension_counts[dim] = dimension_counts.get(dim, 0) + 1
        
        # Sort articles by overall_score (highest first)
        sorted_articles = sorted(
            articles,
            key=lambda x: x.get('overall_score', 0),
            reverse=True
        )
        
        # Use EST timezone for consistency
        from datetime import timezone, timedelta
        est = timezone(timedelta(hours=-5))
        
        # Prepare output data
        output_data = {
            "date": datetime.now(est).date().isoformat(),
            "last_updated": datetime.now(est).isoformat(),
            "run_number": run_number,
            "total_articles": len(articles),
            "graded_articles": graded,
            "pending_articles": len(articles) - graded,
            "articles_with_descriptions": with_desc,
            "metadata_coverage_pct": round(with_desc/len(articles)*100, 1) if articles else 0,
            
            # Dimensional statistics (for graded articles only)
            "dimension_stats": {
                "avg_macro_score": round(avg_macro, 2),
                "avg_equities_score": round(avg_equities, 2),
                "avg_sector_score": round(avg_sector, 2),
                "avg_sentiment": round(avg_sentiment, 2)
            },
            
            "dimension_distribution": dimension_counts,
            
            "articles": sorted_articles
        }
        
        try:
            # Convert to JSON
            json_data = json.dumps(output_data, indent=2, ensure_ascii=False)
            
            # Upload to R2
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json_data.encode('utf-8'),
                ContentType='application/json'
            )
            
            logger.info(f"✅ Saved to R2: {key}")
            logger.info(f"   Run number:        {run_number}/3")
            logger.info(f"   Total articles:    {len(articles)}")
            logger.info(f"   Graded:            {graded}")
            logger.info(f"   Pending:           {len(articles) - graded}")
            logger.info(f"   With descriptions: {with_desc} ({with_desc/len(articles)*100:.1f}%)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save to R2: {e}")
            return False
    
    def list_all_files(self, prefix: str = 'articles/') -> List[Dict]:
        """
        List ALL files in R2 bucket with given prefix (with pagination support)
        
        Args:
            prefix: Prefix to filter files
        
        Returns:
            List of file objects with 'Key' and 'LastModified' fields
        """
        try:
            all_objects = []
            continuation_token = None
            
            while True:
                # Build parameters
                params = {
                    'Bucket': self.bucket_name,
                    'Prefix': prefix  # ✅ Use Prefix, not Key
                }
                
                if continuation_token:
                    params['ContinuationToken'] = continuation_token
                
                response = self.s3_client.list_objects_v2(**params)
                
                # Add objects from this page
                if 'Contents' in response:
                    all_objects.extend(response['Contents'])
                
                # Check if there are more pages
                if not response.get('IsTruncated', False):
                    break
                    
                continuation_token = response.get('NextContinuationToken')
            
            logger.info(f"Found {len(all_objects)} files with prefix '{prefix}'")
            return all_objects
            
        except ClientError as e:
            # Handle empty bucket or missing prefix gracefully
            if e.response['Error']['Code'] in ['NoSuchKey', 'NoSuchBucket']:
                logger.info(f"No files found with prefix '{prefix}' (bucket may be empty)")
                return []
            else:
                logger.error(f"Failed to list R2 files: {e}")
                return []
        except Exception as e:
            logger.error(f"Failed to list R2 files: {e}")
            return []
    
    def list_recent_files(self, days: int = 7) -> List[str]:
        """
        List recent article files from R2
        
        Args:
            days: Number of recent days to list
        
        Returns:
            List of S3 keys
        """
        try:
            all_objects = self.list_all_files(prefix='articles/')
            
            if not all_objects:
                return []
            
            # Get keys and sort by date (newest first)
            keys = [obj['Key'] for obj in all_objects]
            keys.sort(reverse=True)
            
            return keys[:days]
            
        except Exception as e:
            logger.error(f"Failed to list recent files: {e}")
            return []
    
    def cleanup_old_files(self, keep_days: int = 30):
        """
        Delete article files older than keep_days
        
        Args:
            keep_days: Number of days to retain
        """
        try:
            logger.info(f"🗑️  Starting cleanup of files older than {keep_days} days...")
            
            # Get ALL files (not just recent ones)
            all_objects = self.list_all_files(prefix='articles/')
            
            if not all_objects:
                logger.info("No files found to clean up")
                return
            
            # Use EST timezone for consistency
            from datetime import timezone, timedelta
            est = timezone(timedelta(hours=-5))
            cutoff_date = datetime.now(est).date() - timedelta(days=keep_days)
            cutoff_str = cutoff_date.isoformat()
            
            deleted = 0
            for obj in all_objects:
                key = obj['Key']
                # Extract date from key like "articles/2025-01-15.json"
                try:
                    file_date = key.split('/')[1].split('.')[0]  # Gets "2025-01-15"
                    
                    if file_date < cutoff_str:
                        self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
                        deleted += 1
                        logger.info(f"   Deleted old file: {key}")
                except (IndexError, ValueError) as e:
                    # Skip files that don't match expected format
                    logger.debug(f"   Skipping file with unexpected format: {key}")
                    continue
            
            if deleted > 0:
                logger.info(f"✅ Cleaned up {deleted} old files")
            else:
                logger.info("No old files to clean up")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# Singleton instance
_storage = None

def get_r2_storage() -> R2Storage:
    """Get or create R2 storage singleton"""
    global _storage
    if _storage is None:
        _storage = R2Storage()
    return _storage