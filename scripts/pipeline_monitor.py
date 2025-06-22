#!/usr/bin/env python3
"""
Email RAG Pipeline Monitor
Monitors and logs statistics for the enhanced email processing pipeline
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

import psycopg2
from psycopg2.extras import RealDictCursor
import requests


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline_monitor.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PipelineStats:
    """Pipeline statistics structure"""
    timestamp: str
    emails_total: int
    emails_processed: int
    processing_rate: float
    
    # IMAP sync stats
    imap_messages_total: int
    imap_sync_rate_hour: int
    imap_sync_lag_seconds: float
    
    # Email scoring stats
    threads_classified: int
    avg_human_score: float
    avg_importance_score: float
    avg_commercial_score: float
    avg_sentiment_score: float
    
    # Content processing stats
    emails_content_processed: int
    elements_extracted: int
    chunks_created: int
    embeddings_generated: int
    avg_processing_time_ms: float
    avg_quality_score: float
    
    # Queue stats
    queue_classification_pending: int
    queue_content_pending: int
    queue_failed: int
    
    # Service health
    email_scorer_healthy: bool
    content_processor_healthy: bool
    database_healthy: bool
    
    # Resource usage
    memory_usage_mb: float
    cpu_usage_percent: float

class PipelineMonitor:
    """Monitor for the email RAG pipeline"""
    
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL", "postgresql://email_user:email_pass@localhost:5432/email_rag")
        self.email_scorer_url = os.getenv("EMAIL_SCORER_URL", "http://localhost:8081")
        self.content_processor_url = os.getenv("CONTENT_PROCESSOR_URL", "http://localhost:8082")
        
        # Connect to database
        try:
            self.db_conn = psycopg2.connect(self.db_url)
            logger.info("✅ Connected to database")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            sys.exit(1)
    
    def get_service_health(self, service_url: str) -> Dict[str, Any]:
        """Check health of a service"""
        try:
            response = requests.get(f"{service_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with self.db_conn.cursor(cursor_factory=RealDictCursor) as cursor:
                stats = {}
                
                # Email counts and IMAP sync stats
                cursor.execute("SELECT COUNT(*) as total FROM emails")
                stats['emails_total'] = cursor.fetchone()['total']
                
                # IMAP sync progress (raw messages vs processed emails)
                cursor.execute("SELECT COUNT(*) as imap_messages FROM imap_messages")
                imap_result = cursor.fetchone()
                stats['imap_messages_total'] = imap_result['imap_messages'] if imap_result else 0
                
                # IMAP sync rate (last hour)
                cursor.execute("""
                    SELECT COUNT(*) as recent_synced
                    FROM emails 
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                """)
                sync_result = cursor.fetchone()
                stats['imap_sync_rate_hour'] = sync_result['recent_synced'] if sync_result else 0
                
                # Latest IMAP message vs latest email (sync lag)
                cursor.execute("""
                    SELECT 
                        (SELECT MAX(created_at) FROM imap_messages) as latest_imap,
                        (SELECT MAX(created_at) FROM emails) as latest_email
                """)
                sync_lag = cursor.fetchone()
                if sync_lag['latest_imap'] and sync_lag['latest_email']:
                    lag_seconds = (sync_lag['latest_imap'] - sync_lag['latest_email']).total_seconds()
                    stats['imap_sync_lag_seconds'] = max(0, lag_seconds)
                else:
                    stats['imap_sync_lag_seconds'] = 0
                
                cursor.execute("""
                    SELECT COUNT(*) as processed 
                    FROM emails e 
                    JOIN processing_queue pq ON pq.thread_id = e.thread_id 
                    WHERE pq.status = 'completed'
                """)
                stats['emails_processed'] = cursor.fetchone()['processed']
                
                # Classification stats
                cursor.execute("""
                    SELECT 
                        COUNT(*) as classified,
                        AVG(human_score) as avg_human,
                        AVG(importance_score) as avg_importance,
                        AVG(commercial_score) as avg_commercial,
                        AVG(sentiment_score) as avg_sentiment
                    FROM classifications
                    WHERE human_score IS NOT NULL
                """)
                classification_stats = cursor.fetchone()
                stats.update({
                    'threads_classified': classification_stats['classified'] or 0,
                    'avg_human_score': float(classification_stats['avg_human'] or 0),
                    'avg_importance_score': float(classification_stats['avg_importance'] or 0),
                    'avg_commercial_score': float(classification_stats['avg_commercial'] or 0),
                    'avg_sentiment_score': float(classification_stats['avg_sentiment'] or 0),
                })
                
                # Content processing stats
                cursor.execute("""
                    SELECT 
                        COUNT(DISTINCT email_id) as emails_processed,
                        COUNT(*) as elements_total,
                        SUM(CASE WHEN is_cleaned THEN 1 ELSE 0 END) as elements_cleaned
                    FROM email_elements
                """)
                elements_stats = cursor.fetchone()
                stats.update({
                    'emails_content_processed': elements_stats['emails_processed'] or 0,
                    'elements_extracted': elements_stats['elements_total'] or 0,
                    'elements_cleaned': elements_stats['elements_cleaned'] or 0,
                })
                
                # Embeddings stats
                cursor.execute("""
                    SELECT 
                        COUNT(*) as embeddings_total,
                        COUNT(DISTINCT email_id) as emails_with_embeddings,
                        AVG(quality_score) as avg_quality
                    FROM enhanced_embeddings
                """)
                embeddings_stats = cursor.fetchone()
                stats.update({
                    'embeddings_generated': embeddings_stats['embeddings_total'] or 0,
                    'emails_with_embeddings': embeddings_stats['emails_with_embeddings'] or 0,
                    'avg_embedding_quality': float(embeddings_stats['avg_quality'] or 0),
                })
                
                # Processing time stats
                cursor.execute("""
                    SELECT 
                        AVG(processing_time_ms) as avg_time,
                        AVG(quality_score) as avg_quality
                    FROM processing_queue 
                    WHERE status = 'completed' AND actual_processing_time > 0
                """)
                time_stats = cursor.fetchone()
                stats.update({
                    'avg_processing_time_ms': float(time_stats['avg_time'] or 0),
                    'avg_quality_score': float(time_stats['avg_quality'] or 0),
                })
                
                # Queue stats
                cursor.execute("""
                    SELECT 
                        queue_type,
                        status,
                        COUNT(*) as count
                    FROM processing_queue 
                    GROUP BY queue_type, status
                """)
                queue_results = cursor.fetchall()
                
                queue_stats = {
                    'classification_pending': 0,
                    'content_pending': 0,
                    'failed_total': 0
                }
                
                for row in queue_results:
                    if row['queue_type'] == 'classification' and row['status'] == 'pending':
                        queue_stats['classification_pending'] = row['count']
                    elif row['queue_type'] == 'content_processing' and row['status'] == 'pending':
                        queue_stats['content_pending'] = row['count']
                    elif row['status'] == 'failed':
                        queue_stats['failed_total'] += row['count']
                
                stats.update(queue_stats)
                
                # Recent processing rate (last hour)
                cursor.execute("""
                    SELECT COUNT(*) as recent_processed
                    FROM processing_queue 
                    WHERE status = 'completed' 
                    AND processing_completed_at > NOW() - INTERVAL '1 hour'
                """)
                recent_stats = cursor.fetchone()
                stats['processing_rate_per_hour'] = recent_stats['recent_processed'] or 0
                
                return stats
                
        except Exception as e:
            logger.error(f"❌ Database stats error: {e}")
            return {}
    
    def collect_stats(self) -> PipelineStats:
        """Collect comprehensive pipeline statistics"""
        timestamp = datetime.utcnow().isoformat()
        
        # Get database stats
        db_stats = self.get_database_stats()
        
        # Get service health
        scorer_health = self.get_service_health(self.email_scorer_url)
        processor_health = self.get_service_health(self.content_processor_url)
        
        # Calculate processing rate
        processing_rate = db_stats.get('processing_rate_per_hour', 0) / 60.0  # per minute
        
        return PipelineStats(
            timestamp=timestamp,
            emails_total=db_stats.get('emails_total', 0),
            emails_processed=db_stats.get('emails_processed', 0),
            processing_rate=processing_rate,
            
            # IMAP sync stats
            imap_messages_total=db_stats.get('imap_messages_total', 0),
            imap_sync_rate_hour=db_stats.get('imap_sync_rate_hour', 0),
            imap_sync_lag_seconds=db_stats.get('imap_sync_lag_seconds', 0.0),
            
            # Classification stats
            threads_classified=db_stats.get('threads_classified', 0),
            avg_human_score=db_stats.get('avg_human_score', 0.0),
            avg_importance_score=db_stats.get('avg_importance_score', 0.0),
            avg_commercial_score=db_stats.get('avg_commercial_score', 0.0),
            avg_sentiment_score=db_stats.get('avg_sentiment_score', 0.0),
            
            # Content processing stats
            emails_content_processed=db_stats.get('emails_content_processed', 0),
            elements_extracted=db_stats.get('elements_extracted', 0),
            chunks_created=db_stats.get('embeddings_generated', 0),  # Using embeddings as proxy for chunks
            embeddings_generated=db_stats.get('embeddings_generated', 0),
            avg_processing_time_ms=db_stats.get('avg_processing_time_ms', 0.0),
            avg_quality_score=db_stats.get('avg_quality_score', 0.0),
            
            # Queue stats
            queue_classification_pending=db_stats.get('classification_pending', 0),
            queue_content_pending=db_stats.get('content_pending', 0),
            queue_failed=db_stats.get('failed_total', 0),
            
            # Service health
            email_scorer_healthy=scorer_health.get('status') == 'healthy',
            content_processor_healthy=processor_health.get('status') == 'healthy',
            database_healthy=bool(db_stats),
            
            # Resource usage (from services)
            memory_usage_mb=processor_health.get('details', {}).get('memory_usage_mb', 0.0),
            cpu_usage_percent=processor_health.get('details', {}).get('cpu_usage_percent', 0.0),
        )
    
    def print_stats(self, stats: PipelineStats):
        """Print formatted statistics"""
        print("\n" + "="*80)
        print(f"📊 EMAIL RAG PIPELINE STATISTICS - {stats.timestamp}")
        print("="*80)
        
        # IMAP sync status
        sync_pct = (stats.emails_total / stats.imap_messages_total * 100) if stats.imap_messages_total > 0 else 0
        sync_lag_mins = stats.imap_sync_lag_seconds / 60.0
        print(f"\n📨 IMAP SYNC STATUS:")
        print(f"   Raw IMAP messages:            {stats.imap_messages_total:,}")
        print(f"   Synced to emails table:       {stats.emails_total:,} ({sync_pct:.1f}%)")
        print(f"   IMAP sync rate (last hour):   {stats.imap_sync_rate_hour:,} emails")
        print(f"   Sync lag:                     {sync_lag_mins:.1f} minutes")
        
        # Email processing overview  
        processing_pct = (stats.emails_processed / stats.emails_total * 100) if stats.emails_total > 0 else 0
        print(f"\n📧 EMAIL PROCESSING OVERVIEW:")
        print(f"   Total emails in database:     {stats.emails_total:,}")
        print(f"   Emails processed:             {stats.emails_processed:,} ({processing_pct:.1f}%)")
        print(f"   Processing rate:              {stats.processing_rate:.1f} emails/minute")
        
        # Classification results
        print(f"\n🧠 AI CLASSIFICATION RESULTS:")
        print(f"   Threads classified:           {stats.threads_classified:,}")
        print(f"   Avg human score:              {stats.avg_human_score:.3f}")
        print(f"   Avg importance score:         {stats.avg_importance_score:.3f}")
        print(f"   Avg commercial score:         {stats.avg_commercial_score:.3f}")
        print(f"   Avg sentiment score:          {stats.avg_sentiment_score:.3f}")
        
        # Content processing results
        print(f"\n🔧 CONTENT PROCESSING RESULTS:")
        print(f"   Emails content processed:     {stats.emails_content_processed:,}")
        print(f"   Elements extracted:           {stats.elements_extracted:,}")
        print(f"   Chunks created:               {stats.chunks_created:,}")
        print(f"   Embeddings generated:         {stats.embeddings_generated:,}")
        print(f"   Avg processing time:          {stats.avg_processing_time_ms:.0f}ms")
        print(f"   Avg quality score:            {stats.avg_quality_score:.3f}")
        
        # Queue status
        total_pending = stats.queue_classification_pending + stats.queue_content_pending
        print(f"\n⏳ PROCESSING QUEUE STATUS:")
        print(f"   Classification pending:       {stats.queue_classification_pending:,}")
        print(f"   Content processing pending:   {stats.queue_content_pending:,}")
        print(f"   Total pending:                {total_pending:,}")
        print(f"   Failed items:                 {stats.queue_failed:,}")
        
        # Service health
        scorer_status = "🟢 HEALTHY" if stats.email_scorer_healthy else "🔴 UNHEALTHY"
        processor_status = "🟢 HEALTHY" if stats.content_processor_healthy else "🔴 UNHEALTHY"
        db_status = "🟢 HEALTHY" if stats.database_healthy else "🔴 UNHEALTHY"
        
        print(f"\n🏥 SERVICE HEALTH:")
        print(f"   Email Scorer:                 {scorer_status}")
        print(f"   Content Processor:            {processor_status}")
        print(f"   Database:                     {db_status}")
        
        # Resource usage
        print(f"\n💻 RESOURCE USAGE:")
        print(f"   Memory usage:                 {stats.memory_usage_mb:.1f} MB")
        print(f"   CPU usage:                    {stats.cpu_usage_percent:.1f}%")
        
        print("="*80)
    
    def save_stats_json(self, stats: PipelineStats, filename: str = None):
        """Save statistics to JSON file"""
        if filename is None:
            filename = f"pipeline_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        stats_dict = {
            'timestamp': stats.timestamp,
            'imap_sync': {
                'imap_messages_total': stats.imap_messages_total,
                'emails_synced': stats.emails_total,
                'sync_rate_hour': stats.imap_sync_rate_hour,
                'sync_lag_seconds': stats.imap_sync_lag_seconds
            },
            'emails': {
                'total': stats.emails_total,
                'processed': stats.emails_processed,
                'processing_rate_per_minute': stats.processing_rate
            },
            'classification': {
                'threads_classified': stats.threads_classified,
                'avg_human_score': stats.avg_human_score,
                'avg_importance_score': stats.avg_importance_score,
                'avg_commercial_score': stats.avg_commercial_score,
                'avg_sentiment_score': stats.avg_sentiment_score
            },
            'content_processing': {
                'emails_processed': stats.emails_content_processed,
                'elements_extracted': stats.elements_extracted,
                'chunks_created': stats.chunks_created,
                'embeddings_generated': stats.embeddings_generated,
                'avg_processing_time_ms': stats.avg_processing_time_ms,
                'avg_quality_score': stats.avg_quality_score
            },
            'queue': {
                'classification_pending': stats.queue_classification_pending,
                'content_pending': stats.queue_content_pending,
                'failed': stats.queue_failed
            },
            'services': {
                'email_scorer_healthy': stats.email_scorer_healthy,
                'content_processor_healthy': stats.content_processor_healthy,
                'database_healthy': stats.database_healthy
            },
            'resources': {
                'memory_usage_mb': stats.memory_usage_mb,
                'cpu_usage_percent': stats.cpu_usage_percent
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(stats_dict, f, indent=2)
            logger.info(f"💾 Saved statistics to {filename}")
        except Exception as e:
            logger.error(f"❌ Failed to save stats: {e}")
    
    def monitor_continuous(self, interval: int = 60):
        """Continuously monitor pipeline with specified interval"""
        logger.info(f"🔄 Starting continuous monitoring (interval: {interval}s)")
        
        try:
            while True:
                stats = self.collect_stats()
                self.print_stats(stats)
                
                # Save timestamped stats
                self.save_stats_json(stats)
                
                # Sleep until next iteration
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("⏹️ Monitoring stopped by user")
        except Exception as e:
            logger.error(f"❌ Monitoring error: {e}")
    
    def monitor_once(self):
        """Run monitoring once and exit"""
        stats = self.collect_stats()
        self.print_stats(stats)
        self.save_stats_json(stats, "pipeline_stats_latest.json")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Email RAG Pipeline Monitor")
    parser.add_argument("--continuous", "-c", action="store_true", 
                       help="Run continuous monitoring")
    parser.add_argument("--interval", "-i", type=int, default=60, 
                       help="Monitoring interval in seconds (default: 60)")
    parser.add_argument("--once", "-o", action="store_true", 
                       help="Run once and exit (default)")
    
    args = parser.parse_args()
    
    monitor = PipelineMonitor()
    
    if args.continuous:
        monitor.monitor_continuous(args.interval)
    else:
        monitor.monitor_once()

if __name__ == "__main__":
    main()