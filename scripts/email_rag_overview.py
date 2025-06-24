#!/usr/bin/env python3
"""
Email RAG System Overview
Quick overview of emails, threads, classifications, and processing status
"""

import os
import sys
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import DictCursor

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://email_user:email_pass@localhost:5433/email_rag")

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(DATABASE_URL)

def get_overview_stats():
    """Get comprehensive overview statistics"""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            stats = {}
            
            # Basic counts
            cur.execute("SELECT COUNT(*) as count FROM emails")
            stats['total_emails'] = cur.fetchone()['count']
            
            cur.execute("SELECT COUNT(*) as count FROM threads")
            stats['total_threads'] = cur.fetchone()['count']
            
            cur.execute("SELECT COUNT(*) as count FROM classifications")
            stats['total_classifications'] = cur.fetchone()['count']
            
            # Email date range
            cur.execute("SELECT MIN(date_received) as earliest, MAX(date_received) as latest FROM emails")
            date_range = cur.fetchone()
            stats['earliest_email'] = date_range['earliest']
            stats['latest_email'] = date_range['latest']
            
            # Recent activity (last 24 hours)
            cur.execute("""
                SELECT COUNT(*) as count 
                FROM emails 
                WHERE date_received > NOW() - INTERVAL '24 hours'
            """)
            stats['emails_last_24h'] = cur.fetchone()['count']
            
            # Classifications by type
            cur.execute("""
                SELECT 
                    classification, 
                    COUNT(*) as count,
                    AVG(confidence) as avg_confidence
                FROM classifications 
                GROUP BY classification
                ORDER BY count DESC
            """)
            stats['classifications_by_type'] = [dict(row) for row in cur.fetchall()]
            
            # Threading stats
            cur.execute("""
                SELECT 
                    COUNT(*) as single_message_threads,
                    AVG(message_count) as avg_messages_per_thread,
                    MAX(message_count) as max_messages_in_thread
                FROM threads
                WHERE message_count = 1
            """)
            threading_stats = cur.fetchone()
            stats.update(dict(threading_stats))
            
            cur.execute("""
                SELECT COUNT(*) as multi_message_threads
                FROM threads
                WHERE message_count > 1
            """)
            stats['multi_message_threads'] = cur.fetchone()['multi_message_threads']
            
            # Top participants
            cur.execute("""
                SELECT 
                    from_email,
                    COUNT(*) as email_count
                FROM emails
                GROUP BY from_email
                ORDER BY email_count DESC
                LIMIT 10
            """)
            stats['top_senders'] = [dict(row) for row in cur.fetchall()]
            
            # Classification quality metrics
            cur.execute("""
                SELECT 
                    AVG(confidence) as avg_confidence,
                    COUNT(CASE WHEN should_process = true THEN 1 END) as should_process_count,
                    AVG(relationship_strength) as avg_relationship_strength,
                    AVG(sentiment_score) as avg_sentiment_score
                FROM classifications
            """)
            quality_stats = cur.fetchone()
            if quality_stats['avg_confidence']:
                stats.update({
                    'avg_classification_confidence': float(quality_stats['avg_confidence']),
                    'emails_should_process': quality_stats['should_process_count'],
                    'avg_relationship_strength': float(quality_stats['avg_relationship_strength'] or 0),
                    'avg_sentiment_score': float(quality_stats['avg_sentiment_score'] or 0)
                })
            
            # Recent classifications
            cur.execute("""
                SELECT 
                    email_id,
                    classification,
                    confidence,
                    sentiment,
                    priority,
                    model_used,
                    created_at
                FROM classifications
                ORDER BY created_at DESC
                LIMIT 5
            """)
            stats['recent_classifications'] = [dict(row) for row in cur.fetchall()]
            
            # Processing queue status
            try:
                cur.execute("""
                    SELECT 
                        queue_type,
                        status,
                        COUNT(*) as count
                    FROM processing_queue
                    GROUP BY queue_type, status
                """)
                queue_stats = [dict(row) for row in cur.fetchall()]
                stats['processing_queue'] = queue_stats
            except:
                stats['processing_queue'] = []
            
            return stats

def print_overview():
    """Print comprehensive overview"""
    try:
        stats = get_overview_stats()
        
        print("="*80)
        print("📊 EMAIL RAG SYSTEM OVERVIEW")
        print("="*80)
        
        # Basic Statistics
        print(f"\n📧 BASIC STATISTICS:")
        print(f"   Total Emails:          {stats['total_emails']:,}")
        print(f"   Total Threads:         {stats['total_threads']:,}")
        print(f"   Classified Emails:     {stats['total_classifications']:,}")
        
        # Calculate percentages
        if stats['total_emails'] > 0:
            classification_pct = (stats['total_classifications'] / stats['total_emails']) * 100
            print(f"   Classification Rate:   {classification_pct:.1f}%")
        
        # Date range
        if stats['earliest_email'] and stats['latest_email']:
            duration = stats['latest_email'] - stats['earliest_email']
            print(f"   Date Range:            {stats['earliest_email'].date()} → {stats['latest_email'].date()}")
            print(f"   Timeline Duration:     {duration.days:,} days")
        
        # Recent activity
        print(f"   Recent Activity:       {stats['emails_last_24h']:,} emails in last 24h")
        
        # Threading Statistics
        print(f"\n🧵 THREADING STATISTICS:")
        print(f"   Single-message threads: {stats.get('single_message_threads', 0):,}")
        print(f"   Multi-message threads:  {stats.get('multi_message_threads', 0):,}")
        
        if stats.get('avg_messages_per_thread'):
            print(f"   Avg messages/thread:    {stats['avg_messages_per_thread']:.1f}")
            print(f"   Largest thread:         {stats.get('max_messages_in_thread', 0):,} messages")
        
        # Classification Results
        if stats['classifications_by_type']:
            print(f"\n🧠 AI CLASSIFICATION RESULTS:")
            for classification in stats['classifications_by_type']:
                cls_type = classification['classification']
                count = classification['count']
                confidence = classification['avg_confidence'] or 0
                print(f"   {cls_type:12}: {count:4,} emails (avg confidence: {confidence:.2f})")
        
        # Quality Metrics
        if stats.get('avg_classification_confidence'):
            print(f"\n📈 QUALITY METRICS:")
            print(f"   Avg Classification Confidence: {stats['avg_classification_confidence']:.2f}")
            print(f"   Emails marked for processing:  {stats.get('emails_should_process', 0):,}")
            print(f"   Avg Relationship Strength:     {stats.get('avg_relationship_strength', 0):.2f}")
            print(f"   Avg Sentiment Score:           {stats.get('avg_sentiment_score', 0):.2f}")
        
        # Top Senders
        if stats['top_senders']:
            print(f"\n👥 TOP EMAIL SENDERS:")
            for i, sender in enumerate(stats['top_senders'][:5], 1):
                print(f"   {i}. {sender['from_email']:30} ({sender['email_count']:,} emails)")
        
        # Recent Classifications
        if stats['recent_classifications']:
            print(f"\n🕒 RECENT CLASSIFICATIONS:")
            for classification in stats['recent_classifications']:
                email_id = classification['email_id'][:8]
                cls_type = classification['classification'] or 'unknown'
                confidence = classification['confidence'] or 0
                sentiment = classification['sentiment'] or 'neutral'
                model = classification['model_used'] or 'unknown'
                date = classification['created_at'].strftime('%m-%d %H:%M') if classification['created_at'] else 'unknown'
                print(f"   {date} | {email_id}... | {cls_type:12} ({confidence:.2f}) | {sentiment:8} | {model}")
        
        # Processing Queue
        if stats['processing_queue']:
            print(f"\n⏳ PROCESSING QUEUE STATUS:")
            queue_summary = {}
            for item in stats['processing_queue']:
                queue_type = item['queue_type']
                status = item['status']
                count = item['count']
                
                if queue_type not in queue_summary:
                    queue_summary[queue_type] = {}
                queue_summary[queue_type][status] = count
            
            for queue_type, statuses in queue_summary.items():
                print(f"   {queue_type}:")
                for status, count in statuses.items():
                    print(f"     - {status}: {count:,}")
        
        print("\n="*80)
        print(f"📊 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error generating overview: {e}")
        sys.exit(1)

def main():
    """Main entry point"""
    print_overview()

if __name__ == "__main__":
    main()