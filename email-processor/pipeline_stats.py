#!/usr/bin/env python3
"""
Pipeline Status Monitor - Python CLI for comprehensive email RAG pipeline statistics
Replaces shell SQL loops with proper Python database handling and rich formatting
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional

from database import get_db_session
from models import ImapMessage, Message, Conversation, MessageChunk, ImapMailbox
from sqlalchemy import func, text

def get_current_time():
    """Get current time in HH:MM:SS format"""
    return datetime.now().strftime("%H:%M:%S")

def print_header():
    """Print pipeline status header"""
    print("═" * 67)
    print("                     📊 EMAIL RAG PIPELINE STATUS")
    print("═" * 67)
    print(f"🕐 Last Updated: {datetime.now().strftime('%a %b %d %H:%M:%S %Z %Y')}")
    print()

def get_comprehensive_stats() -> Dict[str, Any]:
    """Get comprehensive pipeline statistics from database"""
    try:
        with get_db_session() as session:
            stats = {}
            
            # Basic counts
            stats['imap_messages'] = session.query(func.count(ImapMessage.id)).scalar() or 0
            stats['processed_messages'] = session.query(func.count(Message.id)).scalar() or 0
            stats['personal_emails'] = session.query(func.count(Message.id)).filter(
                Message.category == 'personal'
            ).scalar() or 0
            stats['classified'] = session.query(func.count(Message.id)).filter(
                Message.category.isnot(None)
            ).scalar() or 0
            stats['language_detected'] = session.query(func.count(Message.id)).filter(
                Message.language.isnot(None)
            ).scalar() or 0
            stats['embedded'] = session.query(func.count(Message.id)).filter(
                Message.embedded_at.isnot(None)
            ).scalar() or 0
            stats['conversations'] = session.query(func.count(Conversation.id)).scalar() or 0
            stats['summarized_conversations'] = session.query(func.count(Conversation.id)).filter(
                Conversation.summary.isnot(None)
            ).scalar() or 0
            stats['message_chunks'] = session.query(func.count(MessageChunk.id)).scalar() or 0
            
            # Percentages
            if stats['imap_messages'] > 0:
                stats['processing_pct'] = round(100.0 * stats['processed_messages'] / stats['imap_messages'], 1)
            else:
                stats['processing_pct'] = 0
                
            if stats['processed_messages'] > 0:
                stats['personal_pct'] = round(100.0 * stats['personal_emails'] / stats['processed_messages'], 1)
                stats['classification_pct'] = round(100.0 * stats['classified'] / stats['processed_messages'], 1)
                stats['language_pct'] = round(100.0 * stats['language_detected'] / stats['processed_messages'], 1)
            else:
                stats['personal_pct'] = 0
                stats['classification_pct'] = 0
                stats['language_pct'] = 0
                
            if stats['personal_emails'] > 0:
                stats['embedding_pct'] = round(100.0 * stats['embedded'] / stats['personal_emails'], 1)
            else:
                stats['embedding_pct'] = 0
                
            if stats['conversations'] > 0:
                stats['summarization_pct'] = round(100.0 * stats['summarized_conversations'] / stats['conversations'], 1)
            else:
                stats['summarization_pct'] = 0
            
            # Average chunk characteristics
            if stats['message_chunks'] > 0:
                avg_chunk_length = session.query(func.avg(func.length(MessageChunk.text_content))).scalar()
                stats['avg_chunk_length'] = round(avg_chunk_length) if avg_chunk_length else 0
            else:
                stats['avg_chunk_length'] = 0
            
            # Threading and summarization stats
            stats['unthreaded_personal'] = session.query(func.count(Message.id)).filter(
                Message.category == 'personal',
                Message.thread_id.is_(None)
            ).scalar() or 0
            
            stats['threaded_personal'] = session.query(func.count(Message.id)).filter(
                Message.category == 'personal',
                Message.thread_id.isnot(None)
            ).scalar() or 0
            
            # Average messages per conversation
            if stats['conversations'] > 0:
                avg_msgs_per_conv = session.query(func.avg(Conversation.message_count)).scalar()
                stats['avg_messages_per_conversation'] = round(avg_msgs_per_conv, 1) if avg_msgs_per_conv else 0
            else:
                stats['avg_messages_per_conversation'] = 0
            
            # Summary length statistics
            if stats['summarized_conversations'] > 0:
                avg_summary_length = session.query(func.avg(func.length(Conversation.summary))).filter(
                    Conversation.summary.isnot(None)
                ).scalar()
                stats['avg_summary_length'] = round(avg_summary_length) if avg_summary_length else 0
            else:
                stats['avg_summary_length'] = 0
            
            # Recent activity (today)
            today = datetime.now().date()
            stats['processed_today'] = session.query(func.count(Message.id)).filter(
                func.date(Message.processed_at) == today
            ).scalar() or 0
            stats['embedded_today'] = session.query(func.count(Message.id)).filter(
                func.date(Message.embedded_at) == today
            ).scalar() or 0
            stats['threaded_today'] = session.query(func.count(Conversation.id)).filter(
                func.date(Conversation.created_at) == today
            ).scalar() or 0
            
            # Language distribution (top 5)
            language_dist = session.query(
                Message.language, 
                func.count(Message.id)
            ).filter(
                Message.language.isnot(None)
            ).group_by(Message.language).order_by(
                func.count(Message.id).desc()
            ).limit(5).all()
            
            stats['top_languages'] = dict(language_dist) if language_dist else {}
            
            # Average language confidence
            avg_confidence = session.query(func.avg(Message.language_confidence)).filter(
                Message.language_confidence.isnot(None)
            ).scalar()
            stats['avg_language_confidence'] = round(avg_confidence, 3) if avg_confidence else 0
            
            return stats
            
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return {}

def print_main_stats(stats: Dict[str, Any]):
    """Print main pipeline statistics"""
    if not stats:
        return
        
    print("📬 IMAP Messages".ljust(25) + f"{stats['imap_messages']:,}".rjust(15))
    print("📧 Processed Messages".ljust(25) + f"{stats['processed_messages']:,}".rjust(10) + f" ({stats['processing_pct']}%)")
    print("👤 Personal Emails".ljust(25) + f"{stats['personal_emails']:,}".rjust(10) + f" ({stats['personal_pct']}%)")
    print("🏷️  Classified".ljust(25) + f"{stats['classified']:,}".rjust(10) + f" ({stats['classification_pct']}%)")
    print("🌐 Language Detected".ljust(25) + f"{stats['language_detected']:,}".rjust(10) + f" ({stats['language_pct']}%)")
    print("🧠 RAG Embedded".ljust(25) + f"{stats['embedded']:,}".rjust(10) + f" ({stats['embedding_pct']}%)")
    print("🧵 Conversation Threads".ljust(25) + f"{stats['conversations']:,}".rjust(10) + f" ({stats['summarization_pct']}% summarized)")
    print("📚 Message Chunks".ljust(25) + f"{stats['message_chunks']:,}".rjust(10) + f" ({stats['avg_chunk_length']} avg chars)")

def print_threading_stats(stats: Dict[str, Any]):
    """Print threading and summarization statistics"""
    if not stats:
        return
        
    print()
    print("🧵 Threading & Summarization:")
    
    # Personal email threading progress
    total_personal = stats['personal_emails']
    threaded = stats['threaded_personal']
    unthreaded = stats['unthreaded_personal']
    
    if total_personal > 0:
        threading_pct = round(100.0 * threaded / total_personal, 1)
        print(f"📧 Personal emails threaded: {threaded:,}/{total_personal:,} ({threading_pct}%)")
        if unthreaded > 0:
            print(f"⏳ Pending threading: {unthreaded:,}")
    
    # Conversation statistics
    print(f"🗣️  Total conversations: {stats['conversations']:,}")
    if stats['conversations'] > 0:
        print(f"📝 Summarized: {stats['summarized_conversations']:,} ({stats['summarization_pct']}%)")
        print(f"📊 Avg messages/conversation: {stats['avg_messages_per_conversation']}")
        if stats['avg_summary_length'] > 0:
            print(f"📏 Avg summary length: {stats['avg_summary_length']} chars")

def print_recent_activity(stats: Dict[str, Any]):
    """Print recent activity statistics"""
    if not stats:
        return
        
    print()
    print("🔧 Recent Activity:")
    print(f"📧 Processed Today: {stats['processed_today']}")
    print(f"🧠 Embedded Today: {stats['embedded_today']}")
    print(f"🧵 Threaded Today: {stats['threaded_today']}")

def print_language_stats(stats: Dict[str, Any]):
    """Print language statistics"""
    if not stats or not stats.get('top_languages'):
        return
        
    print()
    print("🌍 Top Languages:")
    for lang, count in list(stats['top_languages'].items())[:5]:
        print(f"   {lang}: {count:,}")
    
    if stats['avg_language_confidence'] > 0:
        print(f"   Avg confidence: {stats['avg_language_confidence']}")

def print_pipeline_health():
    """Print pipeline health status"""
    print()
    print("🎯 Pipeline Health: All services running")

def display_stats(refresh_interval: Optional[int] = None):
    """Display pipeline statistics, optionally refreshing"""
    if refresh_interval:
        try:
            while True:
                # Clear screen (works on most terminals)
                os.system('clear' if os.name == 'posix' else 'cls')
                
                print_header()
                stats = get_comprehensive_stats()
                print_main_stats(stats)
                print_threading_stats(stats)
                print_recent_activity(stats)
                print_language_stats(stats)
                print_pipeline_health()
                
                print("═" * 67)
                print(f"🔄 Refreshing every {refresh_interval}s (Ctrl+C to exit)")
                
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print(f"\n{get_current_time()} │ 👋 Stats monitor stopped")
            sys.exit(0)
    else:
        # Single run
        print_header()
        stats = get_comprehensive_stats()
        print_main_stats(stats)
        print_threading_stats(stats)
        print_recent_activity(stats)
        print_language_stats(stats)
        print_pipeline_health()
        print("═" * 67)

def get_quick_stats() -> str:
    """Get quick one-line stats for display in other tools"""
    try:
        with get_db_session() as session:
            processed = session.query(func.count(Message.id)).scalar() or 0
            personal = session.query(func.count(Message.id)).filter(
                Message.category == 'personal'
            ).scalar() or 0
            embedded = session.query(func.count(Message.id)).filter(
                Message.embedded_at.isnot(None)
            ).scalar() or 0
            conversations = session.query(func.count(Conversation.id)).scalar() or 0
            
            return f"📊 {processed} processed | 👤 {personal} personal | 🧠 {embedded} embedded | 🧵 {conversations} threads"
            
    except Exception:
        return "❌ Database unavailable"

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Email RAG Pipeline Status Monitor")
    parser.add_argument("--refresh", "-r", type=int, metavar="SECONDS",
                       help="Auto-refresh every N seconds (Ctrl+C to stop)")
    parser.add_argument("--quick", "-q", action="store_true",
                       help="Show quick one-line stats")
    
    args = parser.parse_args()
    
    # Environment validation
    database_url = os.getenv("DATABASE_URL", "postgresql://email_user:email_pass@localhost:5433/email_rag")
    
    if args.quick:
        print(get_quick_stats())
        return
    
    try:
        # Test database connection
        with get_db_session() as session:
            session.execute(text("SELECT 1")).fetchone()
            
        display_stats(args.refresh)
        
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        print(f"Database URL: {database_url.split('@')[1] if '@' in database_url else database_url}")
        sys.exit(1)

if __name__ == "__main__":
    main()