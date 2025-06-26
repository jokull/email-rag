#!/usr/bin/env python3
"""
Mailbox Analysis Script
Analyzes existing mailbox data to understand filtering patterns and recommend optimizations.
"""

import os
import sys
import json
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

# Add the email-processor directory to the path
sys.path.insert(0, '../email-processor')

from database import get_db_session
from models import ImapMailbox, ImapMessage, Message
from sqlalchemy import text, func


def analyze_mailboxes() -> Dict[str, Any]:
    """Analyze all mailboxes and their contents"""
    results = {}
    
    with get_db_session() as session:
        print("📧 Email RAG System - Mailbox Analysis")
        print("=" * 50)
        print()
        
        # Get all mailboxes
        mailboxes = session.query(ImapMailbox).all()
        
        print(f"📁 Found {len(mailboxes)} mailboxes:")
        print()
        
        mailbox_stats = []
        total_messages = 0
        total_processed = 0
        
        for mailbox in mailboxes:
            # Count messages in this mailbox
            message_count = session.query(func.count(ImapMessage.id)).filter(
                ImapMessage.mailbox_id == mailbox.id
            ).scalar()
            
            # Count processed messages from this mailbox
            processed_count = session.query(func.count(Message.id)).join(
                ImapMessage, Message.imap_message_id == ImapMessage.id
            ).filter(
                ImapMessage.mailbox_id == mailbox.id
            ).scalar()
            
            # Get date range
            date_range = session.query(
                func.min(ImapMessage.internal_date),
                func.max(ImapMessage.internal_date)
            ).filter(
                ImapMessage.mailbox_id == mailbox.id
            ).first()
            
            min_date = date_range[0] if date_range[0] else "N/A"
            max_date = date_range[1] if date_range[1] else "N/A"
            
            # Calculate processing rate
            processing_rate = (processed_count / message_count * 100) if message_count > 0 else 0
            
            mailbox_info = {
                'name': mailbox.name,
                'message_count': message_count,
                'processed_count': processed_count,
                'processing_rate': processing_rate,
                'min_date': min_date,
                'max_date': max_date,
                'uidvalidity': mailbox.uidvalidity,
                'uidnext': mailbox.uidnext
            }
            
            mailbox_stats.append(mailbox_info)
            total_messages += message_count
            total_processed += processed_count
            
            status_icon = "✅" if processing_rate > 80 else "⚠️" if processing_rate > 50 else "❌"
            
            print(f"{status_icon} {mailbox.name}")
            print(f"   📊 Messages: {message_count:,}")
            print(f"   ✅ Processed: {processed_count:,} ({processing_rate:.1f}%)")
            print(f"   📅 Date Range: {min_date} → {max_date}")
            print()
        
        # Overall statistics
        overall_rate = (total_processed / total_messages * 100) if total_messages > 0 else 0
        
        print("📈 Overall Statistics:")
        print(f"   📧 Total Messages: {total_messages:,}")
        print(f"   ✅ Total Processed: {total_processed:,} ({overall_rate:.1f}%)")
        print()
        
        # Current filter analysis
        current_filter = os.getenv("MAILBOX_FILTER", "INBOX")
        print(f"🔍 Current Filter: {current_filter}")
        print()
        
        results['mailboxes'] = mailbox_stats
        results['total_messages'] = total_messages
        results['total_processed'] = total_processed
        results['overall_processing_rate'] = overall_rate
        results['current_filter'] = current_filter
        
        return results


def analyze_email_patterns() -> Dict[str, Any]:
    """Analyze email patterns across mailboxes"""
    patterns = {}
    
    with get_db_session() as session:
        print("🔍 Email Pattern Analysis")
        print("=" * 50)
        print()
        
        # Get sample of processed messages with their mailbox info
        sample_messages = session.query(
            Message.from_email,
            Message.subject,
            Message.category,
            Message.confidence,
            Message.date_sent,
            ImapMailbox.name.label('mailbox_name')
        ).join(
            ImapMessage, Message.imap_message_id == ImapMessage.id
        ).join(
            ImapMailbox, ImapMessage.mailbox_id == ImapMailbox.id
        ).filter(
            Message.category.isnot(None)
        ).limit(1000).all()
        
        # Analyze patterns
        category_by_mailbox = defaultdict(lambda: defaultdict(int))
        sender_patterns = defaultdict(set)
        
        for msg in sample_messages:
            category_by_mailbox[msg.mailbox_name][msg.category] += 1
            sender_patterns[msg.category].add(msg.from_email)
        
        print("📊 Category Distribution by Mailbox:")
        for mailbox, categories in category_by_mailbox.items():
            total = sum(categories.values())
            print(f"\n📁 {mailbox} ({total} messages analyzed):")
            for category, count in categories.items():
                percentage = (count / total * 100) if total > 0 else 0
                print(f"   {category}: {count} ({percentage:.1f}%)")
        
        print("\n📧 Unique Senders by Category:")
        for category, senders in sender_patterns.items():
            print(f"   {category}: {len(senders)} unique senders")
        
        patterns['category_by_mailbox'] = dict(category_by_mailbox)
        patterns['sender_counts'] = {cat: len(senders) for cat, senders in sender_patterns.items()}
        
        return patterns


def generate_recommendations(mailbox_stats: List[Dict], patterns: Dict) -> List[str]:
    """Generate recommendations based on analysis"""
    recommendations = []
    
    # Check if we're missing personal emails in other mailboxes
    inbox_stats = next((m for m in mailbox_stats if m['name'].lower() in ['inbox', 'imap']), None)
    sent_stats = next((m for m in mailbox_stats if 'sent' in m['name'].lower()), None)
    archive_stats = next((m for m in mailbox_stats if 'archive' in m['name'].lower()), None)
    
    current_filter = os.getenv("MAILBOX_FILTER", "INBOX")
    
    if current_filter != "ALL":
        recommendations.append(
            f"🔍 Currently filtering to '{current_filter}' mailbox only. "
            f"Consider expanding to capture personal emails from other mailboxes."
        )
    
    if sent_stats and sent_stats['message_count'] > 0:
        recommendations.append(
            f"📧 Found {sent_stats['message_count']:,} messages in Sent folder. "
            f"These contain your outgoing personal emails and should be included for complete conversations."
        )
    
    if archive_stats and archive_stats['message_count'] > 0:
        recommendations.append(
            f"📁 Found {archive_stats['message_count']:,} messages in Archive folder. "
            f"These may contain important personal emails that were archived."
        )
    
    # Check for unbalanced processing
    unprocessed_mailboxes = [m for m in mailbox_stats if m['message_count'] > 100 and m['processing_rate'] < 50]
    if unprocessed_mailboxes:
        recommendations.append(
            f"⚠️ Found {len(unprocessed_mailboxes)} mailboxes with low processing rates. "
            f"Consider investigating why these aren't being processed."
        )
    
    # Check for high-volume non-inbox mailboxes
    high_volume_others = [m for m in mailbox_stats if m['name'].lower() != 'inbox' and m['message_count'] > 1000]
    if high_volume_others:
        recommendations.append(
            f"📈 Found {len(high_volume_others)} non-inbox mailboxes with >1000 messages. "
            f"These may contain valuable personal emails: {', '.join(m['name'] for m in high_volume_others)}"
        )
    
    return recommendations


def main():
    """Main analysis function"""
    try:
        # Analyze mailboxes
        mailbox_results = analyze_mailboxes()
        
        # Analyze email patterns
        pattern_results = analyze_email_patterns()
        
        # Generate recommendations
        recommendations = generate_recommendations(
            mailbox_results['mailboxes'], 
            pattern_results
        )
        
        print("💡 Recommendations:")
        print("=" * 50)
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
                print()
        else:
            print("✅ No specific recommendations - current setup looks optimal!")
            print()
        
        # Suggested configuration
        print("⚙️ Suggested Configuration Changes:")
        print("=" * 50)
        
        # Check if we should expand beyond INBOX
        current_filter = os.getenv("MAILBOX_FILTER", "INBOX")
        sent_messages = sum(m['message_count'] for m in mailbox_results['mailboxes'] if 'sent' in m['name'].lower())
        archive_messages = sum(m['message_count'] for m in mailbox_results['mailboxes'] if 'archive' in m['name'].lower())
        
        if current_filter != "ALL" and (sent_messages > 0 or archive_messages > 0):
            print("Consider updating your .env file:")
            print("MAILBOX_FILTER=ALL")
            print()
            print("Or to include specific mailboxes:")
            important_mailboxes = [m['name'] for m in mailbox_results['mailboxes'] 
                                 if m['message_count'] > 100 and 
                                 ('sent' in m['name'].lower() or 'archive' in m['name'].lower() or 'inbox' in m['name'].lower())]
            print(f"MAILBOX_FILTER={','.join(important_mailboxes)}")
            print()
        
        # Save results to file
        results_file = 'mailbox_analysis_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'analysis_date': datetime.now().isoformat(),
                'mailbox_stats': mailbox_results,
                'email_patterns': pattern_results,
                'recommendations': recommendations,
                'current_filter': current_filter
            }, f, indent=2, default=str)
        
        print(f"📄 Detailed results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()