#!/usr/bin/env python3
"""
Classification Analysis Script
Shows senders and body snippets for automated/promotional emails to assess false positives
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple
import argparse

# Add the email-processor directory to path
sys.path.append('/Users/jokull/Code/email-rag/services/email-processor')

from database import get_db_session
from models import Message

def analyze_classifications(category: str, limit: int = 50) -> List[Tuple[str, str, str, str, datetime]]:
    """Get sample of classified emails with sender, subject, body snippet"""
    with get_db_session() as session:
        messages = session.query(Message).filter(
            Message.category == category,
            Message.body_text.isnot(None)
        ).order_by(Message.date_sent.desc()).limit(limit).all()
        
        results = []
        for msg in messages:
            # Clean up body text - first 200 chars
            body_snippet = (msg.body_text or "").strip()
            if len(body_snippet) > 200:
                body_snippet = body_snippet[:197] + "..."
            
            # Clean up subject
            subject = (msg.subject or "No Subject").strip()
            if len(subject) > 80:
                subject = subject[:77] + "..."
                
            results.append((
                msg.from_email or "Unknown Sender",
                subject,
                body_snippet,
                f"{msg.confidence:.2f}" if msg.confidence else "N/A",
                msg.date_sent or datetime.min
            ))
        
        return results

def get_sender_stats(category: str) -> List[Tuple[str, int, float]]:
    """Get sender statistics for a category"""
    with get_db_session() as session:
        from sqlalchemy import func
        
        stats = session.query(
            Message.from_email,
            func.count(Message.id).label('count'),
            func.avg(Message.confidence).label('avg_confidence')
        ).filter(
            Message.category == category
        ).group_by(
            Message.from_email
        ).order_by(
            func.count(Message.id).desc()
        ).limit(20).all()
        
        return [(sender or "Unknown", count, avg_conf or 0.0) for sender, count, avg_conf in stats]

def print_category_analysis(category: str, limit: int = 20):
    """Print comprehensive analysis for a category"""
    print(f"\n{'='*80}")
    print(f"📧 {category.upper()} EMAILS ANALYSIS")
    print(f"{'='*80}")
    
    # Get sender stats
    print(f"\n📊 TOP SENDERS ({category.upper()}):")
    print("-" * 60)
    print(f"{'SENDER':<35} {'COUNT':<8} {'AVG_CONF':<10}")
    print("-" * 60)
    
    sender_stats = get_sender_stats(category)
    for sender, count, avg_conf in sender_stats[:15]:
        sender_display = sender[:34] if len(sender) <= 34 else sender[:31] + "..."
        print(f"{sender_display:<35} {count:<8} {avg_conf:<10.2f}")
    
    # Get sample emails
    print(f"\n📝 SAMPLE EMAILS ({category.upper()}):")
    print("-" * 80)
    
    samples = analyze_classifications(category, limit)
    for i, (sender, subject, body, confidence, date) in enumerate(samples, 1):
        print(f"\n{i}. FROM: {sender}")
        print(f"   SUBJECT: {subject}")
        print(f"   DATE: {date.strftime('%Y-%m-%d %H:%M') if date != datetime.min else 'Unknown'}")
        print(f"   CONFIDENCE: {confidence}")
        print(f"   BODY: {body}")
        print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description='Analyze email classifications')
    parser.add_argument('--category', choices=['automated', 'promotional', 'personal', 'all'], 
                       default='all', help='Category to analyze')
    parser.add_argument('--limit', type=int, default=20, 
                       help='Number of sample emails to show per category')
    parser.add_argument('--senders-only', action='store_true',
                       help='Show only sender statistics, no email samples')
    
    args = parser.parse_args()
    
    print("🔍 EMAIL CLASSIFICATION ANALYSIS")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get overall stats
    with get_db_session() as session:
        from sqlalchemy import func
        total_classified = session.query(func.count(Message.id)).filter(
            Message.category.isnot(None)
        ).scalar()
        
        category_stats = session.query(
            Message.category,
            func.count(Message.id)
        ).filter(
            Message.category.isnot(None)
        ).group_by(Message.category).all()
    
    print(f"\n📈 OVERALL CLASSIFICATION STATS:")
    print(f"Total classified emails: {total_classified}")
    for category, count in category_stats:
        percentage = (count / total_classified * 100) if total_classified > 0 else 0
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    # Analyze specific categories
    if args.category == 'all':
        categories = ['automated', 'promotional', 'personal']
    else:
        categories = [args.category]
    
    for category in categories:
        if args.senders_only:
            print(f"\n📊 TOP SENDERS ({category.upper()}):")
            print("-" * 60)
            print(f"{'SENDER':<35} {'COUNT':<8} {'AVG_CONF':<10}")
            print("-" * 60)
            
            sender_stats = get_sender_stats(category)
            for sender, count, avg_conf in sender_stats:
                sender_display = sender[:34] if len(sender) <= 34 else sender[:31] + "..."
                print(f"{sender_display:<35} {count:<8} {avg_conf:<10.2f}")
        else:
            print_category_analysis(category, args.limit)
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()