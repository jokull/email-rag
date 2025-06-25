#!/usr/bin/env python3
"""
Test LLM email classification on 20 emails that get PAST the first-line defense
This tests the harder cases where LLM + correspondence intelligence is needed
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from database import get_db_session
from models import Message, ImapMessage, ImapMailbox
from llm_classifier import get_llm_classifier, initialize_llm_classifier
from sqlalchemy import desc, and_


def would_pass_first_line_defense(from_email: str) -> bool:
    """Check if email would pass first-line defense (i.e., NOT caught by obvious patterns)"""
    from_email_lower = from_email.lower()
    
    # Exact prefix patterns for automated emails
    automated_prefixes = [
        'noreply@', 'no-reply@', 'notifications@', 'notification@',
        'donotreply@', 'do-not-reply@', 'support@', 'help@',
        'alerts@', 'alert@', 'system@', 'admin@', 'postmaster@',
        'outbound@', 'hello@'
    ]
    
    # Check exact prefixes
    for pattern in automated_prefixes:
        if from_email_lower.startswith(pattern):
            return False  # Would be caught by first-line defense
    
    # Check contains patterns
    automated_contains = ['noreply', 'invoice']
    
    for pattern in automated_contains:
        if pattern in from_email_lower:
            return False  # Would be caught by first-line defense
    
    return True  # Would pass to LLM classification


def test_llm_classification():
    """Test LLM classification on 20 emails that get past first-line defense"""
    
    print("🎯 Testing LLM Email Classification on emails that PASS first-line defense...")
    
    # Initialize classifier
    if not initialize_llm_classifier():
        print("❌ Failed to initialize LLM classifier")
        return
    
    classifier = get_llm_classifier()
    print(f"✅ Using model: {classifier.model_name}")
    
    # Get emails from Inbox that would pass first-line defense
    with get_db_session() as session:
        # Get a larger set to filter from
        messages_query = session.query(Message).join(
            ImapMessage, Message.imap_message_id == ImapMessage.id
        ).join(
            ImapMailbox, ImapMessage.mailbox_id == ImapMailbox.id
        ).filter(
            and_(
                ImapMailbox.name == 'INBOX',  # Only Inbox messages
                Message.from_email.isnot(None),
                Message.subject.isnot(None),
                Message.body_text.isnot(None)
            )
        ).order_by(desc(Message.date_sent)).limit(100)  # Get more to filter from
        
        # Extract data while filtering for emails that pass first-line defense
        all_test_data = []
        for msg in messages_query:
            if would_pass_first_line_defense(msg.from_email):
                all_test_data.append({
                    'imap_message_id': msg.imap_message_id,
                    'from_email': msg.from_email,
                    'subject': msg.subject or '',
                    'body_text': msg.body_text or '',
                    'category': msg.category
                })
        
        # Take first 20 that pass the filter
        test_data = all_test_data[:20]
    
    if not test_data:
        print("❌ No test messages found that pass first-line defense")
        return
    
    print(f"🔍 Found {len(test_data)} emails that pass first-line defense (from {len(all_test_data)} candidates)")
    
    # Setup detailed logging to file
    timestamp = datetime.now().isoformat()
    log_filename = f"llm_only_classification_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    results = []
    
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        # Write simple header
        log_file.write(f"# LLM Classification Test - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        log_file.write(f"# Model: {classifier.model_name}\n\n")
        
        for i, message in enumerate(test_data, 1):            
            try:
                # Test classification - this should go straight to LLM since it passed first-line defense
                result = classifier.classify_email(
                    from_email=message['from_email'],
                    subject=message['subject'],
                    body=message['body_text'][:500]  # Limit body size
                )
                
                if result:
                    # Get correspondence data for tracking
                    outbound_count = classifier._get_outbound_email_count(message['from_email'])
                    
                    # Format body preview (max 5 lines)
                    body_lines = message['body_text'].replace('\r\n', '\n').replace('\r', '\n').split('\n')
                    preview_lines = []
                    for line in body_lines[:5]:
                        if line.strip():  # Skip empty lines
                            preview_lines.append(line.strip()[:80])  # Max 80 chars per line
                        if len(preview_lines) >= 5:
                            break
                    preview = '\n'.join(preview_lines)
                    if len(body_lines) > 5:
                        preview += "\n..."
                    
                    # Write concise log entry
                    log_file.write(f"# Test {i}\n")
                    log_file.write(f"From: {message['from_email']}\n")
                    log_file.write(f"Preview: {preview}\n")
                    log_file.write(f"Result: {result.category}\n\n")
                    
                    # Track results
                    results.append({
                        'message_id': message['imap_message_id'],
                        'from_email': message['from_email'],
                        'existing_category': message['category'],
                        'new_category': result.category,
                        'confidence': result.confidence,
                        'outbound_count': outbound_count,
                        'match': message['category'] == result.category if message['category'] else 'unknown'
                    })
                    
                    # Show correspondence insight in console
                    correspondence_signal = f"({outbound_count} sent)" if outbound_count > 0 else "(0 sent)"
                    print(f"✅ {i:2d}/20 - {message['from_email'][:25]:25} -> {result.category:12} {correspondence_signal} ({result.confidence:.2f})")
                    
                else:
                    log_file.write(f"# Test {i}\n")
                    log_file.write(f"From: {message['from_email']}\n")
                    log_file.write(f"Result: FAILED\n\n")
                    print(f"❌ {i:2d}/20 - {message['from_email'][:30]} -> FAILED")
                    
            except Exception as e:
                log_file.write(f"# Test {i}\n")
                log_file.write(f"From: {message['from_email']}\n")
                log_file.write(f"Result: ERROR - {str(e)}\n\n")
                print(f"❌ {i:2d}/20 - {message['from_email'][:30]} -> ERROR: {e}")
    
    # Print detailed summary
    print(f"\n📊 LLM Classification Results Summary:")
    print(f"Total emails tested: {len(results)}")
    
    if results:
        matches = sum(1 for r in results if r['match'] == True)
        total_with_category = sum(1 for r in results if r['existing_category'] is not None)
        
        if total_with_category > 0:
            accuracy = (matches / total_with_category) * 100
            print(f"Accuracy: {matches}/{total_with_category} = {accuracy:.1f}%")
        
        # Category breakdown
        from collections import Counter
        new_categories = Counter(r['new_category'] for r in results)
        print(f"New classifications: {dict(new_categories)}")
        
        # Correspondence analysis
        personal_with_correspondence = sum(1 for r in results if r['new_category'] == 'personal' and r['outbound_count'] > 0)
        personal_without_correspondence = sum(1 for r in results if r['new_category'] == 'personal' and r['outbound_count'] == 0)
        automated_with_correspondence = sum(1 for r in results if r['new_category'] == 'automated' and r['outbound_count'] > 0)
        promotional_zero_correspondence = sum(1 for r in results if r['new_category'] == 'promotional' and r['outbound_count'] == 0)
        
        print(f"\n📈 Correspondence Intelligence Analysis:")
        print(f"Personal with correspondence (expected): {personal_with_correspondence}")
        print(f"Personal without correspondence (suspicious): {personal_without_correspondence}")
        print(f"Automated with correspondence (suspicious): {automated_with_correspondence}")
        print(f"Promotional with zero correspondence (expected): {promotional_zero_correspondence}")
        
        # Show some examples
        print(f"\n🔍 Examples by category:")
        for category in ['personal', 'promotional', 'automated']:
            examples = [r for r in results if r['new_category'] == category][:3]
            if examples:
                print(f"\n{category.upper()}:")
                for ex in examples:
                    print(f"  • {ex['from_email'][:40]:40} (sent {ex['outbound_count']} times)")
    
    print(f"\n📄 Detailed log saved to: {log_filename}")
    return log_filename


if __name__ == "__main__":
    test_llm_classification()