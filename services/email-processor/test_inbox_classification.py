#!/usr/bin/env python3
"""
Test LLM email classification on 20 real emails from Inbox only
Logs full prompts and outputs to file
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


def setup_logging():
    """Setup logging to file and console"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"inbox_classification_test_{timestamp}.log"
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file


def test_classification():
    """Test classification on 20 real emails from Inbox only"""
    
    print("🧪 Testing LLM Email Classification on Inbox emails...")
    
    # Initialize classifier
    if not initialize_llm_classifier():
        print("❌ Failed to initialize LLM classifier")
        return
    
    classifier = get_llm_classifier()
    print(f"✅ Using model: {classifier.model_name}")
    
    # Get 20 test emails from Inbox only  
    with get_db_session() as session:
        # Join with imap_messages and imap_mailboxes to filter by Inbox
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
        ).order_by(desc(Message.date_sent)).limit(20)
        
        # Extract the data we need while in session
        test_data = []
        for msg in messages_query:
            test_data.append({
                'imap_message_id': msg.imap_message_id,
                'from_email': msg.from_email,
                'subject': msg.subject or '',
                'body_text': msg.body_text or '',
                'category': msg.category
            })
    
    if not test_data:
        print("❌ No test messages found in Inbox")
        return
    
    # Setup detailed logging to file
    timestamp = datetime.now().isoformat()
    log_filename = f"inbox_classification_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    results = []
    
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        # Write header
        log_file.write("📧 INBOX EMAIL CLASSIFICATION TEST LOG\n")
        log_file.write(f"Generated: {timestamp}\n")
        log_file.write(f"LLM Model: {classifier.model_name}\n")
        log_file.write(f"My Email: {classifier.my_email}\n")
        log_file.write(f"Total Test Cases: {len(test_data)}\n")
        log_file.write("=" * 80 + "\n\n")
        
        for i, message in enumerate(test_data, 1):
            test_start = datetime.now().isoformat()
            
            log_file.write("=" * 80 + "\n")
            log_file.write(f"TEST #{i} - {test_start}\n")
            log_file.write("=" * 80 + "\n")
            log_file.write(f"MESSAGE ID: {message['imap_message_id']}\n")
            log_file.write(f"FROM: {message['from_email']}\n")
            log_file.write(f"SUBJECT: {message['subject'][:70]}{'...' if len(message['subject']) > 70 else ''}\n")
            log_file.write(f"BODY PREVIEW: {message['body_text'][:200]}{'...' if len(message['body_text']) > 200 else ''}\n")
            log_file.write(f"EXISTING CATEGORY: {message['category']}\n")
            log_file.write("\n")
            
            try:
                # Test classification
                result = classifier.classify_email(
                    from_email=message['from_email'],
                    subject=message['subject'],
                    body=message['body_text'][:500]  # Limit body size
                )
                
                if result:
                    log_file.write(f"FULL PROMPT SENT TO LLM:\n")
                    log_file.write("-" * 40 + "\n")
                    
                    # Get the correspondence count to recreate the prompt
                    outbound_count = classifier._get_outbound_email_count(message['from_email'])
                    log_file.write(f"CORRESPONDENCE DATA: {outbound_count} outbound emails to this sender\n\n")
                    
                    # Check if this would be caught by obvious patterns
                    obvious_result = classifier._check_obvious_automated(message['from_email'])
                    if obvious_result:
                        log_file.write(f"🚫 OBVIOUS AUTOMATED PATTERN DETECTED: {obvious_result.reasoning}\n")
                        log_file.write(f"CLASSIFICATION RESULT: {obvious_result.category}\n")
                        log_file.write(f"CONFIDENCE: {obvious_result.confidence}\n")
                        log_file.write(f"PROCESSING TIME: {obvious_result.processing_time_ms:.1f}ms\n")
                    else:
                        # Recreate the full prompt for logging
                        full_prompt = classifier._create_classification_prompt(
                            message['from_email'], message['subject'], message['body_text'][:250]
                        )
                        log_file.write(full_prompt + "\n")
                        log_file.write("-" * 40 + "\n\n")
                        
                        log_file.write(f"LLM MODEL USED: {classifier.model_name}\n")
                        log_file.write(f"CLASSIFICATION RESULT: {result.category}\n")
                        log_file.write(f"CONFIDENCE: {result.confidence}\n")
                        log_file.write(f"PROCESSING TIME: {result.processing_time_ms:.1f}ms\n")
                        log_file.write(f"FULL LLM RESPONSE:\n{result.reasoning}\n")
                    
                    # Track results
                    results.append({
                        'message_id': message['imap_message_id'],
                        'from_email': message['from_email'],
                        'existing_category': message['category'],
                        'new_category': result.category,
                        'confidence': result.confidence,
                        'match': message['category'] == result.category if message['category'] else 'unknown'
                    })
                    
                    print(f"✅ {i:2d}/20 - {message['from_email'][:30]:30} -> {result.category:12} ({result.confidence:.2f})")
                    
                else:
                    log_file.write("❌ CLASSIFICATION FAILED\n")
                    print(f"❌ {i:2d}/20 - {message['from_email'][:30]} -> FAILED")
                    
            except Exception as e:
                log_file.write(f"❌ ERROR: {str(e)}\n")
                print(f"❌ {i:2d}/20 - {message['from_email'][:30]} -> ERROR: {e}")
            
            log_file.write("\n" + "=" * 80 + "\n\n")
    
    # Print summary
    print(f"\n📊 Test Results Summary:")
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
    
    print(f"\n📄 Detailed log saved to: {log_filename}")
    return log_filename


if __name__ == "__main__":
    test_classification()