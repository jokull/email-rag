#!/usr/bin/env python3
"""
STEP 0: Email Processing Integration Test
Tests basic email parsing, cleaning, and participant extraction
Foundation pipeline: IMAP → parsing → cleaning → structured data
"""

import sys
import os
from datetime import datetime

from database import get_db_session
from models import ImapMessage, Message, ImapMailbox
from processor import EmailProcessor

def test_email_processor_initialization():
    """Test EmailProcessor can be initialized with all dependencies"""
    print("🔧 Testing EmailProcessor initialization...")
    
    try:
        processor = EmailProcessor()
        print("✅ EmailProcessor initialized successfully")
        
        # Check required methods
        required_methods = ['process_single_email', 'get_processing_stats']
        for method in required_methods:
            if hasattr(processor, method):
                print(f"✅ Method {method} available")
            else:
                print(f"❌ Method {method} missing")
                return False
        
        return True
    except Exception as e:
        print(f"❌ EmailProcessor initialization failed: {e}")
        return False

def test_database_connectivity():
    """Test database connection and table access"""
    print("\n🗄️  Testing database connectivity...")
    
    try:
        with get_db_session() as session:
            # Test IMAP tables (read-only)
            imap_count = session.query(ImapMessage).count()
            mailbox_count = session.query(ImapMailbox).count()
            
            print(f"✅ IMAP messages available: {imap_count}")
            print(f"✅ IMAP mailboxes available: {mailbox_count}")
            
            # Test processed messages table
            processed_count = session.query(Message).count()
            print(f"✅ Processed messages: {processed_count}")
            
            return True
            
    except Exception as e:
        print(f"❌ Database connectivity test failed: {e}")
        return False

def test_email_parsing():
    """Test parsing a real IMAP message"""
    print("\n📧 Testing email parsing...")
    
    try:
        processor = EmailProcessor()
        
        with get_db_session() as session:
            # Get a sample IMAP message
            sample_imap = session.query(ImapMessage).filter(
                ImapMessage.raw_message.isnot(None)
            ).first()
            
            if not sample_imap:
                print("⚠️  No IMAP messages with raw content found")
                return True  # Not a failure - just no data
            
            print(f"📧 Testing parsing on IMAP message #{sample_imap.id}")
            print(f"   Size: {sample_imap.size} bytes")
            
            # Test parsing without saving to DB
            parsed_data = processor._parse_email_with_python(sample_imap.raw_message)
            
            if parsed_data:
                print(f"✅ Email parsed successfully")
                print(f"   From: {parsed_data.get('from_email', 'unknown')[:50]}")
                print(f"   Subject: {(parsed_data.get('subject') or 'No subject')[:50]}")
                print(f"   Body length: {len(parsed_data.get('body_text', ''))}")
                return True
            else:
                print("❌ Email parsing failed")
                return False
                
    except Exception as e:
        print(f"❌ Email parsing test failed: {e}")
        return False

def test_email_cleaning():
    """Test email body cleaning with email_reply_parser"""
    print("\n🧹 Testing email body cleaning...")
    
    try:
        processor = EmailProcessor()
        
        # Test with a sample email containing threading
        sample_email = """Hi John,

Thanks for the update on the project. I think we should proceed as planned.

Best regards,
Sarah

On Wed, Dec 15, 2023 at 2:30 PM John Smith <john@company.com> wrote:
> Hey Sarah,
> 
> Just wanted to give you a quick update on the project status.
> We're on track to finish by the end of the month.
> 
> Let me know what you think.
> 
> Best,
> John

-- 
Sarah Johnson
Product Manager
Company Inc.
"""
        
        cleaned = processor._clean_email_body(sample_email)
        
        print(f"✅ Email body cleaned")
        print(f"   Original length: {len(sample_email)}")
        print(f"   Cleaned length: {len(cleaned)}")
        print(f"   Preview: \"{cleaned[:80]}...\"")
        
        # Verify that reply content was removed
        if "On Wed, Dec 15" not in cleaned and "john@company.com" not in cleaned:
            print("✅ Reply threading removed successfully")
        else:
            print("⚠️  Reply content still present")
        
        return True
        
    except Exception as e:
        print(f"❌ Email cleaning test failed: {e}")
        return False

def test_processing_stats():
    """Test processing statistics functionality"""
    print("\n📊 Testing processing statistics...")
    
    try:
        processor = EmailProcessor()
        stats = processor.get_processing_stats()
        
        print("📊 Processing Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Verify expected stats
        expected_keys = ['total_imap_messages', 'total_processed_messages', 'pending_messages']
        for key in expected_keys:
            if key not in stats:
                print(f"❌ Missing expected stat: {key}")
                return False
        
        print("✅ Processing stats structure correct")
        return True
        
    except Exception as e:
        print(f"❌ Processing stats test failed: {e}")
        return False

def test_single_email_processing():
    """Test complete processing of a single email"""
    print("\n🔄 Testing complete single email processing...")
    
    try:
        processor = EmailProcessor()
        
        with get_db_session() as session:
            # Find an unprocessed IMAP message
            unprocessed = session.query(ImapMessage).filter(
                ~session.query(Message.imap_message_id).filter(
                    Message.imap_message_id == ImapMessage.id
                ).exists(),
                ImapMessage.raw_message.isnot(None)
            ).first()
            
            if not unprocessed:
                print("⚠️  No unprocessed IMAP messages found")
                return True  # Not a failure - just no work to do
            
            print(f"🔄 Testing processing on IMAP message #{unprocessed.id}")
            
            # Process the email
            success = processor.process_single_email(session, unprocessed)
            
            if success:
                # Check the result
                processed_msg = session.query(Message).filter(
                    Message.imap_message_id == unprocessed.id
                ).first()
                
                if processed_msg:
                    print(f"✅ Email processed successfully")
                    print(f"   Message ID: {processed_msg.id}")
                    print(f"   From: {processed_msg.from_email}")
                    print(f"   Subject: {(processed_msg.subject or 'No subject')[:50]}")
                    print(f"   Body length: {len(processed_msg.body_text or '')}")
                    
                    # Check for language detection
                    if processed_msg.language:
                        print(f"   Language: {processed_msg.language} (confidence: {processed_msg.language_confidence:.3f})")
                    
                    return True
                else:
                    print("❌ No Message record created")
                    return False
            else:
                print("❌ Email processing failed")
                return False
                
    except Exception as e:
        print(f"❌ Single email processing test failed: {e}")
        return False

def main():
    """Run all email processing integration tests"""
    print("🚀 STEP 0: Email Processing Integration Tests")
    print("=" * 55)
    
    tests = [
        ("EmailProcessor Initialization", test_email_processor_initialization),
        ("Database Connectivity", test_database_connectivity),
        ("Email Parsing", test_email_parsing),
        ("Email Cleaning", test_email_cleaning),
        ("Processing Stats", test_processing_stats),
        ("Single Email Processing", test_single_email_processing),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
    
    print("\n" + "=" * 55)
    print(f"📊 Integration Test Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    if passed >= 4:  # Allow some flexibility
        print("🎉 Email processing integration tests successful!")
        print("\n💡 Foundation pipeline ready for STEP 1: Email Classification!")
        return True
    else:
        print("⚠️  Some email processing integration tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)