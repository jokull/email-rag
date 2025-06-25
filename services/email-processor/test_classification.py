#!/usr/bin/env python3
"""
Test script for email classification
Tests the SetFit classifier with sample emails
"""

import logging
import sys
from setfit_classifier import initialize_classifier, get_classifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test email samples
TEST_EMAILS = [
    {
        "subject": "Re: Meeting tomorrow",
        "content": "Hi John,\n\nThanks for confirming. I'll see you at 2pm in the conference room.\n\nBest regards,\nSarah",
        "expected": "personal"
    },
    {
        "subject": "Special Offer - 50% Off Everything!",
        "content": "Don't miss out on our biggest sale of the year! Use code SAVE50 at checkout. This offer expires in 24 hours.\n\nShop now: https://example.com\n\nUnsubscribe: https://example.com/unsubscribe",
        "expected": "promotional"
    },
    {
        "subject": "Your order has shipped",
        "content": "Order #12345 has been shipped via FedEx. Tracking number: 1234567890.\n\nExpected delivery: Tomorrow\n\nThis is an automated email, please do not reply.",
        "expected": "automated"
    },
    {
        "subject": "Weekly newsletter",
        "content": "Here are this week's top stories:\n\n1. Tech News Update\n2. Market Analysis\n3. Product Reviews\n\nTo unsubscribe, click here.",
        "expected": "promotional"
    },
    {
        "subject": "Project discussion",
        "content": "Hey team,\n\nI've reviewed the proposal and have some feedback. Can we schedule a call this week to discuss?\n\nLet me know what works for your schedules.\n\nThanks!",
        "expected": "personal"
    }
]

def test_classifier():
    """Test the classifier with sample emails"""
    print("🧪 Testing SetFit Email Classifier\n")
    
    # Initialize classifier
    print("📥 Initializing classifier...")
    if not initialize_classifier():
        print("❌ Failed to initialize classifier")
        return False
    
    classifier = get_classifier()
    print(f"✅ Classifier initialized: {classifier.model_name}\n")
    
    # Test individual classifications
    correct = 0
    total = len(TEST_EMAILS)
    
    print("📧 Testing individual email classifications:")
    print("=" * 60)
    
    for i, test_email in enumerate(TEST_EMAILS, 1):
        print(f"\nTest {i}/{total}:")
        print(f"Subject: {test_email['subject']}")
        print(f"Content: {test_email['content'][:100]}...")
        print(f"Expected: {test_email['expected']}")
        
        try:
            result = classifier.classify_single(
                email_content=test_email['content'],
                subject=test_email['subject']
            )
            
            print(f"Predicted: {result.category} (confidence: {result.confidence:.2f})")
            print(f"Time: {result.processing_time_ms:.1f}ms")
            
            if result.category == test_email['expected']:
                print("✅ CORRECT")
                correct += 1
            else:
                print("❌ INCORRECT")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Results: {correct}/{total} correct ({correct/total*100:.1f}%)")
    
    # Test batch classification
    print("\n🚀 Testing batch classification:")
    try:
        contents = [email['content'] for email in TEST_EMAILS]
        subjects = [email['subject'] for email in TEST_EMAILS]
        
        results = classifier.classify_batch(contents, subjects)
        
        print(f"✅ Batch classified {len(results)} emails")
        for i, result in enumerate(results):
            expected = TEST_EMAILS[i]['expected']
            status = "✅" if result.category == expected else "❌"
            print(f"  {i+1}. {result.category} (expected: {expected}) {status}")
            
    except Exception as e:
        print(f"❌ Batch classification failed: {e}")
    
    # Performance stats
    print("\n📈 Performance Statistics:")
    stats = classifier.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return correct >= total * 0.6  # 60% accuracy threshold


def test_database_integration():
    """Test classification with actual database messages"""
    print("\n🗄️ Testing database integration...")
    
    try:
        from processor import EmailProcessor
        from database import get_db_session
        from models import Message
        
        processor = EmailProcessor()
        
        # Get some unclassified messages
        with get_db_session() as session:
            unclassified = session.query(Message).filter(
                Message.cleaned_at.isnot(None),
                Message.classified_at.is_(None)
            ).limit(3).all()
            
            if not unclassified:
                print("ℹ️ No unclassified messages found in database")
                return True
            
            print(f"📧 Found {len(unclassified)} unclassified messages")
            
            # Test classification
            for msg in unclassified:
                print(f"\nClassifying message {msg.id}:")
                print(f"  Subject: {msg.subject}")
                print(f"  From: {msg.from_email}")
                
                success = processor.classify_single_message(session, msg)
                if success:
                    print(f"  ✅ Classified as: {msg.category}")
                else:
                    print(f"  ❌ Classification failed")
            
            # Get classification stats
            stats = processor.get_classification_stats()
            print(f"\n📊 Classification Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database integration test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 Email Classification Test Suite")
    print("=" * 50)
    
    # Test 1: Classifier functionality
    classifier_ok = test_classifier()
    
    # Test 2: Database integration
    db_ok = test_database_integration()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"  Classifier: {'✅ PASS' if classifier_ok else '❌ FAIL'}")
    print(f"  Database:   {'✅ PASS' if db_ok else '❌ FAIL'}")
    
    if classifier_ok and db_ok:
        print("\n🎉 All tests passed! Classification system is ready.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Check the logs above.")
        return 1


if __name__ == "__main__":
    exit(main())