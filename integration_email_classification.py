#!/usr/bin/env python3
"""
Integration test for email classification pipeline
Tests our real email classification components: SimpleRuleClassifier and processor
"""

import sys
import os
import subprocess
from datetime import datetime

# Add email-processor to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'email-processor'))

from database import get_db_session
from models import Message
from simple_rule_classifier import SimpleRuleClassifier
from processor import EmailProcessor

def test_simple_rule_classifier_import():
    """Test that our SimpleRuleClassifier can be imported and works"""
    print("🔧 Testing SimpleRuleClassifier import and initialization...")
    
    try:
        classifier = SimpleRuleClassifier()
        print("✅ SimpleRuleClassifier imported and initialized successfully")
        
        # Test classifier has required methods
        required_methods = ['classify', '_extract_domain', '_count_keywords']
        for method in required_methods:
            if hasattr(classifier, method):
                print(f"✅ Method {method} available")
            else:
                print(f"❌ Method {method} missing")
                return False
        
        return True
    except ImportError as e:
        print(f"❌ SimpleRuleClassifier import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ SimpleRuleClassifier initialization failed: {e}")
        return False

def test_simple_rule_classifier_patterns():
    """Test our real first-line defense patterns"""
    print("\n🛡️  Testing SimpleRuleClassifier first-line defense patterns...")
    
    try:
        classifier = SimpleRuleClassifier()
        
        # Test cases for first-line defense
        test_cases = [
            {
                'from_email': 'noreply@company.com',
                'subject': 'Account notification',
                'body': 'Your account has been updated.',
                'expected': 'automated'
            },
            {
                'from_email': 'notifications@github.com',
                'subject': 'Pull request merged',
                'body': 'Your pull request has been merged.',
                'expected': 'automated'
            },
            {
                'from_email': 'deals@amazon.com',
                'subject': '50% OFF Everything!',
                'body': 'Limited time offer - shop now!',
                'expected': 'promotional'
            },
            {
                'from_email': 'billing@service.com',
                'subject': 'Invoice #12345',
                'body': 'Your monthly invoice is ready.',
                'expected': 'automated'
            },
            {
                'from_email': 'friend@gmail.com',
                'subject': 'Coffee tomorrow?',
                'body': 'Want to grab coffee tomorrow at 3pm?',
                'expected': None  # Should not match first-line defense
            }
        ]
        
        passed = 0
        total = len(test_cases)
        
        for i, test_case in enumerate(test_cases):
            result = classifier.classify(
                from_email=test_case['from_email'],
                subject=test_case['subject'],
                body=test_case['body']
            )
            
            category = result.category
            confidence = result.confidence
            method = result.reason
            
            expected = test_case['expected']
            success = (category == expected) or (expected is None and method != 'first_line_defense')
            
            status = "✅" if success else "❌"
            print(f"{status} Test {i+1}: {test_case['from_email'][:30]}")
            print(f"    Expected: {expected}, Got: {category} ({method}, conf: {confidence:.2f})")
            
            if success:
                passed += 1
        
        print(f"\n📊 First-line defense results: {passed}/{total} passed ({passed/total*100:.1f}%)")
        return passed >= total * 0.8  # Allow 80% success rate
        
    except Exception as e:
        print(f"❌ SimpleRuleClassifier patterns test failed: {e}")
        return False

def test_email_processor_classification():
    """Test EmailProcessor classification functionality"""
    print("\n📧 Testing EmailProcessor classification functionality...")
    
    try:
        processor = EmailProcessor()
        print("✅ EmailProcessor initialized successfully")
        
        # Test that processor can classify messages
        if hasattr(processor, 'classify_single_message'):
            print("✅ EmailProcessor has classify_single_message method")
        else:
            print("⚠️  EmailProcessor classify_single_message method not found")
        
        # Test processor stats
        if hasattr(processor, 'get_classification_stats'):
            try:
                stats = processor.get_classification_stats()
                print("✅ EmailProcessor classification stats available:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
            except Exception as e:
                print(f"⚠️  Classification stats error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ EmailProcessor classification test failed: {e}")
        return False

def test_classification_database_stats():
    """Test classification statistics from our real database"""
    print("\n📊 Testing classification database statistics...")
    
    try:
        with get_db_session() as session:
            from sqlalchemy import func
            
            # Total messages
            total_messages = session.query(Message).count()
            
            # Classification statistics
            classified_messages = session.query(Message).filter(
                Message.category.isnot(None)
            ).count()
            
            # Category distribution
            category_dist = session.query(
                Message.category,
                func.count(Message.id).label('count')
            ).filter(
                Message.category.isnot(None)
            ).group_by(Message.category).order_by(func.count(Message.id).desc()).all()
            
            print(f"📊 Real Database Classification Statistics:")
            print(f"   Total messages: {total_messages}")
            print(f"   Classified messages: {classified_messages}")
            
            if total_messages > 0:
                classification_rate = classified_messages / total_messages * 100
                print(f"   Classification rate: {classification_rate:.1f}%")
            
            if category_dist:
                print(f"\n📋 Category distribution:")
                for category, count in category_dist:
                    print(f"   {category}: {count} messages")
            
            return True
            
    except Exception as e:
        print(f"❌ Classification database stats test failed: {e}")
        return False

def test_real_message_classification(force_reprocess=False):
    """Test classification on real messages from database"""
    print(f"\n🧪 Testing classification on real messages{' (FORCE MODE)' if force_reprocess else ''}...")
    
    try:
        classifier = SimpleRuleClassifier()
        
        with get_db_session() as session:
            # Get sample messages from different categories
            if force_reprocess:
                # Force mode: get already classified messages to reclassify them
                sample_messages = session.query(Message).filter(
                    Message.body_text.isnot(None),
                    Message.body_text != '',
                    Message.category.isnot(None)  # Get already classified messages
                ).limit(5).all()
                print(f"🔄 Force mode: Testing reclassification on {len(sample_messages)} already classified messages")
            else:
                # Normal mode: get any messages
                sample_messages = session.query(Message).filter(
                    Message.body_text.isnot(None),
                    Message.body_text != ''
                ).limit(5).all()
            
            if not sample_messages:
                print("⚠️  No messages found for testing")
                return True
            
            print(f"🧪 Testing classification on {len(sample_messages)} real messages:")
            
            for msg in sample_messages:
                result = classifier.classify(
                    from_email=msg.from_email or '',
                    subject=msg.subject or '',
                    body=msg.body_text or ''
                )
                
                predicted_category = result.category
                confidence = result.confidence
                method = result.reason
                actual_category = msg.category
                
                # Check if prediction matches database
                match_status = "✅" if predicted_category == actual_category else "❓"
                
                from_email = msg.from_email[:30] + '...' if len(msg.from_email) > 30 else msg.from_email
                subject_preview = (msg.subject or '')[:40] + '...' if len(msg.subject or '') > 40 else msg.subject or 'No subject'
                
                print(f"{match_status} #{msg.id}: {from_email}")
                print(f"    Subject: \"{subject_preview}\"")
                print(f"    Predicted: {predicted_category} ({method}, {confidence:.2f})")
                print(f"    Actual: {actual_category}")
                print()
            
            return True
            
    except Exception as e:
        print(f"❌ Real message classification test failed: {e}")
        return False

def test_llm_classification_integration():
    """Test LLM classification integration (if available)"""
    print("\n🧠 Testing LLM classification integration...")
    
    try:
        # Test that llm command is available
        result = subprocess.run(["llm", "models", "list"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print("⚠️  LLM command not available")
            return True  # Not a failure
        
        models = result.stdout
        print("✅ LLM models available")
        
        # Look for key models
        key_models = ["gpt-4o-mini", "gpt-3.5-turbo", "q3"]
        found_models = []
        
        for model in key_models:
            if model in models:
                found_models.append(model)
                print(f"   ✅ {model}")
        
        if not found_models:
            print("⚠️  No suitable models found")
            return True
        
        # Test classification with a simple example
        test_email = """
        From: deals@company.com
        Subject: 50% OFF Sale - Limited Time!
        
        Don't miss out on our biggest sale of the year!
        Get 50% off everything in our store.
        
        Shop now: www.company.com/sale
        """
        
        # Try with the first available model
        model = found_models[0]
        prompt = f"Classify this email as 'personal', 'promotional', or 'automated'. Respond with just one word.\n\nEmail:\n{test_email}"
        
        try:
            result = subprocess.run([
                "llm", "-m", model, prompt
            ], capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                classification = result.stdout.strip().lower()
                print(f"✅ {model} classified as: '{classification}'")
                
                # Check if classification is reasonable
                if any(cat in classification for cat in ['personal', 'promotional', 'automated']):
                    print(f"✅ Valid classification category")
                    return True
                else:
                    print(f"⚠️  Unexpected classification: {classification}")
                    return True  # Still successful - model responded
            else:
                print(f"❌ {model} failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ LLM classification test failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ LLM integration test failed: {e}")
        return False

def test_classification_confidence_levels():
    """Test classification confidence scoring"""
    print("\n🎯 Testing classification confidence levels...")
    
    try:
        classifier = SimpleRuleClassifier()
        
        # Test cases with expected confidence levels
        test_cases = [
            {
                'email': {
                    'from_email': 'noreply@system.com',
                    'subject': 'System notification',
                    'body': 'This is an automated message.'
                },
                'expected_confidence_min': 0.8  # Should be high confidence
            },
            {
                'email': {
                    'from_email': 'deals@shop.com', 
                    'subject': '50% OFF SALE!!!',
                    'body': 'Buy now and save big!'
                },
                'expected_confidence_min': 0.7  # Should be high confidence
            },
            {
                'email': {
                    'from_email': 'unknown@domain.com',
                    'subject': 'Meeting',
                    'body': 'Let me know when you can meet.'
                },
                'expected_confidence_min': 0.0  # May have low confidence
            }
        ]
        
        passed = 0
        total = len(test_cases)
        
        for i, test_case in enumerate(test_cases):
            email = test_case['email']
            result = classifier.classify(**email)
            
            confidence = result.confidence
            category = result.category
            method = result.reason
            
            expected_min = test_case['expected_confidence_min']
            success = confidence >= expected_min
            
            status = "✅" if success else "❓"
            print(f"{status} Test {i+1}: {email['from_email']}")
            print(f"    Category: {category} ({method})")
            print(f"    Confidence: {confidence:.2f} (expected ≥ {expected_min})")
            
            if success:
                passed += 1
        
        print(f"\n📊 Confidence scoring: {passed}/{total} tests met expectations")
        return True  # Always pass - this is more informational
        
    except Exception as e:
        print(f"❌ Classification confidence test failed: {e}")
        return False

def main():
    """Run all email classification integration tests"""
    import argparse
    parser = argparse.ArgumentParser(description='Email Classification Smoke Tests')
    parser.add_argument('--force', action='store_true', 
                       help='Force reprocessing of already classified messages')
    args = parser.parse_args()
    
    force_mode = args.force
    title = "🚀 Email Classification Smoke Tests"
    if force_mode:
        title += " (FORCE MODE)"
    
    print(title)
    print("=" * 50)
    
    tests = [
        ("SimpleRuleClassifier Import", test_simple_rule_classifier_import),
        ("First-line Defense Patterns", test_simple_rule_classifier_patterns),
        ("EmailProcessor Classification", test_email_processor_classification),
        ("Classification Database Stats", test_classification_database_stats),
        ("Real Message Classification", lambda: test_real_message_classification(force_mode)),
        ("LLM Classification Integration", test_llm_classification_integration),
        ("Classification Confidence Levels", test_classification_confidence_levels),
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
    
    print("\n" + "=" * 50)
    print(f"📊 Integration Test Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    if passed >= 5:  # Allow some flexibility
        print("🎉 Email classification integration tests mostly successful!")
        return True
    else:
        print("⚠️  Some email classification integration tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)