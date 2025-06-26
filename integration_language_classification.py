#!/usr/bin/env python3
"""
Integration test for language classification pipeline
Tests that language detection works with real emails from the database
"""

import sys
import os
from datetime import datetime

# Add email-processor to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'email-processor'))

from database import get_db_session
from models import Message
from language_classifier import LanguageClassifier

def test_language_classifier_initialization():
    """Test that language classifier initializes correctly"""
    print("🔧 Testing language classifier initialization...")
    
    try:
        classifier = LanguageClassifier()
        print("✅ Language classifier initialized successfully")
        
        # Test supported languages
        languages = classifier.get_supported_languages()
        print(f"📋 Supports {len(languages)} languages: {list(languages.keys())[:10]}...")
        
        return True
    except Exception as e:
        print(f"❌ Language classifier initialization failed: {e}")
        return False

def test_language_detection_samples():
    """Test language detection with sample texts"""
    print("\n🧪 Testing language detection with sample texts...")
    
    classifier = LanguageClassifier()
    
    test_cases = [
        ("Hello, how are you doing today? I hope everything is going well and you're having a great time.", "en", "English"),
        ("Bonjour, comment allez-vous aujourd'hui? J'espère que tout va bien pour vous.", "fr", "French"),
        ("Hola, ¿cómo estás hoy? Espero que todo esté bien y que tengas un buen día.", "es", "Spanish"),
        ("Hallo, wie geht es dir heute? Ich hoffe, dass alles gut läuft und du einen schönen Tag hast.", "de", "German"),
        ("Ciao, come stai oggi? Spero che tutto vada bene e che tu stia passando una bella giornata.", "it", "Italian"),
        ("Halló! Hvað segirðu gott í dag? Ég vona að allt sé í lagi og þú hafir góðan dag.", "is", "Icelandic"),
    ]
    
    passed = 0
    total = len(test_cases)
    
    for text, expected_code, expected_name in test_cases:
        result = classifier.classify_email_content(body=text)
        
        detected_code = result.get('language_code')
        detected_name = result.get('language_name')
        confidence = result.get('confidence', 0)
        
        success = detected_code == expected_code
        status = "✅" if success else "❌"
        
        print(f"{status} {expected_name}: detected '{detected_name}' ({detected_code}) confidence: {confidence:.3f}")
        
        if success:
            passed += 1
    
    print(f"\n📊 Language detection results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    return passed == total

def test_database_language_integration():
    """Test language classification integration with database"""
    print("\n🗄️  Testing database integration...")
    
    try:
        with get_db_session() as session:
            # Count messages with language data
            total_messages = session.query(Message).count()
            messages_with_language = session.query(Message).filter(Message.language.isnot(None)).count()
            
            print(f"📊 Database stats:")
            print(f"   Total messages: {total_messages}")
            print(f"   Messages with language: {messages_with_language}")
            
            if total_messages > 0:
                detection_rate = messages_with_language / total_messages * 100
                print(f"   Detection rate: {detection_rate:.1f}%")
            
            # Get sample of classified messages
            sample_messages = session.query(Message).filter(
                Message.language.isnot(None),
                Message.language_confidence.isnot(None)
            ).limit(5).all()
            
            if sample_messages:
                print(f"\n📧 Sample classified messages:")
                for msg in sample_messages:
                    confidence = msg.language_confidence or 0
                    body_preview = (msg.body_text or '')[:50] + '...' if len(msg.body_text or '') > 50 else msg.body_text or ''
                    print(f"   #{msg.id}: {msg.language} ({confidence:.3f}) - \"{body_preview}\"")
                
                return True
            else:
                print("⚠️  No classified messages found in database")
                return False
                
    except Exception as e:
        print(f"❌ Database integration test failed: {e}")
        return False

def test_real_email_classification():
    """Test classification on real emails from database"""
    print("\n📧 Testing real email classification...")
    
    try:
        classifier = LanguageClassifier()
        
        with get_db_session() as session:
            # Get unclassified messages for testing
            unclassified = session.query(Message).filter(
                Message.language.is_(None),
                Message.body_text.isnot(None),
                Message.body_text != ''
            ).limit(3).all()
            
            if not unclassified:
                print("⚠️  No unclassified messages found for testing")
                # Get some classified messages to verify they still work
                sample_messages = session.query(Message).filter(
                    Message.body_text.isnot(None),
                    Message.body_text != ''
                ).limit(3).all()
                unclassified = sample_messages
            
            if not unclassified:
                print("❌ No messages found in database")
                return False
            
            print(f"🧪 Testing classification on {len(unclassified)} real emails:")
            
            for msg in unclassified:
                result = classifier.classify_email_content(
                    subject=msg.subject or '',
                    body=msg.body_text or ''
                )
                
                language_code = result.get('language_code')
                language_name = result.get('language_name')
                confidence = result.get('confidence', 0)
                confidence_level = result.get('confidence_level', 'unknown')
                
                from_email = msg.from_email[:30] + '...' if len(msg.from_email) > 30 else msg.from_email
                subject_preview = (msg.subject or '')[:40] + '...' if len(msg.subject or '') > 40 else msg.subject or ''
                
                status = "✅" if language_code else "❓"
                print(f"{status} #{msg.id} from {from_email}")
                print(f"    Subject: \"{subject_preview}\"")
                if language_code:
                    print(f"    Language: {language_name} ({language_code}) - {confidence:.3f} ({confidence_level})")
                else:
                    print(f"    Language: Not detected")
                print()
            
            return True
            
    except Exception as e:
        print(f"❌ Real email classification test failed: {e}")
        return False

def test_language_statistics():
    """Test language statistics from database"""
    print("\n📈 Testing language statistics...")
    
    try:
        with get_db_session() as session:
            from sqlalchemy import func
            
            # Language distribution
            language_dist = session.query(
                Message.language, 
                func.count(Message.id).label('count')
            ).filter(
                Message.language.isnot(None)
            ).group_by(Message.language).order_by(func.count(Message.id).desc()).limit(10).all()
            
            if language_dist:
                print("🌍 Language distribution (top 10):")
                for lang, count in language_dist:
                    print(f"   {lang}: {count} messages")
            
            # Average confidence by language
            avg_confidence = session.query(
                Message.language,
                func.avg(Message.language_confidence).label('avg_conf')
            ).filter(
                Message.language.isnot(None),
                Message.language_confidence.isnot(None)
            ).group_by(Message.language).order_by(func.avg(Message.language_confidence).desc()).limit(5).all()
            
            if avg_confidence:
                print("\n🎯 Average confidence by language (top 5):")
                for lang, avg_conf in avg_confidence:
                    print(f"   {lang}: {avg_conf:.3f}")
            
            return True
            
    except Exception as e:
        print(f"❌ Language statistics test failed: {e}")
        return False

def main():
    """Run all language classification integration tests"""
    print("🚀 Language Classification Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Language Classifier Initialization", test_language_classifier_initialization),
        ("Language Detection Samples", test_language_detection_samples),
        ("Database Integration", test_database_language_integration),
        ("Real Email Classification", test_real_email_classification),
        ("Language Statistics", test_language_statistics),
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
    
    if passed == total:
        print("🎉 All language classification integration tests passed!")
        return True
    else:
        print("⚠️  Some language classification integration tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)