#!/usr/bin/env python3
"""
Email Classifier Health Check

Simple health check script for the email classification worker.
Checks if the worker is processing emails and if the classifier is responsive.
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from database import get_db_session
    from models import Message
    from sqlalchemy import desc, func
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure dependencies are installed: uv add sqlalchemy psycopg2-binary")
    sys.exit(1)


def check_recent_classifications():
    """Check if emails have been classified recently"""
    try:
        with get_db_session() as session:
            # Check for classifications in the last hour
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            
            recent_count = session.query(func.count(Message.id)).filter(
                Message.classified_at >= one_hour_ago
            ).scalar() or 0
            
            # Check for unclassified emails
            unclassified_count = session.query(func.count(Message.id)).filter(
                Message.category.is_(None)
            ).scalar() or 0
            
            # Get latest classification
            latest = session.query(Message).filter(
                Message.classified_at.isnot(None)
            ).order_by(desc(Message.classified_at)).first()
            
            return {
                "recent_classifications": recent_count,
                "unclassified_emails": unclassified_count,
                "latest_classification": latest.classified_at.isoformat() if latest else None,
                "database_accessible": True
            }
            
    except Exception as e:
        return {
            "database_accessible": False,
            "error": str(e)
        }


def check_classifier_health():
    """Check if the LLM classifier is working"""
    try:
        from llm_classifier import LLMEmailClassifier, initialize_llm_classifier
        
        if not initialize_llm_classifier():
            return {
                "classifier_ready": False,
                "error": "Failed to initialize classifier"
            }
        
        classifier = LLMEmailClassifier()
        if not classifier.initialize():
            return {
                "classifier_ready": False,
                "error": "Failed to initialize classifier instance"
            }
        
        # Quick test classification
        start_time = datetime.utcnow()
        result = classifier.classify_email(
            from_email="health-check@example.com",
            subject="Health check",
            body="This is a health check message"
        )
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "classifier_ready": True,
            "test_classification": result.category if result else "failed",
            "response_time_ms": response_time,
            "model": classifier.model_name
        }
        
    except Exception as e:
        return {
            "classifier_ready": False,
            "error": str(e)
        }


def main():
    """Main health check"""
    print("🔍 Email Classifier Health Check")
    print("=" * 50)
    
    # Check database and recent activity
    print("📊 Checking database and recent activity...")
    db_health = check_recent_classifications()
    
    if db_health.get("database_accessible"):
        print(f"✅ Database accessible")
        print(f"📧 Recent classifications (1h): {db_health['recent_classifications']}")
        print(f"⏳ Unclassified emails: {db_health['unclassified_emails']}")
        if db_health['latest_classification']:
            print(f"🕐 Latest classification: {db_health['latest_classification']}")
        else:
            print("⚠️  No classifications found")
    else:
        print(f"❌ Database error: {db_health.get('error')}")
        return 1
    
    print()
    
    # Check classifier
    print("🤖 Checking LLM classifier...")
    classifier_health = check_classifier_health()
    
    if classifier_health.get("classifier_ready"):
        print(f"✅ Classifier ready")
        print(f"🏷️  Test classification: {classifier_health.get('test_classification')}")
        print(f"⚡ Response time: {classifier_health.get('response_time_ms', 0):.0f}ms")
        print(f"🧠 Model: {classifier_health.get('model')}")
    else:
        print(f"❌ Classifier error: {classifier_health.get('error')}")
        return 1
    
    print()
    
    # Overall status
    if (db_health.get("database_accessible") and 
        classifier_health.get("classifier_ready")):
        print("🎉 Overall health: GOOD")
        
        # Warn if there are many unclassified emails
        unclassified = db_health.get("unclassified_emails", 0)
        if unclassified > 100:
            print(f"⚠️  Warning: {unclassified} unclassified emails (consider checking worker)")
        
        return 0
    else:
        print("💥 Overall health: PROBLEMS DETECTED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)