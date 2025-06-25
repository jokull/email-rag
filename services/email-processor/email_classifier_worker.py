#!/usr/bin/env python3
"""
Host-based Email Classification Queue Worker

This service runs on the host machine to classify emails using the enhanced LLM classifier
with first-line defense and correspondence intelligence. It uses a queue worker pattern:
1. Select unclassified emails from INBOX 
2. Classify using LLM Python library (with GPU acceleration)
3. Update database with results
4. Sleep and repeat

Usage:
    python email_classifier_worker.py

Environment Variables:
    DATABASE_URL - PostgreSQL connection string
    LLM_MODEL_NAME - LLM model to use (default: gpt-4o-mini)
    MY_EMAIL - Your email for correspondence analysis (default: jokull@solberg.is)
    BATCH_SIZE - Emails to process per batch (default: 10)
    SLEEP_INTERVAL - Seconds between batches (default: 30)
    MAILBOX_FILTER - Which mailbox to process (default: INBOX)
"""

import logging
import time
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from database import get_db_session
from models import Message, ImapMessage, ImapMailbox
from llm_classifier import LLMEmailClassifier, initialize_llm_classifier
from sqlalchemy import and_, desc, func
from sqlalchemy.orm import Session


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('email_classifier_worker.log')
    ]
)
logger = logging.getLogger(__name__)


class EmailClassificationWorker:
    """Host-based email classification queue worker"""
    
    def __init__(self):
        # Configuration from environment
        self.batch_size = int(os.getenv("BATCH_SIZE", "10"))
        self.sleep_interval = int(os.getenv("SLEEP_INTERVAL", "30"))
        self.mailbox_filter = os.getenv("MAILBOX_FILTER", "INBOX")
        
        # Statistics tracking
        self.total_processed = 0
        self.total_errors = 0
        self.start_time = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        
        # Initialize classifier
        logger.info("🚀 Starting Email Classification Worker")
        logger.info(f"📧 Processing {self.mailbox_filter} mailbox")
        logger.info(f"📦 Batch size: {self.batch_size}")
        logger.info(f"⏰ Sleep interval: {self.sleep_interval}s")
        
        # Initialize LLM classifier
        if not initialize_llm_classifier():
            raise RuntimeError("Failed to initialize LLM classifier")
        
        self.classifier = LLMEmailClassifier()
        if not self.classifier.initialize():
            raise RuntimeError("Failed to initialize classifier instance")
        
        logger.info("✅ Email Classification Worker ready")
    
    def get_unclassified_emails(self, session: Session) -> List[Message]:
        """Get batch of unclassified emails from specified mailbox"""
        query = session.query(Message).join(
            ImapMessage, Message.imap_message_id == ImapMessage.id
        ).join(
            ImapMailbox, ImapMessage.mailbox_id == ImapMailbox.id
        ).filter(
            and_(
                ImapMailbox.name == self.mailbox_filter,
                Message.category.is_(None),  # Unclassified
                Message.from_email.isnot(None),
                Message.subject.isnot(None),
                Message.body_text.isnot(None)
            )
        ).order_by(desc(Message.date_sent)).limit(self.batch_size)
        
        return query.all()
    
    def classify_single_email(self, session: Session, message: Message) -> bool:
        """Classify a single email and update database"""
        try:
            # Classify using enhanced LLM classifier
            result = self.classifier.classify_email(
                from_email=message.from_email,
                subject=message.subject or "",
                body=message.body_text or ""
            )
            
            if not result:
                logger.warning(f"❌ No classification result for message {message.id}")
                return False
            
            # Update message with classification
            message.category = result.category
            message.confidence = result.confidence
            message.classified_at = datetime.utcnow()
            
            session.commit()
            
            # Log classification with processing time
            logger.info(f"✅ Classified {message.id}: {message.from_email[:30]} -> {result.category} "
                       f"({result.confidence:.2f}) in {result.processing_time_ms:.0f}ms")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to classify message {message.id}: {e}")
            session.rollback()
            
            # Mark as failed for debugging
            try:
                message.category = "error"
                message.confidence = 0.0
                message.classified_at = datetime.utcnow()
                session.commit()
            except Exception as commit_error:
                logger.error(f"Failed to mark classification error: {commit_error}")
                session.rollback()
            
            return False
    
    def process_classification_batch(self) -> int:
        """Process a batch of unclassified emails"""
        processed_count = 0
        
        with get_db_session() as session:
            # Get unclassified emails
            unclassified_emails = self.get_unclassified_emails(session)
            
            if not unclassified_emails:
                logger.debug(f"No unclassified emails found in {self.mailbox_filter}")
                return 0
            
            logger.info(f"📧 Processing batch of {len(unclassified_emails)} emails")
            
            # Process each email
            for email in unclassified_emails:
                if self.classify_single_email(session, email):
                    processed_count += 1
                    self.total_processed += 1
                else:
                    self.total_errors += 1
                
                # Update last activity
                self.last_activity = datetime.utcnow()
        
        return processed_count
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        uptime = datetime.utcnow() - self.start_time
        classifier_stats = self.classifier.get_stats()
        
        return {
            "worker_status": "running",
            "uptime_seconds": int(uptime.total_seconds()),
            "total_processed": self.total_processed,
            "total_errors": self.total_errors,
            "last_activity": self.last_activity.isoformat(),
            "configuration": {
                "batch_size": self.batch_size,
                "sleep_interval": self.sleep_interval,
                "mailbox_filter": self.mailbox_filter
            },
            "classifier_stats": classifier_stats
        }
    
    def log_periodic_stats(self):
        """Log worker statistics periodically"""
        stats = self.get_worker_stats()
        uptime_hours = stats["uptime_seconds"] // 3600
        
        logger.info(f"📊 Worker Stats: {self.total_processed} processed, "
                   f"{self.total_errors} errors, {uptime_hours}h uptime")
        
        # Log classifier performance
        classifier_stats = stats["classifier_stats"]
        if classifier_stats.get("total_classifications", 0) > 0:
            avg_time = classifier_stats.get("average_time_ms", 0)
            rate = classifier_stats.get("classifications_per_second", 0)
            logger.info(f"🚀 Classifier: {avg_time:.0f}ms avg, {rate:.1f}/sec")
    
    def run(self):
        """Main worker loop"""
        logger.info("🔄 Starting classification worker loop")
        
        stats_log_interval = 300  # Log stats every 5 minutes
        last_stats_log = time.time()
        
        try:
            while True:
                batch_start = time.time()
                
                # Process a batch of emails
                processed_count = self.process_classification_batch()
                
                # Log batch results
                batch_time = time.time() - batch_start
                if processed_count > 0:
                    logger.info(f"📦 Batch complete: {processed_count} emails in {batch_time:.1f}s")
                
                # Periodic stats logging
                if time.time() - last_stats_log > stats_log_interval:
                    self.log_periodic_stats()
                    last_stats_log = time.time()
                
                # Sleep between batches
                logger.debug(f"😴 Sleeping for {self.sleep_interval}s")
                time.sleep(self.sleep_interval)
                
        except KeyboardInterrupt:
            logger.info("🛑 Received interrupt signal, shutting down gracefully")
        except Exception as e:
            logger.error(f"💥 Worker crashed: {e}")
            raise
        finally:
            # Log final stats
            self.log_periodic_stats()
            logger.info("👋 Email Classification Worker stopped")


def main():
    """Main entry point"""
    try:
        worker = EmailClassificationWorker()
        worker.run()
    except Exception as e:
        logger.error(f"Failed to start worker: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()