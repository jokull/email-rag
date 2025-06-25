#!/usr/bin/env python3
"""
Email Classification Worker
Continuously polls for cleaned emails and classifies them using SetFit model
"""

import asyncio
import logging
import os
import time
from datetime import datetime

from processor import EmailProcessor
from setfit_classifier import initialize_classifier, get_classifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClassificationWorker:
    """Background worker for email classification"""
    
    def __init__(self):
        self.processor = EmailProcessor()
        self.running = False
        
        # Configuration from environment
        self.poll_interval = int(os.getenv('POLL_INTERVAL', '30'))  # seconds
        self.batch_size = int(os.getenv('BATCH_SIZE', '50'))
        
        logger.info(f"🏷️ Classification worker initialized:")
        logger.info(f"   Poll interval: {self.poll_interval}s")
        logger.info(f"   Batch size: {self.batch_size}")
    
    async def start(self):
        """Start the classification worker"""
        logger.info("🚀 Starting email classification worker...")
        
        # Initialize classifier
        logger.info("📥 Initializing SetFit classifier...")
        if not initialize_classifier():
            logger.error("❌ Failed to initialize classifier, exiting")
            return
        
        classifier = get_classifier()
        logger.info(f"✅ Classifier ready: {classifier.model_name}")
        
        self.running = True
        
        # Main processing loop
        try:
            while self.running:
                await self._process_batch()
                await asyncio.sleep(self.poll_interval)
                
        except KeyboardInterrupt:
            logger.info("👋 Received interrupt, shutting down...")
        except Exception as e:
            logger.error(f"❌ Worker crashed: {e}")
        finally:
            self.running = False
            logger.info("🛑 Classification worker stopped")
    
    async def _process_batch(self):
        """Process a batch of unclassified emails"""
        try:
            start_time = time.time()
            
            # Process unclassified emails
            classified_count = self.processor.process_unclassified_emails(limit=self.batch_size)
            
            processing_time = time.time() - start_time
            
            if classified_count > 0:
                rate = classified_count / processing_time if processing_time > 0 else 0
                logger.info(f"📧 Classified {classified_count} emails in {processing_time:.1f}s ({rate:.1f} emails/sec)")
                
                # Log classification statistics
                stats = self.processor.get_classification_stats()
                logger.info(f"📊 Classification progress: {stats['classification_rate']} "
                          f"({stats['pending_classification']} pending)")
                
                # Log category breakdown
                categories = stats.get('category_breakdown', {})
                if categories:
                    category_summary = ", ".join([f"{cat}: {count}" for cat, count in categories.items()])
                    logger.info(f"🏷️ Categories: {category_summary}")
            else:
                logger.debug("😴 No unclassified emails found")
                
        except Exception as e:
            logger.error(f"❌ Error processing batch: {e}")
    
    def stop(self):
        """Stop the worker"""
        self.running = False


async def main():
    """Main entry point"""
    logger.info("🏷️ Email Classification Worker Starting...")
    
    worker = ClassificationWorker()
    
    try:
        await worker.start()
    except Exception as e:
        logger.error(f"❌ Worker failed to start: {e}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())