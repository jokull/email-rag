#!/usr/bin/env python3
"""
Email processing worker - Simple polling loop that processes emails from imap_messages
"""

import time
import logging
import os
import signal
import sys
from datetime import datetime
from typing import Optional

from processor import EmailProcessor
from database import get_db_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmailWorker:
    """Simple email processing worker with polling"""
    
    def __init__(self):
        self.processor = EmailProcessor()
        self.running = True
        self.poll_interval = int(os.getenv("POLL_INTERVAL", "30"))  # seconds
        self.batch_size = int(os.getenv("BATCH_SIZE", "50"))
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Email worker initialized with poll_interval={self.poll_interval}s, batch_size={self.batch_size}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def run(self):
        """Main worker loop"""
        logger.info("Starting email processing worker...")
        
        while self.running:
            try:
                # Process a batch of emails
                processed_count = self.processor.process_unprocessed_emails(limit=self.batch_size)
                
                if processed_count > 0:
                    logger.info(f"Processed {processed_count} emails in this batch")
                    
                    # If we processed a full batch, don't wait - there might be more
                    if processed_count >= self.batch_size:
                        logger.info("Full batch processed, checking for more emails immediately...")
                        continue
                else:
                    logger.debug("No emails to process")
                
                # Wait before next poll
                self._sleep_with_interrupt_check(self.poll_interval)
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                # Wait a bit longer on error to avoid tight error loops
                self._sleep_with_interrupt_check(min(self.poll_interval * 2, 60))
        
        logger.info("Email processing worker stopped")
    
    def _sleep_with_interrupt_check(self, duration: int):
        """Sleep for duration seconds, but check for shutdown signal periodically"""
        start_time = time.time()
        while time.time() - start_time < duration and self.running:
            time.sleep(min(1, duration - (time.time() - start_time)))
    
    def health_check(self) -> bool:
        """Check if worker is healthy"""
        try:
            with get_db_session() as session:
                # Simple database connection test
                from sqlalchemy import text
                session.execute(text("SELECT 1")).fetchone()
                return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

def main():
    """Main entry point"""
    logger.info("Email processing worker starting up...")
    
    # Environment validation
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable is required")
        sys.exit(1)
    
    # Create and run worker
    worker = EmailWorker()
    
    try:
        # Initial health check
        if not worker.health_check():
            logger.error("Initial health check failed, exiting")
            sys.exit(1)
        
        # Get initial stats
        stats = worker.processor.get_processing_stats()
        logger.info(f"Worker starting with stats: {stats}")
        
        # Run the worker
        worker.run()
        
    except Exception as e:
        logger.error(f"Worker failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()